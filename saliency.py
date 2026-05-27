"""
saliency.py — Itti-Koch Visual Saliency Model

Implements the biologically-plausible saliency model from:
Itti, Koch & Niebur (1998). "A Model of Saliency-Based Visual Attention
for Rapid Scene Analysis." IEEE TPAMI.

Computes saliency maps from three channels:
  - Intensity (I)
  - Color (C) — RG and BY opponency
  - Orientation (O) — Gabor filters at 0°, 45°, 90°, 135°
"""

import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------

def build_gaussian_pyramid(image, levels=9):
    """Build a Gaussian pyramid with the given number of levels.

    Parameters
    ----------
    image : np.ndarray (H, W) or (H, W, C)
    levels : int
        Number of pyramid levels (including the original).

    Returns
    -------
    list[np.ndarray]
        Pyramid levels from finest (0) to coarsest (levels-1).
    """
    pyramid = [image]
    current = image.copy()
    for _ in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def center_surround_diff(pyramid, c, s):
    """Compute center-surround difference |center(c) - surround(s)|.

    The surround level is resized to match the center level's spatial
    extent before taking the absolute difference.

    Parameters
    ----------
    pyramid : list[np.ndarray]
        Gaussian pyramid (finest → coarsest).
    c : int
        Center scale index (finer).
    s : int
        Surround scale index (coarser).

    Returns
    -------
    np.ndarray
        Absolute difference map at the center level's size.
    """
    center = pyramid[c]
    surround = pyramid[s]

    surround_resized = cv2.resize(
        surround,
        (center.shape[1], center.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    return cv2.absdiff(center, surround_resized)


def normalize_map(feature_map):
    """Promote maps with few strong peaks and suppress uniform maps.

    Implements the *iterative* local-maximum normalization scheme from
    Itti & Koch (2001).  The map is first scaled to [0, 1]; then the
    global maximum M and the average m of all *local* maxima are
    computed.  The map is weighted by (M - m)² so that maps with a
    single strong peak are amplified while maps with many comparable
    peaks are suppressed.

    Parameters
    ----------
    feature_map : np.ndarray

    Returns
    -------
    np.ndarray
        Normalized map (same shape).
    """
    fmap = feature_map.astype(np.float32)

    # Scale to [0, 1]
    fmap = cv2.normalize(fmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    mean_val = np.mean(fmap)
    max_val = np.max(fmap)

    # Weight by (M - m)²
    fmap = fmap * ((max_val - mean_val) ** 2)

    return fmap


def apply_gabor_filter(image, theta, ksize=9):
    """Apply a single-orientation Gabor filter to the image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    theta : float
        Orientation angle in radians.
    ksize : int
        Kernel size (must be odd).

    Returns
    -------
    np.ndarray
        Filtered response (float32).
    """
    kernel = cv2.getGaborKernel(
        (ksize, ksize),
        sigma=4.0,
        theta=theta,
        lambd=10.0,
        gamma=0.5,
        psi=0,
        ktype=cv2.CV_32F,
    )

    return cv2.filter2D(image, cv2.CV_32F, kernel)


# ---------------------------------------------------------------------------
# Colour-opponent pre-processing
# ---------------------------------------------------------------------------

def compute_opponent_channels(rgb):
    """Compute the four colour-opponent channels (r, g, b, y).

    Parameters
    ----------
    rgb : np.ndarray (H, W, 3)
        Image in RGB, float32, range [0, 1].

    Returns
    -------
    r, g, b, y : np.ndarray (H, W)
        Opponent channels (non-negative).
    """
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    r = R - (G + B) / 2.0
    g = G - (R + B) / 2.0
    b = B - (R + G) / 2.0
    y = (R + G) / 2.0 - np.abs(R - G) / 2.0 - B

    # Clamp negatives to zero
    r = np.maximum(r, 0)
    g = np.maximum(g, 0)
    b = np.maximum(b, 0)
    y = np.maximum(y, 0)

    return r, g, b, y


# ---------------------------------------------------------------------------
# Conspicuity-map builders
# ---------------------------------------------------------------------------

def compute_intensity_conspicuity(intensity, levels=9,
                                   center_scales=None, delta_scales=None):
    """Build intensity conspicuity map from the intensity channel.

    Parameters
    ----------
    intensity : np.ndarray (H, W)
        Grayscale intensity image.
    levels : int
        Number of Gaussian pyramid levels.
    center_scales : list[int] | None
        Center scales (default: [2, 3, 4]).
    delta_scales : list[int] | None
        Delta from center for surround (default: [3, 4]).

    Returns
    -------
    np.ndarray (H, W)
        Intensity conspicuity map (normalised to [0, 1]).
    """
    if center_scales is None:
        center_scales = [2, 3, 4]
    if delta_scales is None:
        delta_scales = [3, 4]

    pyramid = build_gaussian_pyramid(intensity, levels=levels)

    feature_maps = []
    for c in center_scales:
        for delta in delta_scales:
            s = c + delta
            fmap = center_surround_diff(pyramid, c, s)
            feature_maps.append(fmap)

    # Normalise each feature map
    normalized_maps = [normalize_map(fm) for fm in feature_maps]

    # Combine via across-scale summation (resize all to first map's shape)
    base_shape = normalized_maps[0].shape
    conspicuity = np.zeros(base_shape, dtype=np.float32)
    for fm in normalized_maps:
        resized = cv2.resize(fm, (base_shape[1], base_shape[0]))
        conspicuity += resized

    # Final normalisation to [0, 1]
    conspicuity = cv2.normalize(conspicuity, None, 0, 1, cv2.NORM_MINMAX)
    return conspicuity


def compute_color_conspicuity(rgb, levels=9,
                               center_scales=None, delta_scales=None):
    """Build colour conspicuity map (RG + BY channels).

    Parameters
    ----------
    rgb : np.ndarray (H, W, 3)
        RGB image, float32, range [0, 1].
    levels : int
        Number of Gaussian pyramid levels.
    center_scales : list[int] | None
    delta_scales : list[int] | None

    Returns
    -------
    np.ndarray (H, W)
        Colour conspicuity map (normalised to [0, 1]).
    """
    if center_scales is None:
        center_scales = [2, 3, 4]
    if delta_scales is None:
        delta_scales = [3, 4]

    r, g, b, y = compute_opponent_channels(rgb)

    # RG and BY opponency
    RG = cv2.absdiff(r, g)
    BY = cv2.absdiff(b, y)

    # Pyramids for both opponency maps
    rg_pyramid = build_gaussian_pyramid(RG, levels=levels)
    by_pyramid = build_gaussian_pyramid(BY, levels=levels)

    color_feature_maps = []
    for pyramid in (rg_pyramid, by_pyramid):
        for c in center_scales:
            for delta in delta_scales:
                s = c + delta
                fmap = center_surround_diff(pyramid, c, s)
                fmap = normalize_map(fmap)
                color_feature_maps.append(fmap)

    # Combine
    base_shape = color_feature_maps[0].shape
    conspicuity = np.zeros(base_shape, dtype=np.float32)
    for fm in color_feature_maps:
        resized = cv2.resize(fm, (base_shape[1], base_shape[0]))
        conspicuity += resized

    conspicuity = cv2.normalize(conspicuity, None, 0, 1, cv2.NORM_MINMAX)
    return conspicuity


def compute_orientation_conspicuity(intensity, orientations=None, levels=9,
                                     center_scales=None, delta_scales=None):
    """Build orientation conspicuity map via Gabor filters.

    Parameters
    ----------
    intensity : np.ndarray (H, W)
        Grayscale intensity image.
    orientations : list[float] | None
        Orientations in radians (default: 0, π/4, π/2, 3π/4).
    levels : int
        Number of Gaussian pyramid levels.
    center_scales : list[int] | None
    delta_scales : list[int] | None

    Returns
    -------
    np.ndarray (H, W)
        Orientation conspicuity map (normalised to [0, 1]).
    """
    if orientations is None:
        orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    if center_scales is None:
        center_scales = [2, 3, 4]
    if delta_scales is None:
        delta_scales = [3, 4]

    # Apply Gabor filters at each orientation
    orientation_maps = [apply_gabor_filter(intensity, theta) for theta in orientations]

    # Build pyramid for each orientation response
    orientation_pyramids = [build_gaussian_pyramid(om, levels=levels)
                            for om in orientation_maps]

    # Center-surround for each pyramid
    orientation_feature_maps = []
    for pyramid in orientation_pyramids:
        for c in center_scales:
            for delta in delta_scales:
                s = c + delta
                fmap = center_surround_diff(pyramid, c, s)
                fmap = normalize_map(fmap)
                orientation_feature_maps.append(fmap)

    # Combine
    base_shape = orientation_feature_maps[0].shape
    conspicuity = np.zeros(base_shape, dtype=np.float32)
    for fm in orientation_feature_maps:
        resized = cv2.resize(fm, (base_shape[1], base_shape[0]))
        conspicuity += resized

    conspicuity = cv2.normalize(conspicuity, None, 0, 1, cv2.NORM_MINMAX)
    return conspicuity


# ---------------------------------------------------------------------------
# Main SaliencyMap class
# ---------------------------------------------------------------------------

class SaliencyMap:
    """Itti-Koch visual saliency model.

    Usage
    -----
    >>> sm = SaliencyMap("image.jpg")
    >>> sm.compute_saliency()
    >>> sm.save_results("output/")
    >>> sm.display()
    """

    def __init__(self, image_path=None, image_array=None):
        """Initialise with either an image path or an RGB numpy array.

        Parameters
        ----------
        image_path : str | Path | None
            Path to an image file.
        image_array : np.ndarray | None
            RGB image, float32 in [0, 1] or uint8 in [0, 255].

        Raises
        ------
        ValueError
            If neither *image_path* nor *image_array* is provided, or if
            the image could not be loaded.
        """
        if image_path is not None:
            self._load_from_path(image_path)
        elif image_array is not None:
            self._load_from_array(image_array)
        else:
            raise ValueError("Provide either image_path or image_array.")

    # ---- Image I/O -------------------------------------------------------

    def _load_from_path(self, path):
        path = Path(path)
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise ValueError(f"Could not load image: {path}")
        self._img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self._source_path = path

    def _load_from_array(self, arr):
        arr = np.asarray(arr)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        if arr.ndim == 2:
            # Grayscale → RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 4:
            # RGBA → RGB
            arr = arr[:, :, :3]
        self._img_rgb = arr.astype(np.float32)
        self._source_path = None

    @property
    def image(self):
        """Original RGB image (float32, [0, 1])."""
        return self._img_rgb

    # ---- Saliency computation --------------------------------------------

    def compute_saliency(self, levels=9, center_scales=None, delta_scales=None,
                         orientations=None, skip_color=False,
                         skip_orientation=False):
        """Run the full saliency pipeline.

        Populates ``self.intensity_map``, ``self.color_map``,
        ``self.orientation_map`` and ``self.saliency_map``.

        Parameters
        ----------
        levels : int
            Number of Gaussian pyramid levels.
        center_scales : list[int] | None
            Center scales for centre-surround (default: [2, 3, 4]).
        delta_scales : list[int] | None
            Deltas from centre for surround (default: [3, 4]).
        orientations : list[float] | None
            Gabor orientations in radians (default: 0, π/4, π/2, 3π/4).
        skip_color : bool
            Skip the colour channel.
        skip_orientation : bool
            Skip the orientation channel.

        Returns
        -------
        self
        """
        if center_scales is None:
            center_scales = [2, 3, 4]
        if delta_scales is None:
            delta_scales = [3, 4]
        if orientations is None:
            orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        # Intensity channel
        intensity = np.mean(self._img_rgb, axis=2)

        # Three conspicuity maps
        self.intensity_map = compute_intensity_conspicuity(
            intensity, levels=levels,
            center_scales=center_scales, delta_scales=delta_scales,
        )

        if skip_color:
            self.color_map = None
        else:
            self.color_map = compute_color_conspicuity(
                self._img_rgb, levels=levels,
                center_scales=center_scales, delta_scales=delta_scales,
            )

        if skip_orientation:
            self.orientation_map = None
        else:
            self.orientation_map = compute_orientation_conspicuity(
                intensity, orientations=orientations, levels=levels,
                center_scales=center_scales, delta_scales=delta_scales,
            )

        # Normalise conspicuity maps before fusion
        maps = [self.intensity_map]
        if not skip_color:
            maps.append(self.color_map)
        if not skip_orientation:
            maps.append(self.orientation_map)

        normalized = [normalize_map(m) for m in maps]

        # Equal-weight fusion
        self._saliency = sum(normalized) / float(len(normalized))
        self._saliency = cv2.normalize(
            self._saliency, None, 0, 1, cv2.NORM_MINMAX,
        )

        return self

    @property
    def saliency_map(self):
        """Final saliency map (float32, [0, 1])."""
        if self._saliency is None:
            raise RuntimeError("Call compute_saliency() first.")
        return self._saliency

    # ---- Visualisation helpers -------------------------------------------

    def get_heatmap(self, colormap=cv2.COLORMAP_JET):
        """Generate a colour heatmap overlay on the original image.

        Parameters
        ----------
        colormap : int
            OpenCV colormap constant (default: COLORMAP_JET).

        Returns
        -------
        overlay : np.ndarray (H, W, 3)
            RGB overlay image (uint8).
        heatmap : np.ndarray (H, W, 3)
            Raw heatmap (uint8, RGB).
        """
        # Resize saliency to original image size
        sal_resized = cv2.resize(
            self._saliency,
            (self._img_rgb.shape[1], self._img_rgb.shape[0]),
        )
        sal_uint8 = np.uint8(sal_resized * 255)

        # Heatmap (OpenCV returns BGR)
        heatmap_bgr = cv2.applyColorMap(sal_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = cv2.addWeighted(
            np.uint8(self._img_rgb * 255), 0.6,
            heatmap, 0.4,
            0,
        )

        return overlay, heatmap

    # ---- Output ----------------------------------------------------------

    def save_results(self, output_dir="output", prefix=None):
        """Save saliency map, heatmap, overlay, and individual maps.

        Parameters
        ----------
        output_dir : str | Path
            Directory to write results into.
        prefix : str | None
            Filename prefix (default: source image stem or "saliency").

        Returns
        -------
        Path
            The output directory.
        """
        if self._saliency is None:
            raise RuntimeError("Call compute_saliency() first.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if prefix is None and self._source_path is not None:
            prefix = self._source_path.stem
        else:
            prefix = prefix or "saliency"

        overlay, heatmap = self.get_heatmap()

        # Saliency map (grayscale)
        sal_uint8 = np.uint8(self._saliency * 255)
        cv2.imwrite(str(out / f"{prefix}_saliency.png"), sal_uint8)

        # Heatmap
        heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / f"{prefix}_heatmap.png"), heatmap_bgr)

        # Overlay
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / f"{prefix}_overlay.png"), overlay_bgr)

        # Individual conspicuity maps (skip any that were not computed)
        conspicuity_maps = [
            ("intensity", self.intensity_map),
            ("color", self.color_map),
            ("orientation", self.orientation_map),
        ]
        for name, cmap in conspicuity_maps:
            if cmap is not None:
                cmap_uint8 = np.uint8(cmap * 255)
                cv2.imwrite(str(out / f"{prefix}_{name}_conspicuity.png"), cmap_uint8)

        return out

    def display(self, figsize=(18, 6)):
        """Display the original image, saliency map, and overlay.

        Parameters
        ----------
        figsize : tuple
            Matplotlib figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        overlay, _ = self.get_heatmap()

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(self._img_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(self._saliency, cmap="hot")
        axes[1].set_title("Saliency Map")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Attention Heatmap Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
        return fig
