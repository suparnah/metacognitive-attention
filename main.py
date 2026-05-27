#!/usr/bin/env python3
"""
main.py — CLI entry point for the Itti-Koch Visual Saliency Model.

Usage
-----
    python main.py image.jpg                          # basic
    python main.py image.jpg --output results --show   # with options
    python main.py image.jpg --levels 7                # custom pyramid depth
"""

import argparse
import sys
from pathlib import Path

from saliency import SaliencyMap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a visual saliency map from an input image "
                    "(Itti-Koch model).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s image.jpg\n"
            "  %(prog)s image.jpg -o results --show\n"
            "  %(prog)s image.jpg --levels 7 --no-color\n"
        ),
    )

    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument(
        "--output", "-o", type=str, default="output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display results with matplotlib",
    )
    parser.add_argument(
        "--levels", type=int, default=9,
        help="Number of Gaussian pyramid levels (default: 9)",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Skip colour channel (grayscale-only saliency)",
    )
    parser.add_argument(
        "--no-orientation", action="store_true",
        help="Skip orientation channel",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not Path(args.image).is_file():
        print(f"Error: file not found — {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading image: {args.image}")
    sm = SaliencyMap(image_path=args.image)

    print("Computing saliency map …")
    sm.compute_saliency(
        levels=args.levels,
        skip_color=args.no_color,
        skip_orientation=args.no_orientation,
    )

    print(f"Saving results to: {args.output}/")
    out_dir = sm.save_results(args.output)
    print(f"  → {out_dir}/{Path(args.image).stem}_saliency.png")
    print(f"  → {out_dir}/{Path(args.image).stem}_heatmap.png")
    print(f"  → {out_dir}/{Path(args.image).stem}_overlay.png")
    print(f"  → {out_dir}/{Path(args.image).stem}_{{intensity,color,orientation}}_conspicuity.png")

    if args.show:
        sm.display()

    print("Done.")


if __name__ == "__main__":
    main()
