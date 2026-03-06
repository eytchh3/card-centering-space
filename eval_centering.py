from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from detector import analyze_centering

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(fixtures_dir: Path) -> list[Path]:
    images = [
        path
        for path in fixtures_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(images)


def run_eval(fixtures_dir: Path) -> int:
    images = collect_images(fixtures_dir)

    if not images:
        print(
            "No local fixture images found in "
            f"'{fixtures_dir}'. Add image files under fixtures/centering/ and rerun."
        )
        return 0

    print(f"Found {len(images)} fixture image(s) in '{fixtures_dir}'.")

    success_count = 0
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[SKIP] {image_path}: could not be read by OpenCV")
            continue

        result = analyze_centering(image)
        if "error" in result:
            print(f"[FAIL] {image_path}: {result['error']}")
            continue

        success_count += 1
        print(
            f"[OK] {image_path}: "
            f"L/R {result['centering_lr']} | T/B {result['centering_tb']}"
        )

    print(f"Completed evaluation. Successful detections: {success_count}/{len(images)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run centering evaluation on local fixture images.")
    parser.add_argument(
        "--fixtures-dir",
        default="fixtures/centering",
        help="Directory containing local evaluation fixture images.",
    )
    args = parser.parse_args()

    return run_eval(Path(args.fixtures_dir))


if __name__ == "__main__":
    raise SystemExit(main())
