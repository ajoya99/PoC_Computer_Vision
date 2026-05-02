from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on real scenario images.")
    parser.add_argument("--weights", required=True, help="Path to trained weights (.pt)")
    parser.add_argument("--source", default="data/external_scenarios/raw", help="Folder with images/videos")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0", help="Use 0 for first GPU, or cpu")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=str(Path(args.source).resolve()),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project="data/external_scenarios/predictions",
        name="latest",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
