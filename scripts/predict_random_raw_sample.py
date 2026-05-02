from __future__ import annotations

from datetime import datetime
from pathlib import Path
import random

from ultralytics import YOLO


def main() -> None:
    random.seed(42)

    root = Path(".")
    raw_dir = root / "data" / "external_scenarios" / "raw"
    weights = root / "runs" / "detect" / "models" / "yolo26s_high_s_e18_f50" / "weights" / "best.pt"

    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts]
    if len(images) < 4:
        raise RuntimeError(f"Need at least 4 images in {raw_dir}, found {len(images)}")

    picked = random.sample(images, 4)
    print("=== RANDOM RAW SAMPLE (4) ===")
    for p in picked:
        print(p.as_posix())

    run_name = f"input_sample_4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project_dir = (root / "data" / "external_scenarios" / "predictions").resolve()

    model = YOLO(str(weights))
    model.predict(
        source=[str(p) for p in picked],
        conf=0.25,
        imgsz=960,
        device="0",
        save=True,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )

    print(f"SAVED_RUN={project_dir.as_posix()}/{run_name}")


if __name__ == "__main__":
    main()
