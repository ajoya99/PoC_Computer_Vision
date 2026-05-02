from __future__ import annotations

from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener


def main() -> None:
    root = Path(".")
    raw_dir = root / "data" / "external_scenarios" / "raw"
    heic_files = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".heic")

    if not heic_files:
        print(f"No HEIC files found in {raw_dir.as_posix()}")
        return

    register_heif_opener()

    converted = 0
    for source in heic_files:
        target = source.with_suffix(".jpg")
        with Image.open(source) as image:
            rgb_image = image.convert("RGB")
            rgb_image.save(target, format="JPEG", quality=95)

        source.unlink()
        converted += 1
        print(f"CONVERTED {source.name} -> {target.name}")

    print(f"TOTAL_CONVERTED={converted}")


if __name__ == "__main__":
    main()