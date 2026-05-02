from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_dataset_root(extract_dir: Path) -> Path | None:
    candidates = [extract_dir] + [p for p in extract_dir.rglob("*") if p.is_dir()]
    for c in candidates:
        has_any_split = (
            (c / "train").exists()
            or (c / "training").exists()
            or (c / "val").exists()
            or (c / "valid").exists()
            or (c / "validation").exists()
            or (c / "test").exists()
        )
        if has_any_split:
            return c
    return None


def read_yaml_names(dataset_root: Path) -> List[str]:
    yaml_candidates = list(dataset_root.glob("*.yaml")) + list(dataset_root.glob("*.yml"))
    if not yaml_candidates:
        return []

    for yaml_path in yaml_candidates:
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            names = data.get("names", [])
            if isinstance(names, dict):
                ordered = [names[k] for k in sorted(names.keys())]
                return [str(x) for x in ordered]
            if isinstance(names, list):
                return [str(x) for x in names]
        except Exception:
            continue
    return []


def pick_split_dir(dataset_root: Path, candidates: List[str]) -> Path | None:
    for name in candidates:
        p = dataset_root / name
        if p.exists() and p.is_dir():
            return p
    return None


def get_image_dir(split_dir: Path) -> Path | None:
    images = split_dir / "images"
    if images.exists() and images.is_dir():
        return images
    if split_dir.name in {"images", "imgs"}:
        return split_dir
    return None


def get_label_dir(split_dir: Path) -> Path | None:
    labels = split_dir / "labels"
    if labels.exists() and labels.is_dir():
        return labels
    return None


def build_class_index_map(source_names: List[str], alias_map: Dict[str, str], target_to_idx: Dict[str, int]) -> Dict[int, int]:
    idx_map: Dict[int, int] = {}
    for src_idx, src_name in enumerate(source_names):
        normalized = normalize_class_name(src_name)
        mapped = alias_map.get(normalized)
        if mapped is None:
            continue
        idx_map[src_idx] = target_to_idx[mapped]
    return idx_map


def deterministic_split(key: str) -> str:
    # 80/10/10 split derived from a stable hash for reproducibility.
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def transform_label_file(src_label: Path, dst_label: Path, idx_map: Dict[int, int]) -> Tuple[int, Counter]:
    kept = 0
    class_counter: Counter = Counter()

    lines_out: List[str] = []
    if src_label.exists():
        for line in src_label.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                src_cls = int(float(parts[0]))
            except ValueError:
                continue
            if src_cls not in idx_map:
                continue
            dst_cls = idx_map[src_cls]
            lines_out.append(" ".join([str(dst_cls)] + parts[1:]))
            kept += 1
            class_counter[dst_cls] += 1

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return kept, class_counter


def prepare_dataset(config_path: Path, keep_staging: bool) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    target_classes: List[str] = config["target_classes"]
    alias_map_raw: Dict[str, str] = config["class_alias_map"]
    zip_inputs: List[str] = config["zip_inputs"]
    output_dataset_dir = Path(config["output_dataset_dir"])

    alias_map = {normalize_class_name(k): normalize_class_name(v) for k, v in alias_map_raw.items()}
    target_to_idx = {c: i for i, c in enumerate(target_classes)}

    project_root = config_path.parent.parent
    zips = [(project_root / p).resolve() for p in zip_inputs]

    # Use a short temp path to avoid Windows MAX_PATH extraction failures.
    staging_root = Path(tempfile.gettempdir()) / "poc_cv_staging"
    processed_root = (project_root / output_dataset_dir).resolve()

    reset_dir(processed_root)
    for split in ["train", "val", "test"]:
        (processed_root / split / "images").mkdir(parents=True, exist_ok=True)
        (processed_root / split / "labels").mkdir(parents=True, exist_ok=True)

    report = {
        "datasets": {},
        "global": {
            "images": {"train": 0, "val": 0, "test": 0},
            "labels_kept": {"train": 0, "val": 0, "test": 0},
            "class_instances": defaultdict(int),
        },
    }

    split_aliases = {
        "train": ["train", "training"],
        "val": ["val", "valid", "validation"],
        "test": ["test"],
    }

    valid_img_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for zip_path in zips:
        ds_name = zip_path.stem.replace(" ", "_")
        ds_report = {
            "zip": str(zip_path),
            "status": "pending",
            "note": "",
            "images": {"train": 0, "val": 0, "test": 0},
            "labels_kept": {"train": 0, "val": 0, "test": 0},
            "class_instances": defaultdict(int),
        }

        if not zip_path.exists():
            ds_report["status"] = "missing_zip"
            ds_report["note"] = "ZIP file not found"
            report["datasets"][ds_name] = ds_report
            continue

        extract_dir = staging_root / ds_name
        reset_dir(extract_dir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        dataset_root = find_dataset_root(extract_dir)
        if dataset_root is None:
            ds_report["status"] = "skipped"
            ds_report["note"] = "No YOLO split structure found (train/val/test)"
            report["datasets"][ds_name] = ds_report
            continue

        source_names = read_yaml_names(dataset_root)
        if not source_names:
            ds_report["status"] = "skipped"
            ds_report["note"] = "Could not read class names from YAML"
            report["datasets"][ds_name] = ds_report
            continue

        idx_map = build_class_index_map(source_names, alias_map, target_to_idx)
        if not idx_map:
            ds_report["status"] = "skipped"
            ds_report["note"] = "No overlapping classes after mapping"
            report["datasets"][ds_name] = ds_report
            continue

        available_split_dirs = {k: pick_split_dir(dataset_root, v) for k, v in split_aliases.items()}
        only_train_available = available_split_dirs["train"] is not None and available_split_dirs["val"] is None and available_split_dirs["test"] is None

        for out_split, split_dir in available_split_dirs.items():
            if split_dir is None:
                continue

            img_dir = get_image_dir(split_dir)
            lbl_dir = get_label_dir(split_dir)
            if img_dir is None or lbl_dir is None:
                continue

            for img_path in img_dir.rglob("*"):
                if not img_path.is_file() or img_path.suffix.lower() not in valid_img_ext:
                    continue

                rel_img = img_path.relative_to(img_dir)
                rel_stem_str = rel_img.with_suffix("").as_posix()
                stem_hash = hashlib.sha1(rel_stem_str.encode("utf-8")).hexdigest()[:12]
                base_name = img_path.stem[:40]
                out_stem = f"{ds_name}__{base_name}__{stem_hash}"

                final_split = deterministic_split(out_stem) if (only_train_available and out_split == "train") else out_split

                out_img = processed_root / final_split / "images" / f"{out_stem}{img_path.suffix.lower()}"
                out_lbl = processed_root / final_split / "labels" / f"{out_stem}.txt"

                shutil.copy2(img_path, out_img)

                rel_label = rel_img.parent / f"{img_path.stem}.txt"
                src_lbl = lbl_dir / rel_label
                kept, class_counter = transform_label_file(src_lbl, out_lbl, idx_map)

                ds_report["images"][final_split] += 1
                ds_report["labels_kept"][final_split] += kept
                report["global"]["images"][final_split] += 1
                report["global"]["labels_kept"][final_split] += kept

                for class_idx, count in class_counter.items():
                    cls_name = target_classes[class_idx]
                    ds_report["class_instances"][cls_name] += count
                    report["global"]["class_instances"][cls_name] += count

        ds_report["status"] = "ok"
        report["datasets"][ds_name] = ds_report

    data_yaml = {
        "path": str(processed_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {idx: name for idx, name in enumerate(target_classes)},
        "nc": len(target_classes),
    }
    (processed_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "dataset_report.json"

    def normalize_defaultdict(o):
        if isinstance(o, defaultdict):
            return dict(o)
        raise TypeError(f"Not JSON serializable: {type(o)!r}")

    report_path.write_text(json.dumps(report, indent=2, default=normalize_defaultdict), encoding="utf-8")

    if not keep_staging and staging_root.exists():
        shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)

    print(f"Prepared dataset at: {processed_root}")
    print(f"Data YAML: {processed_root / 'data.yaml'}")
    print(f"Report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare merged YOLO dataset with class filtering and remapping.")
    parser.add_argument(
        "--config",
        default="configs/project_config.yaml",
        help="Path to project config YAML.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep extracted temporary dataset folders under data/staging.",
    )
    args = parser.parse_args()

    prepare_dataset(Path(args.config).resolve(), keep_staging=args.keep_staging)


if __name__ == "__main__":
    main()
