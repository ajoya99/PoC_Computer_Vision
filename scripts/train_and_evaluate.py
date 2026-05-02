from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Dict, List

import torch
from ultralytics import YOLO

try:
    import psutil
except ImportError:  # pragma: no cover - optional hardening dependency
    psutil = None


def _get_system_memory_gb() -> Dict[str, float]:
    if psutil is None:
        return {"total_ram_gb": 0.0, "available_ram_gb": 0.0}
    vm = psutil.virtual_memory()
    return {
        "total_ram_gb": vm.total / (1024**3),
        "available_ram_gb": vm.available / (1024**3),
    }


def _get_vram_gb(device: str) -> Dict[str, float]:
    if device == "cpu" or not torch.cuda.is_available():
        return {"total_vram_gb": 0.0, "free_vram_gb": 0.0}
    idx = 0
    props = torch.cuda.get_device_properties(idx)
    free_bytes, _ = torch.cuda.mem_get_info(idx)
    return {
        "total_vram_gb": props.total_memory / (1024**3),
        "free_vram_gb": free_bytes / (1024**3),
    }


def _build_retry_schedule(batch: int, imgsz: int, workers: int) -> List[Dict[str, int]]:
    schedule = [
        {"batch": max(1, batch), "imgsz": max(512, imgsz), "workers": max(0, workers)},
        {"batch": max(1, batch // 2), "imgsz": max(768, int(imgsz * 0.9)), "workers": min(max(0, workers), 4)},
        {"batch": max(1, batch // 4), "imgsz": max(640, int(imgsz * 0.8)), "workers": min(max(0, workers), 2)},
        {"batch": max(1, batch // 8), "imgsz": 512, "workers": 0},
    ]

    unique_schedule: List[Dict[str, int]] = []
    seen = set()
    for cfg in schedule:
        key = (cfg["batch"], cfg["imgsz"], cfg["workers"])
        if key not in seen:
            seen.add(key)
            unique_schedule.append(cfg)
    return unique_schedule


def _apply_memory_guard(config: Dict[str, int], device: str) -> Dict[str, int]:
    adjusted = dict(config)
    mem = _get_system_memory_gb()
    vram = _get_vram_gb(device)

    # If free RAM is low, reduce dataloader pressure first.
    if mem["available_ram_gb"] > 0 and mem["available_ram_gb"] < 6:
        adjusted["workers"] = min(adjusted["workers"], 1)
        adjusted["batch"] = max(1, adjusted["batch"] // 2)

    # If free VRAM is low, reduce batch and image size.
    if vram["free_vram_gb"] > 0 and vram["free_vram_gb"] < 2:
        adjusted["batch"] = max(1, adjusted["batch"] // 2)
        adjusted["imgsz"] = max(512, int(adjusted["imgsz"] * 0.8))

    return adjusted


def _is_memory_error(exc: Exception) -> bool:
    text = str(exc).lower()
    patterns = [
        "out of memory",
        "cuda out of memory",
        "can't allocate memory",
        "cudnn_status_alloc_failed",
        "defaultcpuallocator",
    ]
    return any(p in text for p in patterns)


def _train_with_failsafe(
    model_name: str,
    run_name: str,
    data_yaml: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    fraction: float,
) -> Path:
    retry_schedule = _build_retry_schedule(batch=batch, imgsz=imgsz, workers=workers)
    last_exc: Exception | None = None

    for attempt_idx, cfg in enumerate(retry_schedule, start=1):
        model = YOLO(model_name)
        guarded = _apply_memory_guard(cfg, device)

        print(
            f"[{model_name}] attempt {attempt_idx}/{len(retry_schedule)} "
            f"batch={guarded['batch']} imgsz={guarded['imgsz']} workers={guarded['workers']}"
        )

        try:
            model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=guarded["imgsz"],
                batch=guarded["batch"],
                fraction=fraction,
                device=device,
                workers=guarded["workers"],
                amp=True,
                cache=False,
                project="models",
                name=run_name,
                exist_ok=True,
            )

            checkpoint_candidates = [
                Path("runs") / "detect" / "models" / run_name / "weights" / "best.pt",
                Path("runs") / "detect" / "models" / run_name / "weights" / "last.pt",
                Path("models") / run_name / "weights" / "best.pt",
                Path("models") / run_name / "weights" / "last.pt",
            ]

            best_weights = next((p for p in checkpoint_candidates if p.exists()), None)

            if best_weights is None:
                raise RuntimeError(f"Training finished but no checkpoint found for {run_name}")
            return best_weights

        except Exception as exc:  # noqa: BLE001 - intentional retry boundary
            last_exc = exc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not _is_memory_error(exc) or attempt_idx == len(retry_schedule):
                raise

            print(f"[{model_name}] memory pressure detected, retrying with safer settings...")

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unexpected training failure without captured exception")


def run_pipeline(data_yaml: str, epochs: int, imgsz: int, batch: int, device: str, workers: int, fraction: float, models: List[str], tag: str) -> None:

    for model_name in models:
        model_size = model_name.replace(".pt", "")
        run_name = f"{model_size}_{tag}" if tag else f"{model_size}_custom"

        best_weights = _train_with_failsafe(
            model_name=model_name,
            run_name=run_name,
            data_yaml=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            fraction=fraction,
        )

        trained = YOLO(str(best_weights))

        trained.val(data=data_yaml, split="val", project="models", name=f"{run_name}_val", exist_ok=True)
        trained.val(data=data_yaml, split="test", project="models", name=f"{run_name}_test", exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, validate and test yolo26n/yolo26s on merged dataset.")
    parser.add_argument("--data", default="data/processed/merged_v1/data.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0, help="Use a fraction of training data (0-1]")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo26n.pt", "yolo26s.pt"],
        help="Model checkpoint names to train, e.g. yolo26s.pt",
    )
    parser.add_argument("--tag", default="custom", help="Suffix for run names under models/")
    parser.add_argument("--device", default="0", help="Use 0 for first GPU, or cpu")
    args = parser.parse_args()

    if not (0 < args.fraction <= 1.0):
        raise ValueError("--fraction must be in (0, 1]")

    run_pipeline(
        data_yaml=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        models=args.models,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
