"""RT-DETR training script with full val/test evaluation matching train_and_evaluate.py."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RT-DETR and evaluate on val+test splits")
    parser.add_argument("--data", default="data/processed/merged_v1/data.yaml", help="Dataset YAML path")
    parser.add_argument("--model", default="rtdetr-l.pt", help="Model checkpoint to use")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs (default: 2 for quick test)")
    parser.add_argument("--imgsz", type=int, default=768, help="Image size (default: 768)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--workers", type=int, default=2, help="Data loader workers (default: 2)")
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of data to use (default: 0.1 for quick test)")
    parser.add_argument("--device", default="0", help="GPU device ID or 'cpu'")
    parser.add_argument("--tag", default="quick_test", help="Run name suffix")
    args = parser.parse_args()

    model_file = Path(args.model)
    if not model_file.exists():
        print(f"ERROR: Model file not found: {args.model}")
        print(f"\nAvailable models in workspace:")
        for f in Path(".").glob("*.pt"):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        exit(1)

    # Auto-detect device
    if args.device == "0":
        if not torch.cuda.is_available():
            print("⚠ CUDA not available, switching to CPU")
            device = "cpu"
        else:
            device = "0"
    else:
        device = args.device

    print("\n" + "="*60)
    print("RT-DETR Quick Test Training")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Data Fraction: {args.fraction} ({args.fraction*100:.0f}%)")
    print(f"Device: {device}")
    print("="*60 + "\n")

    # Load model
    model = YOLO(args.model)

    # Run training
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        fraction=args.fraction,
        device=device,
        amp=True,
        cache=False,
        project="runs/detect/models",
        name=f"rtdetr_test_{args.tag}",
        exist_ok=True,
        patience=10,
    )

    # Locate best checkpoint
    run_name = f"rtdetr_test_{args.tag}"
    checkpoint_candidates = [
        Path("runs") / "detect" / "models" / run_name / "weights" / "best.pt",
        Path("runs") / "detect" / "runs" / "detect" / "models" / run_name / "weights" / "best.pt",
    ]
    best_weights = next((p for p in checkpoint_candidates if p.exists()), None)
    if best_weights is None:
        print("WARNING: Could not find best.pt — skipping val/test evaluation")
        return

    print(f"\nBest weights: {best_weights}")
    trained = YOLO(str(best_weights))
    data_yaml = str(Path(args.data).resolve())

    print("\n--- Running validation split evaluation ---")
    trained.val(data=data_yaml, split="val", project="runs/detect/models", name=f"{run_name}_val", exist_ok=True)

    print("\n--- Running test split evaluation ---")
    trained.val(data=data_yaml, split="test", project="runs/detect/models", name=f"{run_name}_test", exist_ok=True)

    print("\n" + "="*60)
    print("✓ Training + val + test evaluation complete!")
    print("="*60)
    print(f"\nResults saved under: runs/detect/models/{run_name}/")
    print(f"Val results:  runs/detect/models/{run_name}_val/")
    print(f"Test results: runs/detect/models/{run_name}_test/")


if __name__ == "__main__":
    main()
