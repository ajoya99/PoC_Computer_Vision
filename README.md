# PoC Computer Vision - YOLO26 Training Pipeline

This project trains two models (`yolo26n` and `yolo26s`) for the following final classes only:

1. box
2. pallet
3. person
4. forklift
5. cart
6. wheelchair

All other source classes are ignored.

Merges applied during dataset preparation:

1. `push_wheelchair` -> `wheelchair`
2. `pallets` -> `pallet`
3. `wheel chair` / `wheel_chair` -> `wheelchair`

## Project Structure

```text
PoC_Computer_Vision/
  configs/
    project_config.yaml
  data/
    raw_zips/                  # Input dataset ZIP files
    staging/                   # Temporary extraction (auto-managed)
    processed/
      merged_v1/               # Final merged YOLO dataset output
        train/images
        train/labels
        val/images
        val/labels
        test/images
        test/labels
        data.yaml
    external_scenarios/
      raw/                     # New real-world pictures/videos for inference
      predictions/             # Inference outputs
  models/                      # Training runs and weights
  reports/
    dataset_report.json        # Summary of merged/skipped classes and counts
  scripts/
    sync_zips.ps1              # Copy default ZIPs from Downloads to data/raw_zips
    prepare_dataset.py         # Extract + merge + remap/filter classes
    train_and_evaluate.py      # Train + val + test for yolo26n and yolo26s
    predict_external.py        # Inference on external scenario images
    run_full_pipeline.ps1      # End-to-end automation
  requirements.txt
  .gitignore
  README.md
```

## Current Datasets

The pipeline is currently configured for these 4 ZIP files:

1. `Logistic.yolo26.zip`
2. `Mobility Aids.yolo26.zip`
3. `Warehouse.v1i.yolov8.zip`
4. `wheelchair.yolo26.zip`

If a ZIP contains only `train` split, the script automatically creates deterministic `train/val/test` assignments (80/10/10) from that data.

If a ZIP has no YOLO split folders at all, it is skipped and reported.

## Setup

### 1) Create environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Sync ZIP files into project

```powershell
powershell -ExecutionPolicy Bypass -File scripts/sync_zips.ps1
```

You can also manually drop new dataset ZIPs into `data/raw_zips` and add them under `zip_inputs` in `configs/project_config.yaml`.

## Prepare Merged Dataset

```powershell
python scripts/prepare_dataset.py --config configs/project_config.yaml
```

Output:

1. `data/processed/merged_v1/data.yaml`
2. merged images/labels in train/val/test folders
3. `reports/dataset_report.json` with dataset-level stats and skipped reasons

Current result with your four ZIPs:

1. `Warehouse.v1i.yolov8.zip` included directly (already had train/val/test)
2. `Mobility Aids.yolo26.zip` included using deterministic split from train-only data
3. `wheelchair.yolo26.zip` included using deterministic split from train-only data
4. `Logistic.yolo26.zip` included using deterministic split from train-only data

## Train, Validate, Test (both yolo26n and yolo26s)

```powershell
python scripts/train_and_evaluate.py --data data/processed/merged_v1/data.yaml --epochs 18 --fraction 0.50 --imgsz 768 --batch 4 --workers 2 --device 0
```

For controlled pilot comparisons (single model, smaller data share):

```powershell
python scripts/train_and_evaluate.py --data data/processed/merged_v1/data.yaml --models yolo26s.pt --epochs 5 --fraction 0.20 --imgsz 704 --batch 4 --workers 2 --device 0 --tag pilot_s_e5_f20
```

What this does:

1. trains `yolo26n.pt`
2. trains `yolo26s.pt`
3. validates best/last checkpoint on validation split
4. tests best/last checkpoint on test split

Model outputs are written under `runs/detect/models/`.

## Latest Model Comparison (High Budget - April 22, 2026)

Both models were trained with identical parameters:

- `fraction=0.50`
- `epochs=18`
- `imgsz=768`
- `batch=4`
- `workers=2`

Test split results:

| Metric | yolo26s (small) | yolo26n (nano) | Delta (small - nano) |
|--------|------------------|----------------|----------------------|
| Precision | 0.8414 | 0.7884 | +0.0530 |
| Recall | 0.6096 | 0.5799 | +0.0297 |
| mAP50 | 0.6760 | 0.6465 | +0.0295 |
| mAP50-95 | 0.4949 | 0.4728 | +0.0221 |

Conclusion: `yolo26s` remains the best-performing model for accuracy. `yolo26n` is faster but has lower overall detection quality.

### Training Failsafe (RAM/VRAM protection)

`scripts/train_and_evaluate.py` includes a failsafe to reduce crash risk on heavy runs:

1. Starts with your chosen `batch`, `imgsz`, and `workers`
2. Monitors available system RAM and GPU VRAM before each attempt
3. If memory is tight, automatically lowers workers/batch/image size
4. On OOM errors, retries with progressively safer settings instead of stopping immediately
5. Uses `cache=False` to avoid large RAM spikes

If memory pressure remains too high even at safe settings, the script exits with the final error.

## Full Pipeline (single command)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_full_pipeline.ps1 -PythonCmd python -Device 0 -Epochs 80 -ImageSize 832 -Batch 8
```

## Add New Real-Scenario Pictures Later

For future unlabelled real-world checks:

1. Copy images/videos into `data/external_scenarios/raw`
2. Run inference:

```powershell
python scripts/predict_external.py --weights runs/detect/models/yolo26s_high_s_e18_f50/weights/best.pt --source data/external_scenarios/raw --conf 0.25 --imgsz 960 --device 0
```

Predictions are saved in `data/external_scenarios/predictions/latest`.

For a direct high-budget comparison script (small vs nano), run:

```powershell
python scripts/compare_high_models.py
```

## Notes

1. The class mapping is centralized in `configs/project_config.yaml`.
2. Keep the same target class order unless you intentionally want to retrain label indices.
3. If training fails due to unavailable `yolo26*.pt` names in your Ultralytics version, update model names in `scripts/train_and_evaluate.py` to your local available checkpoints.
