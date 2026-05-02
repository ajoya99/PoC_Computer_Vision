param(
    [string]$PythonCmd = "python",
    [string]$Device = "0",
    [int]$Epochs = 80,
    [int]$ImageSize = 832,
    [int]$Batch = 8
)

$ErrorActionPreference = 'Stop'

Write-Host "1) Sync zip files from Downloads"
& powershell -ExecutionPolicy Bypass -File scripts/sync_zips.ps1

Write-Host "2) Prepare merged dataset"
& $PythonCmd scripts/prepare_dataset.py --config configs/project_config.yaml

Write-Host "3) Train + validate + test yolo26n and yolo26s"
& $PythonCmd scripts/train_and_evaluate.py --data data/processed/merged_v1/data.yaml --epochs $Epochs --imgsz $ImageSize --batch $Batch --device $Device

Write-Host "Pipeline finished"
