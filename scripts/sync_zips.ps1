param(
    [string]$DownloadsDir = "$env:USERPROFILE\Downloads",
    [string]$RawZipsDir = "data/raw_zips"
)

$ErrorActionPreference = 'Stop'

$defaultZipNames = @(
    'Logistic.yolo26.zip',
    'Mobility Aids.yolo26.zip',
    'Warehouse.v1i.yolov8.zip',
    'wheelchair.yolo26.zip'
)

if (-not (Test-Path $RawZipsDir)) {
    New-Item -ItemType Directory -Path $RawZipsDir | Out-Null
}

foreach ($zipName in $defaultZipNames) {
    $source = Join-Path $DownloadsDir $zipName
    $target = Join-Path $RawZipsDir $zipName

    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $target -Force
        Write-Host "Copied: $zipName"
    }
    else {
        Write-Warning "Not found in Downloads: $zipName"
    }
}

Write-Host "ZIP sync complete."
