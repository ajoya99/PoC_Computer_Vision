$ErrorActionPreference = "Stop"

$root = "c:/Users/alvaro.joya machado/Documents/VS_Code/PoC_Computer_Vision"
$zipDir = Join-Path $root "data/raw_zips"
$outDir = Join-Path $root "data/external_scenarios/raw"

if (!(Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

Add-Type -AssemblyName System.IO.Compression.FileSystem

$exts = @(".jpg", ".jpeg", ".png", ".webp", ".bmp")
$zipFiles = Get-ChildItem -Path $zipDir -File -Filter "*.zip"
$totalExtracted = 0

foreach ($zip in $zipFiles) {
    $archive = [System.IO.Compression.ZipFile]::OpenRead($zip.FullName)
    try {
        $zipTag = [System.IO.Path]::GetFileNameWithoutExtension($zip.Name) -replace "\s+", "_"
        $count = 0

        foreach ($entry in $archive.Entries) {
            if ([string]::IsNullOrWhiteSpace($entry.Name)) { continue }

            $ext = [System.IO.Path]::GetExtension($entry.Name).ToLowerInvariant()
            if ($exts -notcontains $ext) { continue }

            $safeBase = ([System.IO.Path]::GetFileNameWithoutExtension($entry.Name) -replace "[^a-zA-Z0-9._-]", "_")
            if ([string]::IsNullOrWhiteSpace($safeBase)) { $safeBase = "image" }
            if ($safeBase.Length -gt 70) { $safeBase = $safeBase.Substring(0, 70) }

            $destName = "${zipTag}__${safeBase}${ext}"
            $destPath = Join-Path $outDir $destName

            $i = 1
            while (Test-Path $destPath) {
                $destName = "${zipTag}__${safeBase}__$i${ext}"
                $destPath = Join-Path $outDir $destName
                $i++
            }

            $entryStream = $entry.Open()
            try {
                $outStream = [System.IO.File]::Create($destPath)
                try {
                    $entryStream.CopyTo($outStream)
                }
                finally {
                    $outStream.Dispose()
                }
            }
            finally {
                $entryStream.Dispose()
            }

            $count++
        }

        $totalExtracted += $count
        Write-Output ("{0}: extracted {1} images" -f $zip.Name, $count)
    }
    finally {
        $archive.Dispose()
    }
}

$rawCount = (Get-ChildItem -Path $outDir -File | Measure-Object).Count
Write-Output ("TOTAL_EXTRACTED={0}" -f $totalExtracted)
Write-Output ("RAW_FOLDER_FILE_COUNT={0}" -f $rawCount)
