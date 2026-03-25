param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$OutputName = "terazi_backend",
    [string]$Version = "1.0.0",
    [switch]$BuildSetup
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    param([string]$BackendDir, [string]$RepoRoot)

    $candidates = @(
        (Join-Path $BackendDir ".venv\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\Scripts\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return $pythonCmd.Source
    }

    throw "Python bulunamadi. backend\\.venv veya sistem python kurulu olmali."
}

function Resolve-IsccExe {
    $candidate = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    if (Test-Path $candidate) {
        return $candidate
    }

    $isccCmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($isccCmd) {
        return $isccCmd.Source
    }

    throw "ISCC.exe bulunamadi. Inno Setup 6 kurulu olmali."
}

function Copy-IfExists {
    param(
        [string]$SourcePath,
        [string]$DestinationPath
    )

    if (-not (Test-Path $SourcePath)) {
        return
    }

    if (Test-Path $DestinationPath) {
        Remove-Item -LiteralPath $DestinationPath -Recurse -Force
    }

    Copy-Item -LiteralPath $SourcePath -Destination $DestinationPath -Recurse -Force
}

$backendDir = Join-Path $ProjectRoot "backend"
if (-not (Test-Path $backendDir)) {
    throw "Backend dizini bulunamadi: $backendDir"
}

$pythonExe = Resolve-PythonExe -BackendDir $backendDir -RepoRoot $ProjectRoot
Write-Host "Python:" $pythonExe

Push-Location $backendDir
try {
    & $pythonExe -m pip install --upgrade pip pyinstaller
    if ($LASTEXITCODE -ne 0) {
        throw "pip install pyinstaller basarisiz oldu."
    }

    $pyiArgs = @(
        "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--console",
        "--name", $OutputName,
        "--collect-all", "fastapi",
        "--collect-all", "uvicorn",
        "--collect-all", "onnxruntime",
        "--collect-all", "torch",
        "--collect-all", "torchvision",
        "--collect-all", "timm",
        "--collect-all", "cv2",
        "run_backend.py"
    )

    & $pythonExe @pyiArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build basarisiz oldu."
    }
}
finally {
    Pop-Location
}

$distDir = Join-Path $backendDir "dist\$OutputName"
if (-not (Test-Path $distDir)) {
    throw "Dist klasoru bulunamadi: $distDir"
}

# Runtime files
$runtimeFiles = @(
    "butcher_config.yaml",
    "tray_roi.json",
    "plu_budgets.json",
    ".env"
)

foreach ($file in $runtimeFiles) {
    $src = Join-Path $backendDir $file
    $dst = Join-Path $distDir $file
    if (Test-Path $src) {
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }
}

# Runtime directories
$runtimeDirs = @(
    "models",
    "incoming",
    "embedding_store"
)

foreach ($dir in $runtimeDirs) {
    $src = Join-Path $backendDir $dir
    $dst = Join-Path $distDir $dir
    Copy-IfExists -SourcePath $src -DestinationPath $dst
}

# Deployment helper files
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "start_backend.bat") -Destination (Join-Path $distDir "start_backend.bat") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "runtime.env.example") -Destination (Join-Path $distDir ".env.example") -Force

Write-Host "EXE hazir:" $distDir

if ($BuildSetup) {
    $isccExe = Resolve-IsccExe
    $issPath = Join-Path $PSScriptRoot "terazi_backend_setup.iss"
    if (-not (Test-Path $issPath)) {
        throw "Inno Setup script bulunamadi: $issPath"
    }

    Push-Location $PSScriptRoot
    try {
        & $isccExe "/DMyAppVersion=$Version" "/DSourceDir=$distDir" $issPath
        if ($LASTEXITCODE -ne 0) {
            throw "Inno Setup build basarisiz oldu."
        }
    }
    finally {
        Pop-Location
    }

    Write-Host "Setup hazir:" (Join-Path $PSScriptRoot "terazi_backend_setup_$Version.exe")
}
