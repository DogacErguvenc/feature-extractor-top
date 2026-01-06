# Terazi AI kurulum yardımcısı (Windows)
# Kullanım: PowerShell'i yönetici olmadan açın, repo kökünde çalıştırın:
#   .\setup_terazi.ps1

param(
    [switch]$ForceEnv    # .env dosyasını mevcutsa da yeniden oluşturur
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonCmd = $null

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[ERR ] $msg" -ForegroundColor Red }

function Ensure-Python {
    $py = (Get-Command python -ErrorAction SilentlyContinue)
    if ($py) {
        $script:PythonCmd = "python"
        Write-Info "Python: $($py.Source)"
        return
    }
    $pyLauncher = (Get-Command py -ErrorAction SilentlyContinue)
    if ($pyLauncher) {
        $script:PythonCmd = "py -3.11"
        Write-Info "Python launcher bulundu (py -3.11 kullanılacak)"
        return
    }
    throw "Python 3.11+ bulunamadı. Lütfen kurun ve tekrar deneyin."
}

function Ensure-Node {
    $node = (Get-Command node -ErrorAction SilentlyContinue)
    if (-not $node) { Write-Warn "Node.js bulunamadı (frontend için gerekli). Devam ediliyor." }
    else { Write-Info "Node.js: $($node.Source)" }
}


function New-ApiKey {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    return ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""
}

function Get-EnvMap($path) {
    $map = @{}
    if (Test-Path $path) {
        Get-Content $path | ForEach-Object {
            if ($_ -match '^\s*([^#=]+)=(.*)$') {
                $map[$matches[1]] = $matches[2]
            }
        }
    }
    return $map
}

function Setup-Backend {
    Write-Info "Backend sanal ortam kuruluyor..."
    $venvPath = Join-Path -Path "backend" -ChildPath ".venv"
    if (-not (Test-Path $venvPath)) {
        & $PythonCmd -m venv $venvPath
    }
    & "$venvPath\Scripts\python.exe" -m pip install --upgrade pip
    & "$venvPath\Scripts\python.exe" -m pip install -r backend/requirements.txt
    Write-Info "Backend bağımlılıkları kuruldu."
}

function Setup-Env {
    $envPath = "backend/.env"
    $frontendEnvPath = "frontend/.env"
    $incomingDir = Join-Path $RepoRoot "backend\incoming"
    $embeddingStoreDir = Join-Path $RepoRoot "backend\embedding_store"
    if (-not (Test-Path $incomingDir)) {
        New-Item -ItemType Directory -Force -Path $incomingDir | Out-Null
    }
    if (-not (Test-Path $embeddingStoreDir)) {
        New-Item -ItemType Directory -Force -Path $embeddingStoreDir | Out-Null
    }
    if ((-not (Test-Path $envPath)) -or $ForceEnv) {
        $modelPath = Join-Path $RepoRoot "backend\models\local_model.onnx"
        $labelsPath = Join-Path $RepoRoot "backend\models\local_labels.json"
        $largeModelPath = Join-Path $RepoRoot "backend\models\local_model_large.onnx"
        $largeLabelsPath = Join-Path $RepoRoot "backend\models\local_labels_large.json"
        $apiKey = New-ApiKey
        @"
MONGO_URL=mongodb://localhost:27017
DB_NAME=terazi_production
AI_PROVIDER=local
LOCAL_MODEL_PATH=$modelPath
LOCAL_LABELS_PATH=$labelsPath
LOCAL_LARGE_MODEL_PATH=$largeModelPath
LOCAL_LARGE_LABELS_PATH=$largeLabelsPath
LOCAL_SMALL_IMAGE_SIZE=224
LOCAL_LARGE_IMAGE_SIZE=300
EMBEDDING_STORE_DIR=$embeddingStoreDir
EMBEDDING_MODEL_NAME=vit_large_patch14_dinov2.lvd142m
EMBEDDING_MODEL_WEIGHTS=
EMBEDDING_IMAGE_SIZE=518
EMBEDDING_SCALE_SIZE=576
EMBEDDING_CROP_MODE=edge5
EMBEDDING_TOP_K=3
EMBEDDING_MIN_SIM=0.35
EMBEDDING_MARGIN=0.05
EMBEDDING_DEVICE=
GOOGLE_API_KEY=
OPENAI_API_KEY=
API_KEY=$apiKey
CORS_ORIGINS=http://localhost:3000
ALLOWED_IMAGE_DIR=$incomingDir
DISABLE_DOCS=true
"@ | Out-File -FilePath $envPath -Encoding ASCII -Force
        Write-Info ".env olusturuldu: $envPath"
    } else {
        Write-Info ".env zaten mevcut, degisiklik yapilmadi (ForceEnv kullanmadikca)."
    }

    $envMap = Get-EnvMap $envPath
    $frontendKey = $envMap["API_KEY"]
    if ((-not (Test-Path $frontendEnvPath)) -or $ForceEnv) {
        @"
REACT_APP_BACKEND_URL=http://localhost:8001
REACT_APP_API_KEY=$frontendKey
"@ | Out-File -FilePath $frontendEnvPath -Encoding ASCII -Force
        Write-Info "Frontend .env olusturuldu: $frontendEnvPath"
    } else {
        Write-Info "Frontend .env zaten mevcut, degisiklik yapilmadi (ForceEnv kullanmadikca)."
    }
}

function Setup-Frontend {
    if (-not (Get-Command yarn -ErrorAction SilentlyContinue)) {
        Write-Warn "Yarn bulunamadı; frontend bağımlılıkları kurulmadı."
        return
    }
    Write-Info "Frontend bağımlılıkları kuruluyor (yarn install)..."
    Push-Location frontend
    yarn install
    Pop-Location
    Write-Info "Frontend bağımlılıkları kuruldu."
}

# Ana akış
try {
    Write-Info "Terazi AI kurulum başlıyor..."
    Ensure-Python
    Ensure-Node
    Setup-Backend
    Setup-Env
    Setup-Frontend
    Write-Info "Kurulum tamamlandı. Çalıştırmak için:"
    Write-Host "  Backend: cd backend; .\\.venv\\Scripts\\activate; uvicorn server:app --host 0.0.0.0 --port 8001 --reload"
    Write-Host "  Frontend: cd frontend; yarn start"
} catch {
    Write-Err $_
    exit 1
}
