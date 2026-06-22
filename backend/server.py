from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Depends, UploadFile, File, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Tuple
import uuid
from datetime import datetime, timezone
import cv2
import base64
from PIL import Image
import io
import asyncio
import json
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from google import genai
from google.genai.types import Content, Part
from openai import AsyncOpenAI
import psutil
import time
import threading

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Get API keys
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# Choose which AI provider to use
# Supported: 'gemini', 'openai', 'local', 'local_large', 'local_embedding', 'butcher_resnet', 'local_gemini', 'local_gemini_consensus'
AI_PROVIDER = os.environ.get('AI_PROVIDER', 'gemini')
AI_MODEL = os.environ.get('AI_MODEL', 'gemini-2.0-flash')  # Model name
LOCAL_MODEL_PATH = Path(os.environ.get('LOCAL_MODEL_PATH', ROOT_DIR / "models" / "local_model.onnx"))
LOCAL_LABELS_PATH = Path(os.environ.get('LOCAL_LABELS_PATH', ROOT_DIR / "models" / "local_labels.json"))
LOCAL_LARGE_MODEL_PATH = Path(os.environ.get('LOCAL_LARGE_MODEL_PATH', ROOT_DIR / "models" / "local_model_large.onnx"))
LOCAL_LARGE_LABELS_PATH = Path(os.environ.get('LOCAL_LARGE_LABELS_PATH', ROOT_DIR / "models" / "local_labels_large.json"))
LOCAL_SMALL_IMAGE_SIZE = int(os.environ.get('LOCAL_SMALL_IMAGE_SIZE', '224'))
LOCAL_LARGE_IMAGE_SIZE = int(os.environ.get('LOCAL_LARGE_IMAGE_SIZE', '300'))
BUTCHER_CONFIG_PATH = Path(os.environ.get('BUTCHER_CONFIG_PATH', ROOT_DIR / "butcher_config.yaml")).resolve()
BUTCHER_TOP_K = int(os.environ.get('BUTCHER_TOP_K', '3'))
REFERENCE_IMAGE_DIR = Path(os.environ.get('REFERENCE_IMAGE_DIR', ROOT_DIR / "reference_images")).resolve()
REFERENCE_MAX_IMAGES = int(os.environ.get('REFERENCE_MAX_IMAGES', '2'))
PROMPT_VERSION = os.environ.get('PROMPT_VERSION', 'dense_v1')
ALLOWED_IMAGE_DIR = Path(os.environ.get('ALLOWED_IMAGE_DIR', ROOT_DIR / "incoming")).resolve()
TRAIN_DATA_DIR = Path(os.environ.get('TRAIN_DATA_DIR', ROOT_DIR.parent / "datasets" / "train")).resolve()
CANDIDATE_REF_DIR = Path(os.environ.get('CANDIDATE_REF_DIR', ROOT_DIR / "candidate_ref_pool")).resolve()
EMBEDDING_STORE_DIR = Path(os.environ.get('EMBEDDING_STORE_DIR', ROOT_DIR / "embedding_store")).resolve()
EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', 'vit_large_patch14_dinov2.lvd142m')
EMBEDDING_MODEL_WEIGHTS = os.environ.get('EMBEDDING_MODEL_WEIGHTS', '').strip()
EMBEDDING_IMAGE_SIZE = int(os.environ.get('EMBEDDING_IMAGE_SIZE', '518'))
EMBEDDING_SCALE_SIZE = int(os.environ.get('EMBEDDING_SCALE_SIZE', '576'))
EMBEDDING_CROP_MODE = os.environ.get('EMBEDDING_CROP_MODE', 'edge5')
EMBEDDING_TOP_K = int(os.environ.get('EMBEDDING_TOP_K', '3'))
EMBEDDING_MIN_SIM = float(os.environ.get('EMBEDDING_MIN_SIM', '0.35'))
EMBEDDING_MARGIN = float(os.environ.get('EMBEDDING_MARGIN', '0.05'))
EMBEDDING_DEVICE = os.environ.get('EMBEDDING_DEVICE', '').strip()
BOOTSTRAP_ENABLE = os.environ.get('BOOTSTRAP_ENABLE', 'false').lower() == 'true'
BOOTSTRAP_MIN_COUNT = int(os.environ.get('BOOTSTRAP_MIN_COUNT', '20'))
BOOTSTRAP_ACCEPT_SIM = float(os.environ.get('BOOTSTRAP_ACCEPT_SIM', '0.90'))
BOOTSTRAP_REJECT_SIM = float(os.environ.get('BOOTSTRAP_REJECT_SIM', '0.60'))
BOOTSTRAP_MAX_POOL = int(os.environ.get('BOOTSTRAP_MAX_POOL', '500'))
BOOTSTRAP_USE_PROTOTYPES = os.environ.get('BOOTSTRAP_USE_PROTOTYPES', 'true').lower() == 'true'
BOOTSTRAP_PROTOTYPE_K = int(os.environ.get('BOOTSTRAP_PROTOTYPE_K', '6'))
BOOTSTRAP_SUPPORT_SIM = float(os.environ.get('BOOTSTRAP_SUPPORT_SIM', str(BOOTSTRAP_ACCEPT_SIM)))
BOOTSTRAP_SUPPORT_COUNT = int(os.environ.get('BOOTSTRAP_SUPPORT_COUNT', '3'))
BOOTSTRAP_FIXED_POOL_SIZE = int(os.environ.get('BOOTSTRAP_FIXED_POOL_SIZE', '100'))
BOOTSTRAP_CLUSTER_K = int(os.environ.get('BOOTSTRAP_CLUSTER_K', str(BOOTSTRAP_PROTOTYPE_K)))
CLUSTER_ENABLE = os.environ.get('CLUSTER_ENABLE', 'true').lower() == 'true'
CLUSTER_DUP_SIM = float(os.environ.get('CLUSTER_DUP_SIM', '0.98'))
PLU_MAX_EMBEDDINGS = int(os.environ.get('PLU_MAX_EMBEDDINGS', '200'))
PLU_MIN_EMBEDDINGS = int(os.environ.get('PLU_MIN_EMBEDDINGS', '40'))
BATCH_FOLDERING_ROOT = Path(os.environ.get('BATCH_FOLDERING_ROOT', ROOT_DIR / "batch_foldering")).resolve()
BATCH_FOLDERING_LOW_PCT = float(os.environ.get('BATCH_FOLDERING_LOW_PCT', '40'))
BATCH_FOLDERING_HIGH_PCT = float(os.environ.get('BATCH_FOLDERING_HIGH_PCT', '70'))
RESNET_ANALYSIS_MAX_SAMPLE_LIMIT = int(os.environ.get('RESNET_ANALYSIS_MAX_SAMPLE_LIMIT', '5000'))
_raw_plu_budget_path = os.environ.get('PLU_BUDGETS_PATH', '').strip()
_default_plu_budget_path = (ROOT_DIR / "plu_budgets.json")
if _raw_plu_budget_path:
    PLU_BUDGETS_PATH = Path(_raw_plu_budget_path).expanduser().resolve()
elif _default_plu_budget_path.exists():
    PLU_BUDGETS_PATH = _default_plu_budget_path
else:
    PLU_BUDGETS_PATH = None
PLU_BUDGETS_CACHE: Optional[dict] = None
TRAY_ROI_ENABLE = os.environ.get('TRAY_ROI_ENABLE', 'false').lower() == 'true'
TRAY_ROI_STRICT = os.environ.get('TRAY_ROI_STRICT', 'false').lower() == 'true'
_raw_tray_roi_path = os.environ.get('TRAY_ROI_PATH', '').strip()
TRAY_ROI_PATH = (
    Path(_raw_tray_roi_path).expanduser().resolve()
    if _raw_tray_roi_path
    else (ROOT_DIR / "tray_roi.json")
)
TRAY_ROI_CACHE: Optional[dict] = None
TRAY_ROI_CACHE_MTIME: Optional[float] = None
DEFAULT_CORS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

def parse_cors_origins() -> list[str]:
    raw = os.environ.get('CORS_ORIGINS', '')
    if raw:
        origins = [o.strip() for o in raw.split(',') if o.strip()]
        return origins
    return DEFAULT_CORS


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    seen = set()
    result = []
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            continue
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _path_is_same_or_child(path: Path, root: Path) -> bool:
    try:
        common = os.path.commonpath([
            os.path.normcase(str(path)),
            os.path.normcase(str(root)),
        ])
    except ValueError:
        return False
    return common == os.path.normcase(str(root))


def _format_folder_entry(path: Path, is_root: bool = False) -> dict:
    name = path.name or str(path)
    return {
        "name": name,
        "path": str(path),
        "is_root": is_root,
    }


def _get_centroid_browser_roots() -> List[Path]:
    raw = os.environ.get("CENTROID_BROWSER_ROOTS", "").strip()
    candidates: List[Path] = []

    if raw:
        parts = [part.strip().strip('"') for part in raw.replace("\n", os.pathsep).split(os.pathsep)]
        candidates.extend(Path(part) for part in parts if part)
    else:
        if os.name == "nt":
            try:
                for partition in psutil.disk_partitions(all=False):
                    mountpoint = (partition.mountpoint or "").strip()
                    opts = (partition.opts or "").lower()
                    if not mountpoint or "cdrom" in opts:
                        continue
                    candidates.append(Path(mountpoint))
            except Exception as exc:
                logging.warning(f"Failed to enumerate Windows drives for folder browser: {exc}")

        candidates.extend([
            ALLOWED_IMAGE_DIR,
            TRAIN_DATA_DIR,
            ROOT_DIR,
            ROOT_DIR / "centroid_rank_jobs",
        ])
        try:
            candidates.append(Path.home() / "Pictures")
        except Exception:
            pass

    roots = [path for path in _dedupe_paths(candidates) if path.exists() and path.is_dir()]
    return roots or [ROOT_DIR.resolve()]


def _resolve_centroid_browser_path(raw_path: str) -> Tuple[Path, Path]:
    if not str(raw_path or "").strip():
        raise HTTPException(status_code=400, detail="path is required")

    try:
        path = Path(str(raw_path)).expanduser().resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}")

    for root in _get_centroid_browser_roots():
        if _path_is_same_or_child(path, root):
            return path, root

    raise HTTPException(status_code=400, detail="Path is outside allowed folder roots")


def resolve_safe_path(path_candidate: Path) -> Path:
    """Resolve a path and ensure it stays under ALLOWED_IMAGE_DIR."""
    real_path = path_candidate.resolve()
    allowed = ALLOWED_IMAGE_DIR
    try:
        common = os.path.commonpath([os.path.normcase(str(real_path)), os.path.normcase(str(allowed))])
    except ValueError:
        raise HTTPException(status_code=400, detail="File path is outside allowed directory")
    if common != os.path.normcase(str(allowed)):
        raise HTTPException(status_code=400, detail="File path is outside allowed directory")
    return real_path


def _load_plu_budgets() -> Optional[dict]:
    global PLU_BUDGETS_CACHE
    if PLU_BUDGETS_CACHE is not None:
        return PLU_BUDGETS_CACHE
    if not PLU_BUDGETS_PATH:
        PLU_BUDGETS_CACHE = None
        return None
    try:
        if not PLU_BUDGETS_PATH.exists():
            PLU_BUDGETS_CACHE = None
            return None
        with PLU_BUDGETS_PATH.open("r", encoding="utf-8") as fh:
            PLU_BUDGETS_CACHE = json.load(fh)
        return PLU_BUDGETS_CACHE
    except Exception as exc:
        logging.error(f"Failed to load PLU budgets from {PLU_BUDGETS_PATH}: {exc}")
        PLU_BUDGETS_CACHE = None
        return None


def _get_plu_budget(plu_code: str) -> tuple[int, int, str]:
    """Return (min_budget, max_budget, tier_name) for a PLU."""
    plu_code = str(plu_code)
    min_budget = max(0, int(PLU_MIN_EMBEDDINGS))
    max_budget = max(0, int(PLU_MAX_EMBEDDINGS))
    tier_name = "default"
    cfg = _load_plu_budgets()
    if not cfg:
        return min_budget, max_budget, tier_name

    default_cfg = cfg.get("default", {})
    if isinstance(default_cfg, dict):
        min_budget = max(0, int(default_cfg.get("min", min_budget)))
        max_budget = max(0, int(default_cfg.get("max", max_budget)))

    # Per-PLU override takes precedence
    plu_overrides = cfg.get("plu_overrides", {}) or {}
    override = plu_overrides.get(plu_code)
    if isinstance(override, dict):
        min_budget = max(0, int(override.get("min", min_budget)))
        max_budget = max(0, int(override.get("max", max_budget)))
        tier_name = "override"
        if max_budget < min_budget:
            max_budget = min_budget
        return min_budget, max_budget, tier_name

    # Otherwise, use tier mapping
    plu_tiers = cfg.get("plu_tiers", {}) or {}
    tiers = cfg.get("tiers", {}) or {}
    tier = plu_tiers.get(plu_code)
    if isinstance(tier, str) and tier in tiers and isinstance(tiers[tier], dict):
        tier_cfg = tiers[tier]
        min_budget = max(0, int(tier_cfg.get("min", min_budget)))
        max_budget = max(0, int(tier_cfg.get("max", max_budget)))
        tier_name = tier

    if max_budget < min_budget:
        max_budget = min_budget
    return min_budget, max_budget, tier_name

def _validate_tray_roi_config(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("ROI config must be a JSON object")
    points = raw.get("points")
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("ROI config requires exactly 4 points")
    normalized_points = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Each point must be [x, y]")
        x = float(point[0])
        y = float(point[1])
        normalized_points.append([x, y])

    source_size = raw.get("source_size")
    if source_size is not None:
        if not isinstance(source_size, list) or len(source_size) != 2:
            raise ValueError("source_size must be [width, height]")
        source_size = [int(source_size[0]), int(source_size[1])]
        if source_size[0] <= 0 or source_size[1] <= 0:
            raise ValueError("source_size must be positive")

    output_size = raw.get("output_size")
    if output_size is not None:
        if not isinstance(output_size, list) or len(output_size) != 2:
            raise ValueError("output_size must be [width, height]")
        output_size = [int(output_size[0]), int(output_size[1])]
        if output_size[0] <= 0 or output_size[1] <= 0:
            raise ValueError("output_size must be positive")

    return {
        "points": normalized_points,
        "source_size": source_size,
        "output_size": output_size,
    }


def _load_tray_roi(force_reload: bool = False) -> Optional[dict]:
    global TRAY_ROI_CACHE
    global TRAY_ROI_CACHE_MTIME
    try:
        if not TRAY_ROI_PATH.exists():
            TRAY_ROI_CACHE = None
            TRAY_ROI_CACHE_MTIME = None
            return None
        stat = TRAY_ROI_PATH.stat()
        mtime = stat.st_mtime
        if (
            not force_reload
            and TRAY_ROI_CACHE is not None
            and TRAY_ROI_CACHE_MTIME is not None
            and mtime == TRAY_ROI_CACHE_MTIME
        ):
            return TRAY_ROI_CACHE

        raw = json.loads(TRAY_ROI_PATH.read_text(encoding="utf-8"))
        cfg = _validate_tray_roi_config(raw)
        TRAY_ROI_CACHE = cfg
        TRAY_ROI_CACHE_MTIME = mtime
        return cfg
    except Exception as exc:
        logging.error(f"Failed to load tray ROI config from {TRAY_ROI_PATH}: {exc}")
        TRAY_ROI_CACHE = None
        TRAY_ROI_CACHE_MTIME = None
        return None


def _save_tray_roi(cfg: dict) -> None:
    global TRAY_ROI_CACHE
    global TRAY_ROI_CACHE_MTIME
    TRAY_ROI_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAY_ROI_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    TRAY_ROI_CACHE = cfg
    TRAY_ROI_CACHE_MTIME = TRAY_ROI_PATH.stat().st_mtime


def _roi_points_to_pixels(points: List[List[float]], width: int, height: int, source_size: Optional[List[int]]) -> np.ndarray:
    arr = np.array(points, dtype=np.float32)
    if arr.shape != (4, 2):
        raise ValueError("ROI points must be shape (4,2)")

    if np.max(arr) <= 1.5 and np.min(arr) >= -0.1:
        arr[:, 0] = arr[:, 0] * float(width)
        arr[:, 1] = arr[:, 1] * float(height)
    elif source_size and len(source_size) == 2 and source_size[0] > 0 and source_size[1] > 0:
        sx = float(width) / float(source_size[0])
        sy = float(height) / float(source_size[1])
        arr[:, 0] = arr[:, 0] * sx
        arr[:, 1] = arr[:, 1] * sy

    arr[:, 0] = np.clip(arr[:, 0], 0, max(0, width - 1))
    arr[:, 1] = np.clip(arr[:, 1], 0, max(0, height - 1))
    return arr.astype(np.float32)


def _compute_roi_output_size(points_px: np.ndarray, output_size: Optional[List[int]]) -> Tuple[int, int]:
    if output_size and len(output_size) == 2:
        return int(output_size[0]), int(output_size[1])

    tl, tr, br, bl = points_px
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    out_w = int(max(width_a, width_b))
    out_h = int(max(height_a, height_b))
    return max(1, out_w), max(1, out_h)


def _encode_pil_to_base64_jpeg(img: Image.Image, quality: int = 85) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def preprocess_image_for_ai(image_base64: str) -> Tuple[str, dict]:
    if not TRAY_ROI_ENABLE:
        return image_base64, {"roi_applied": False, "roi_reason": "disabled"}

    cfg = _load_tray_roi()
    if not cfg:
        if TRAY_ROI_STRICT:
            raise RuntimeError(f"Tray ROI is enabled but config is missing/invalid: {TRAY_ROI_PATH}")
        return image_base64, {"roi_applied": False, "roi_reason": "missing_config"}

    try:
        img = decode_base64_to_pil(image_base64)
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        h, w = arr.shape[:2]
        pts = _roi_points_to_pixels(cfg["points"], w, h, cfg.get("source_size"))
        area = abs(float(cv2.contourArea(pts)))
        if area < 10.0:
            raise ValueError("ROI polygon area is too small")

        out_w, out_h = _compute_roi_output_size(pts, cfg.get("output_size"))
        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(
            arr,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )
        cropped = Image.fromarray(warped)
        return _encode_pil_to_base64_jpeg(cropped, quality=85), {
            "roi_applied": True,
            "roi_reason": "ok",
            "roi_width": out_w,
            "roi_height": out_h,
        }
    except Exception as exc:
        if TRAY_ROI_STRICT:
            raise RuntimeError(f"Tray ROI apply failed: {exc}") from exc
        logging.warning(f"Tray ROI apply skipped: {exc}")
        return image_base64, {"roi_applied": False, "roi_reason": f"apply_failed: {exc}"}

# Configure FastAPI (optionally disable docs in production)
app_kwargs = {}
if os.environ.get("DISABLE_DOCS", "").lower() == "true":
    app_kwargs.update({"docs_url": None, "redoc_url": None, "openapi_url": None})

# Create the main app without a prefix
app = FastAPI(**app_kwargs)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# System mode: "training" or "production"
SYSTEM_MODE = "training"  # Default mode (overwritten at startup if stored)
SYSTEM_CONFIG_ID = "terazi_system_config"

@dataclass(frozen=True)
class LocalModelSpec:
    model_path: Path
    labels_path: Path
    image_size: Tuple[int, int]

LOCAL_MODEL_SPECS = {
    "local": LocalModelSpec(
        model_path=LOCAL_MODEL_PATH,
        labels_path=LOCAL_LABELS_PATH,
        image_size=(LOCAL_SMALL_IMAGE_SIZE, LOCAL_SMALL_IMAGE_SIZE),
    ),
    "local_large": LocalModelSpec(
        model_path=LOCAL_LARGE_MODEL_PATH,
        labels_path=LOCAL_LARGE_LABELS_PATH,
        image_size=(LOCAL_LARGE_IMAGE_SIZE, LOCAL_LARGE_IMAGE_SIZE),
    ),
}
LOCAL_MODEL_CACHE: dict[str, Tuple[ort.InferenceSession, str, List[dict]]] = {}
EMBEDDING_ENGINE = None
EMBEDDING_STORE = None
EMBEDDING_STORE_LOCK = asyncio.Lock()
BOOTSTRAP_POOL_LOCK = asyncio.Lock()
CENTROID_RANK_JOBS: dict[str, dict] = {}
CENTROID_RANK_CANCEL_EVENTS: dict[str, threading.Event] = {}

# Pydantic Models
class PLUProduct(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plu_code: str
    name: str
    description: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PLUProductCreate(BaseModel):
    plu_code: str
    name: str
    description: str

class CapturedImage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plu_code: str
    image_base64: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    phase: str  # "training" or "production"
    ai_provider: Optional[str] = None
    ai_model: Optional[str] = None

class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plu_code: str
    selected_plu_name: str
    image_base64: str
    processed_image_base64: Optional[str] = None
    roi_applied: Optional[bool] = None
    roi_reason: Optional[str] = None
    roi_width: Optional[int] = None
    roi_height: Optional[int] = None
    ai_analysis: str
    analysis_selected_plu: Optional[str] = None
    analysis_selected_score: Optional[str] = None
    analysis_best_other_score: Optional[str] = None
    analysis_predicted_plu: Optional[str] = None
    analysis_predicted_score: Optional[str] = None
    analysis_embedding_count: Optional[str] = None
    top_matches: Optional[List[dict]] = None
    is_match: bool
    confidence: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ai_provider: Optional[str] = None
    ai_model: Optional[str] = None
    processing_ms: Optional[float] = None
    fallback_local_match: Optional[bool] = None
    fallback_local_confidence: Optional[float] = None
    fallback_remote_provider: Optional[str] = None
    fallback_remote_match: Optional[bool] = None
    fallback_remote_confidence: Optional[float] = None
    reference_count: Optional[int] = None
    batch_id: Optional[str] = None
    source: Optional[str] = None
    original_filename: Optional[str] = None

class RefCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plu_code: str
    image_base64: str
    status: str = "pending"  # pending | approved | rejected
    reason: Optional[str] = None
    notes: Optional[str] = None
    original_filename: Optional[str] = None
    source_validation_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    approved_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    approved_path: Optional[str] = None
    bootstrap_score: Optional[float] = None
    bootstrap_support: Optional[int] = None
    bootstrap_proto_k: Optional[int] = None

class RefCandidateCreateFromValidation(BaseModel):
    validation_id: str
    expected_plu: Optional[str] = None
    reason: Optional[str] = None
    notes: Optional[str] = None

class RefCandidateCreateFromFile(BaseModel):
    plu_code: str
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    reason: Optional[str] = None
    notes: Optional[str] = None

class RefCandidateReview(BaseModel):
    notes: Optional[str] = None

class TrayROIConfigUpdate(BaseModel):
    points: List[List[float]]
    source_size: Optional[List[int]] = None
    output_size: Optional[List[int]] = None

class PLUSelection(BaseModel):
    plu_code: str

class SystemModeUpdate(BaseModel):
    mode: str  # "training" or "production"

class ValidateSyncRequest(BaseModel):
    plu_code: str
    image_base64: Optional[str] = None
    filename: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[str] = None

class ResnetTopKSyncRequest(BaseModel):
    image_base64: Optional[str] = None
    filename: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    top_k: int = 5

class LiveValidateRequest(BaseModel):
    plu_code: str
    camera_index: int = 0
    persist_capture: bool = False
    persist_validation: bool = False

class LivePredictRequest(BaseModel):
    image_base64: Optional[str] = None
    camera_index: int = 0

class BatchValidationMeta(BaseModel):
    filename: str
    plu_code: str

class CentroidRankRequest(BaseModel):
    input_dir: str
    out_dir: Optional[str] = None
    recursive: bool = False
    copy_mode: str = "bands"
    bands: int = 5

class AIConfigUpdate(BaseModel):
    provider: str
    model: Optional[str] = None

class DashboardStats(BaseModel):
    total_images: int
    total_validations: int
    match_count: int
    mismatch_count: int
    match_percentage: float
    images_by_plu: dict
    current_mode: str

class HealthStatus(BaseModel):
    mongo_connected: bool
    camera_available: bool
    ai_provider: str
    ai_model: str
    system_mode: str
    cpu_percent: float
    mem_percent: float
    disk_percent: float

# Helper function to capture image from camera
def capture_image_from_camera(camera_index: int = 0, warmup_frames: int = 5) -> Optional[str]:
    """Capture image from default camera and return as base64 string"""
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logging.error("Cannot open camera")
            return None
        
        # Wait a bit for camera to initialize
        for _ in range(max(0, int(warmup_frames))):
            cap.read()
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logging.error("Failed to capture frame from camera")
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize if too large
        max_size = 1920
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = tuple(int(dim * ratio) for dim in pil_image.size)
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64
    except Exception as e:
        logging.error(f"Error capturing image: {e}")
        return None

def encode_uploaded_image(file_bytes: bytes) -> str:
    """Convert uploaded image bytes to base64 JPEG, normalizing size and mode."""
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")

            max_size = 1920
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        raise ValueError(f"Image processing failed: {e}")

async def load_system_mode_from_db() -> str:
    """Load persisted system mode from database (defaults to training)."""
    try:
        doc = await db.system_config.find_one({"id": SYSTEM_CONFIG_ID})
        if doc and doc.get("mode") in ["training", "production"]:
            return doc["mode"]
    except Exception as e:
        logging.error(f"Error loading system mode from DB: {e}")
    return "training"

async def persist_system_mode(mode: str) -> None:
    """Persist system mode in database."""
    try:
        await db.system_config.update_one(
            {"id": SYSTEM_CONFIG_ID},
            {"$set": {"mode": mode}},
            upsert=True
        )
    except Exception as e:
        logging.error(f"Error saving system mode to DB: {e}")

async def load_ai_config_from_db() -> tuple[str, str]:
    """Load persisted AI provider/model from database (fallback to defaults)."""
    try:
        doc = await db.system_config.find_one({"id": SYSTEM_CONFIG_ID})
        if doc:
            provider = doc.get("ai_provider", AI_PROVIDER)
            model = doc.get("ai_model", AI_MODEL)
            return provider, model
    except Exception as e:
        logging.error(f"Error loading AI config from DB: {e}")
    return AI_PROVIDER, AI_MODEL

async def persist_ai_config(provider: str, model: str) -> None:
    """Persist AI provider/model in database."""
    try:
        await db.system_config.update_one(
            {"id": SYSTEM_CONFIG_ID},
            {"$set": {"ai_provider": provider, "ai_model": model}},
            upsert=True
        )
    except Exception as e:
        logging.error(f"Error saving AI config to DB: {e}")


async def ensure_database_indexes() -> None:
    """Create indexes used by dashboard and analysis queries."""
    try:
        await db.validation_results.create_index([("timestamp", -1)], background=True)
        await db.validation_results.create_index([("source", 1), ("timestamp", -1)], background=True)
        await db.validation_results.create_index([("source", 1), ("plu_code", 1), ("timestamp", -1)], background=True)
        await db.validation_results.create_index([("source", 1), ("plu_code", 1)], background=True)
        await db.validation_results.create_index([("ai_provider", 1), ("timestamp", -1)], background=True)
        await db.validation_results.create_index([("ai_provider", 1), ("plu_code", 1), ("timestamp", -1)], background=True)
        await db.validation_results.create_index([("ai_provider", 1), ("plu_code", 1), ("top_matches.plu_code", 1), ("timestamp", -1)], background=True)
        await db.captured_images.create_index([("timestamp", -1)], background=True)
        await db.captured_images.create_index([("plu_code", 1), ("timestamp", -1)], background=True)
    except Exception as e:
        logging.error(f"Error creating database indexes: {e}")


def softmax(logits: np.ndarray) -> np.ndarray:
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def get_local_model_spec(model_key: str) -> LocalModelSpec:
    spec = LOCAL_MODEL_SPECS.get(model_key)
    if not spec:
        raise ValueError(f"Unsupported local model key: {model_key}")
    return spec

def has_local_model(model_key: str) -> bool:
    spec = LOCAL_MODEL_SPECS.get(model_key)
    if not spec:
        return False
    return spec.model_path.exists() and spec.labels_path.exists()

def load_local_labels(labels_path: Path) -> List[dict]:
    """Load class labels mapping for a local model."""
    if not labels_path.exists():
        raise FileNotFoundError(f"LOCAL_LABELS_PATH not found: {labels_path}")
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize to list of dicts with plu_code and name
    labels: List[dict] = []
    for item in data:
        if isinstance(item, dict):
            labels.append({"plu_code": str(item.get("plu_code") or item.get("id") or item.get("code") or item.get("label")),
                           "name": item.get("name") or item.get("label") or str(item.get("plu_code"))})
        else:
            labels.append({"plu_code": str(item), "name": str(item)})
    return labels

def preprocess_for_local_model(image_base64: str, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Prepare image tensor for ONNX model (NCHW, float32, normalized)."""
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(size, Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0  # HWC
    # Imagenet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    # HWC -> CHW
    chw = np.transpose(arr, (2, 0, 1))
    # Add batch dimension
    return np.expand_dims(chw, axis=0)

def load_local_model(model_key: str) -> Tuple[ort.InferenceSession, str, List[dict], Tuple[int, int]]:
    """Load ONNX model and labels for offline inference."""
    if model_key in LOCAL_MODEL_CACHE:
        session, input_name, labels = LOCAL_MODEL_CACHE[model_key]
        return session, input_name, labels, get_local_model_spec(model_key).image_size
    spec = get_local_model_spec(model_key)
    if not spec.model_path.exists():
        raise FileNotFoundError(f"Local model not found: {spec.model_path}")
    session = ort.InferenceSession(str(spec.model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    labels = load_local_labels(spec.labels_path)
    LOCAL_MODEL_CACHE[model_key] = (session, input_name, labels)
    logging.info(
        f"Local model '{model_key}' loaded from {spec.model_path}, labels from {spec.labels_path}"
    )
    return session, input_name, labels, spec.image_size


def _resolve_embedding_device() -> str:
    if EMBEDDING_DEVICE:
        return EMBEDDING_DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_embedding_engine():
    global EMBEDDING_ENGINE
    if EMBEDDING_ENGINE is not None:
        return EMBEDDING_ENGINE
    from embedding_engine import EmbeddingConfig, EmbeddingEngine

    weights_path = Path(EMBEDDING_MODEL_WEIGHTS) if EMBEDDING_MODEL_WEIGHTS else None
    config = EmbeddingConfig(
        model_name=EMBEDDING_MODEL_NAME,
        weights_path=weights_path,
        image_size=EMBEDDING_IMAGE_SIZE,
        scale_size=EMBEDDING_SCALE_SIZE,
        crop_mode=EMBEDDING_CROP_MODE,
        device=_resolve_embedding_device(),
    )
    EMBEDDING_ENGINE = EmbeddingEngine(config)
    return EMBEDDING_ENGINE


def get_embedding_store():
    global EMBEDDING_STORE
    if EMBEDDING_STORE is not None:
        return EMBEDDING_STORE
    from embedding_store import EmbeddingStore

    EMBEDDING_STORE = EmbeddingStore.load(EMBEDDING_STORE_DIR)
    return EMBEDDING_STORE


def get_latest_image_path(allowed_dir: Path) -> Path:
    """Return the most recently modified image file in allowed_dir."""
    if not allowed_dir.exists() or not allowed_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Allowed image dir not found: {allowed_dir}")
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    candidates = []
    for item in allowed_dir.iterdir():
        if item.is_file() and item.suffix.lower() in exts:
            try:
                candidates.append((item.stat().st_mtime, item))
            except OSError:
                continue
    if not candidates:
        raise HTTPException(status_code=400, detail=f"No images found in {allowed_dir}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def decode_base64_to_pil(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img


def _stringify_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _stringify_response(payload: dict) -> dict:
    return {key: _stringify_value(val) for key, val in payload.items()}


def _to_score_pct(match: dict) -> Optional[float]:
    """Normalize a top-match score to percentage when possible."""
    if not isinstance(match, dict):
        return None
    raw_score = match.get("score")
    score_type = str(match.get("score_type") or "").strip().lower()
    prob_raw = match.get("prob")
    score_val: Optional[float] = None
    prob_val: Optional[float] = None

    try:
        if raw_score is not None and raw_score != "":
            score_val = float(raw_score)
    except (TypeError, ValueError):
        score_val = None

    try:
        if prob_raw is not None and prob_raw != "":
            prob_val = float(prob_raw)
    except (TypeError, ValueError):
        prob_val = None

    if score_type == "probability_pct" and score_val is not None:
        return score_val
    if prob_val is not None:
        return prob_val * 100.0
    if score_val is None:
        return None
    if score_val <= 1.0:
        return score_val * 100.0
    return score_val


def _ensure_top_matches(raw_value, top_k: int = 5) -> List[dict]:
    if isinstance(raw_value, str):
        try:
            raw_value = json.loads(raw_value)
        except Exception:
            return []
    if not isinstance(raw_value, list):
        return []
    top: List[dict] = []
    for item in raw_value:
        if isinstance(item, dict):
            top.append(item)
        if len(top) >= max(1, int(top_k)):
            break
    return top


def _extract_resnet_metrics_for_plu(doc: dict, selected_plu: str, low_conf_threshold_pct: float) -> dict:
    top_matches = _ensure_top_matches(doc.get("top_matches"), top_k=5)
    top5_codes: List[str] = []
    for item in top_matches:
        code = str(item.get("plu_code") or "").strip()
        if code:
            top5_codes.append(code)

    top1_code = top5_codes[0] if top5_codes else ""
    top1_score_pct = _to_score_pct(top_matches[0]) if top_matches else None

    selected_score_pct: Optional[float] = None
    for item in top_matches:
        code = str(item.get("plu_code") or "").strip()
        if code == selected_plu:
            selected_score_pct = _to_score_pct(item)
            break

    is_top5_match = selected_plu in top5_codes
    is_top1_match = bool(top1_code and top1_code == selected_plu)
    is_low_conf_top5 = bool(
        is_top5_match
        and selected_score_pct is not None
        and selected_score_pct < low_conf_threshold_pct
    )

    predicted_plu = str(doc.get("analysis_predicted_plu") or top1_code).strip()
    filename = str(doc.get("original_filename") or doc.get("filename") or "").strip()

    return {
        "validation_id": str(doc.get("id") or ""),
        "timestamp": doc.get("timestamp"),
        "plu_code": selected_plu,
        "filename": filename,
        "predicted_plu": predicted_plu,
        "top5_codes": top5_codes,
        "top1_score_pct": round(top1_score_pct, 2) if top1_score_pct is not None else None,
        "selected_score_pct": round(selected_score_pct, 2) if selected_score_pct is not None else None,
        "is_top5_match": is_top5_match,
        "is_top1_match": is_top1_match,
        "is_low_conf_top5": is_low_conf_top5,
    }


def _sanitize_filename(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return "image.jpg"
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".", " ")) else "_" for ch in raw)
    safe = safe.strip(" .")
    return safe or "image.jpg"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _categorize_batch_result(
    expected_plu: str,
    ai_provider: str,
    top_matches_raw,
    low_pct: float,
    high_pct: float,
) -> tuple[str, dict]:
    expected = str(expected_plu or "").strip()
    provider = str(ai_provider or "").strip()
    if provider != "butcher_resnet":
        return "diger_provider", {
            "provider": provider,
            "selected_score_pct": None,
            "in_top5": False,
            "top5_codes": [],
            "top1_code": "",
        }

    top_matches = _ensure_top_matches(top_matches_raw, top_k=5)
    top5_codes: List[str] = []
    selected_score_pct: Optional[float] = None
    for item in top_matches:
        code = str(item.get("plu_code") or "").strip()
        if code:
            top5_codes.append(code)
        if code == expected:
            selected_score_pct = _to_score_pct(item)

    top1_code = top5_codes[0] if top5_codes else ""
    in_top5 = expected in top5_codes
    if not in_top5:
        category = "top5_uyumsuz"
    elif selected_score_pct is None:
        category = "top5_skor_bilinmiyor"
    elif selected_score_pct < low_pct:
        category = "top5_dusuk"
    elif selected_score_pct < high_pct:
        category = "top5_orta"
    else:
        category = "top5_yuksek"

    return category, {
        "provider": provider,
        "selected_score_pct": round(selected_score_pct, 2) if selected_score_pct is not None else None,
        "in_top5": in_top5,
        "top5_codes": top5_codes,
        "top1_code": top1_code,
    }


def _save_batch_file_for_foldering(
    batch_id: str,
    expected_plu: str,
    category: str,
    filename: str,
    file_bytes: bytes,
) -> Path:
    safe_name = _sanitize_filename(filename)
    target_dir = BATCH_FOLDERING_ROOT / str(batch_id) / str(category) / str(expected_plu)
    target_dir.mkdir(parents=True, exist_ok=True)
    dst = _unique_path(target_dir / safe_name)
    dst.write_bytes(file_bytes)
    return dst


def _build_local_top_matches(labels: List[dict], probs: np.ndarray, top_n: int = 3) -> List[dict]:
    if probs.ndim != 1:
        probs = np.array(probs).reshape(-1)
    limit = min(max(0, int(top_n)), int(probs.shape[0]), len(labels))
    if limit <= 0:
        return []

    top_indices = np.argsort(probs)[-limit:][::-1]
    matches: List[dict] = []
    for rank, idx in enumerate(top_indices, start=1):
        label = labels[int(idx)] if int(idx) < len(labels) else {}
        plu_code = str(
            label.get("plu_code")
            or label.get("id")
            or label.get("code")
            or label.get("label")
            or ""
        )
        plu_name = str(label.get("name") or label.get("label") or plu_code)
        score = round(float(probs[int(idx)]) * 100.0, 2)
        matches.append(
            {
                "rank": rank,
                "plu_code": plu_code,
                "plu_name": plu_name,
                "score": score,
                "score_type": "probability_pct",
            }
        )
    return matches


def _normalize_label(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _resolve_selected_prob(class_probs: dict[str, float], plu_product: PLUProduct) -> tuple[Optional[float], Optional[str]]:
    selected_candidates = [
        str(plu_product.plu_code),
        str(plu_product.name),
    ]
    normalized_to_raw = {
        _normalize_label(raw): raw
        for raw in class_probs.keys()
    }
    for candidate in selected_candidates:
        key = normalized_to_raw.get(_normalize_label(candidate))
        if key is not None:
            return float(class_probs[key]), str(key)
    return None, None


def _build_embedding_top_matches(store, query_embeddings: np.ndarray, top_k: int = 3, top_n: int = 3) -> List[dict]:
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings[None, :]
    sims = query_embeddings @ store.embeddings.T
    sims = sims.max(axis=0)

    plu_name_map: dict[str, str] = {}
    for item in store.meta:
        plu_code = str(item.get("plu_code") or "")
        if not plu_code or plu_code in plu_name_map:
            continue
        plu_name = str(item.get("plu_name") or "").strip()
        if plu_name:
            plu_name_map[plu_code] = plu_name

    scores: list[tuple[str, float]] = []
    k_hint = max(1, int(top_k))
    for plu_code, indices in store.plu_index.items():
        idx_arr = np.array(indices, dtype=int)
        if idx_arr.size == 0:
            continue
        values = sims[idx_arr]
        k = min(k_hint, values.size)
        if k == 1:
            score = float(values.max())
        else:
            top_idx = np.argpartition(values, -k)[-k:]
            score = float(values[top_idx].mean())
        scores.append((str(plu_code), score))

    scores.sort(key=lambda item: item[1], reverse=True)
    limit = min(max(0, int(top_n)), len(scores))
    matches: List[dict] = []
    for rank, (plu_code, score) in enumerate(scores[:limit], start=1):
        matches.append(
            {
                "rank": rank,
                "plu_code": plu_code,
                "plu_name": plu_name_map.get(plu_code, plu_code),
                "score": round(score, 4),
                "score_type": "cosine_similarity",
            }
        )
    return matches


def _save_base64_image(target_dir: Path, filename_hint: Optional[str], image_base64: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    image_bytes = base64.b64decode(image_base64)
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        stem = Path(filename_hint).stem if filename_hint else "ref_candidate"
        filename = f"{stem}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        out_path = target_dir / filename
        img.save(out_path, format="JPEG", quality=95)
        return out_path


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def _farthest_point_sampling(embeddings: np.ndarray, k: int, seed_idx: Optional[int] = None) -> list[int]:
    n = embeddings.shape[0]
    if k >= n:
        return list(range(n))
    if seed_idx is None:
        centroid = _normalize_vec(embeddings.mean(axis=0))
        sims = embeddings @ centroid
        seed_idx = int(np.argmax(sims))
    selected = [seed_idx]
    distances = 1.0 - (embeddings @ embeddings[seed_idx])
    for _ in range(1, k):
        idx = int(np.argmax(distances))
        if idx in selected:
            break
        selected.append(idx)
        new_dist = 1.0 - (embeddings @ embeddings[idx])
        distances = np.minimum(distances, new_dist)
    return selected


def _select_bootstrap_eviction_index(
    entries: list[dict],
    incoming_vec: np.ndarray,
    cluster_k: int,
) -> Optional[int]:
    valid_indices: list[int] = []
    vectors: list[np.ndarray] = []
    for idx, item in enumerate(entries):
        emb = item.get("embedding")
        if emb:
            arr = np.array(emb, dtype="float32")
            if arr.ndim == 1 and arr.size > 0:
                vectors.append(arr)
                valid_indices.append(idx)
    if not valid_indices:
        return 0 if entries else None

    mat = _normalize_rows(np.stack(vectors, axis=0))
    vec = _normalize_vec(np.array(incoming_vec, dtype="float32"))
    k = max(1, min(int(cluster_k), mat.shape[0]))
    seed_idx = _farthest_point_sampling(mat, k)
    centroids = mat[np.array(seed_idx, dtype=int)]
    existing_assign = np.argmax(mat @ centroids.T, axis=1)
    incoming_assign = int(np.argmax(centroids @ vec))
    cluster_local = np.where(existing_assign == incoming_assign)[0]
    if cluster_local.size == 0:
        cluster_local = np.arange(mat.shape[0])
    cluster_vecs = mat[cluster_local]
    center = _normalize_vec(cluster_vecs.mean(axis=0))
    farthest_local = int(np.argmin(cluster_vecs @ center))
    victim_local = int(cluster_local[farthest_local])
    return valid_indices[victim_local]


async def _insert_bootstrap_embedding(
    plu_code: str,
    embedding_vec: np.ndarray,
    status: str,
    filename: Optional[str],
    validation_id: Optional[str],
    source_candidate_id: Optional[str] = None,
) -> None:
    doc = {
        "plu_code": str(plu_code),
        "embedding": _normalize_vec(np.array(embedding_vec, dtype="float32")).tolist(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "source_validation_id": validation_id,
        "status": status,
    }
    if source_candidate_id:
        doc["source_candidate_id"] = source_candidate_id

    target = BOOTSTRAP_FIXED_POOL_SIZE
    if target <= 0:
        await db.bootstrap_embeddings.insert_one(doc)
        return

    async with BOOTSTRAP_POOL_LOCK:
        cursor = db.bootstrap_embeddings.find(
            {"plu_code": str(plu_code)},
            {"_id": 1, "embedding": 1, "created_at": 1},
        ).sort("created_at", -1)
        entries = await cursor.to_list(None)
        remove_needed = max(0, len(entries) - target + 1)
        for _ in range(remove_needed):
            victim_idx = _select_bootstrap_eviction_index(entries, embedding_vec, BOOTSTRAP_CLUSTER_K)
            if victim_idx is None:
                break
            victim = entries.pop(victim_idx)
            victim_id = victim.get("_id")
            if victim_id is not None:
                await db.bootstrap_embeddings.delete_one({"_id": victim_id})
        await db.bootstrap_embeddings.insert_one(doc)


async def _append_embedding_to_store(
    plu_code: str,
    image_base64: str,
    filename: Optional[str],
    source_tag: str,
) -> tuple[str, bool, Optional[float]]:
    store = get_embedding_store()
    cfg = store.config
    if (
        cfg.model_name != EMBEDDING_MODEL_NAME
        or cfg.image_size != EMBEDDING_IMAGE_SIZE
        or cfg.scale_size != EMBEDDING_SCALE_SIZE
        or cfg.crop_mode != EMBEDDING_CROP_MODE
    ):
        raise RuntimeError("Embedding store config does not match current embedding settings.")

    img = decode_base64_to_pil(image_base64)
    engine = get_embedding_engine()
    vec = await asyncio.to_thread(engine.embed_image, img, True)
    vec = vec.astype("float32")

    saved_path = _save_base64_image(CANDIDATE_REF_DIR / str(plu_code), filename, image_base64)

    plu_doc = await db.plu_products.find_one({"plu_code": str(plu_code)}, {"_id": 0, "name": 1})
    plu_name = plu_doc.get("name") if plu_doc else ""

    max_sim = None
    added = True

    async with EMBEDDING_STORE_LOCK:
        store = get_embedding_store()
        plu_indices = store.plu_index.get(str(plu_code), [])
        if CLUSTER_ENABLE and plu_indices:
            existing = store.embeddings[np.array(plu_indices, dtype=int)]
            sims = existing @ vec
            max_sim = float(sims.max())
            if max_sim >= CLUSTER_DUP_SIM:
                added = False

        if added:
            new_idx = int(store.embeddings.shape[0])
            store.embeddings = np.vstack([store.embeddings, vec[None, :]])
            store.meta.append({
                "plu_code": str(plu_code),
                "plu_name": plu_name,
                "filename": Path(saved_path).name,
                "source_path": str(saved_path),
                "source": source_tag,
            })
            store.plu_index.setdefault(str(plu_code), []).append(new_idx)

            # Auto-prune per PLU if budget exceeded (tiered budgets supported)
            plu_indices = store.plu_index.get(str(plu_code), [])
            min_budget, max_budget, _tier = _get_plu_budget(str(plu_code))
            if max_budget > 0 and len(plu_indices) > max_budget:
                keep_count = max(min_budget, min(max_budget, len(plu_indices)))
                plu_embeddings = store.embeddings[np.array(plu_indices, dtype=int)]
                # Ensure newest embedding stays
                local_new = len(plu_indices) - 1
                keep_local = _farthest_point_sampling(plu_embeddings, keep_count, seed_idx=local_new)
                keep_set = set(np.array(plu_indices, dtype=int)[np.array(keep_local, dtype=int)].tolist())
                keep_indices = []
                for idx in range(store.embeddings.shape[0]):
                    if idx in keep_set or store.meta[idx].get("plu_code") != str(plu_code):
                        keep_indices.append(idx)
                new_embeddings = store.embeddings[np.array(keep_indices, dtype=int)]
                new_meta = [store.meta[i] for i in keep_indices]
                new_plu_index = {}
                for new_idx2, old_idx in enumerate(keep_indices):
                    plu = str(new_meta[new_idx2].get("plu_code"))
                    new_plu_index.setdefault(plu, []).append(new_idx2)
                store.embeddings = new_embeddings
                store.meta = new_meta
                store.plu_index = new_plu_index

            from embedding_store import EmbeddingStore

            EmbeddingStore.save(EMBEDDING_STORE_DIR, store.embeddings, store.meta, store.plu_index, store.config)

    return str(saved_path), added, max_sim


async def _bootstrap_update(
    plu_code: str,
    embedding_vec: np.ndarray,
    image_base64: str,
    filename: Optional[str],
    validation_id: Optional[str],
) -> None:
    if not BOOTSTRAP_ENABLE:
        return
    plu_code = str(plu_code)
    # Load recent pool embeddings for this PLU
    cursor = db.bootstrap_embeddings.find({"plu_code": plu_code}, {"_id": 0, "embedding": 1}).sort("created_at", -1).limit(BOOTSTRAP_MAX_POOL)
    items = await cursor.to_list(BOOTSTRAP_MAX_POOL)
    pool = [np.array(item["embedding"], dtype="float32") for item in items if item.get("embedding")]
    if len(pool) < BOOTSTRAP_MIN_COUNT:
        await _insert_bootstrap_embedding(
            plu_code=plu_code,
            embedding_vec=embedding_vec,
            status="warmup",
            filename=filename,
            validation_id=validation_id,
        )
        return

    pool_mat = np.stack(pool, axis=0)
    sims = pool_mat @ embedding_vec
    support_count = int((sims >= BOOTSTRAP_SUPPORT_SIM).sum())
    proto_k = 0
    if BOOTSTRAP_USE_PROTOTYPES and pool_mat.shape[0] > 0:
        proto_k = min(BOOTSTRAP_PROTOTYPE_K, pool_mat.shape[0])
        proto_idx = _farthest_point_sampling(pool_mat, proto_k)
        proto_mat = pool_mat[np.array(proto_idx, dtype=int)]
        proto_sims = proto_mat @ embedding_vec
        k = min(3, proto_sims.size)
        top_idx = np.argpartition(proto_sims, -k)[-k:]
        score = float(proto_sims[top_idx].mean())
    else:
        k = min(3, sims.size)
        top_idx = np.argpartition(sims, -k)[-k:]
        score = float(sims[top_idx].mean())

    status = "pending"
    if score >= BOOTSTRAP_ACCEPT_SIM and support_count >= BOOTSTRAP_SUPPORT_COUNT:
        status = "approved"
    elif score <= BOOTSTRAP_REJECT_SIM:
        status = "rejected"

    candidate = RefCandidate(
        plu_code=plu_code,
        image_base64=image_base64,
        status=status if status != "approved" else "approved",
        reason="bootstrap",
        notes=(
            f"bootstrap_score={score:.4f}; "
            f"support={support_count}/{BOOTSTRAP_SUPPORT_COUNT}; "
            f"proto_k={proto_k}"
        ),
        original_filename=filename,
        source_validation_id=validation_id,
        approved_at=datetime.now(timezone.utc) if status == "approved" else None,
        rejected_at=datetime.now(timezone.utc) if status == "rejected" else None,
        bootstrap_score=score,
        bootstrap_support=support_count,
        bootstrap_proto_k=proto_k if proto_k > 0 else None,
    )
    doc = candidate.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    if doc.get("approved_at"):
        doc["approved_at"] = candidate.approved_at.isoformat()
    if doc.get("rejected_at"):
        doc["rejected_at"] = candidate.rejected_at.isoformat()

    if status == "approved":
        try:
            out_path, added, max_sim = await _append_embedding_to_store(
                plu_code,
                image_base64,
                filename,
                source_tag="bootstrap_auto",
            )
            doc["approved_path"] = str(out_path)
            doc["approved_at"] = datetime.now(timezone.utc).isoformat()
            if not added:
                doc["notes"] = f"{doc.get('notes', '')} | redundant_max_sim={max_sim:.4f}".strip()
        except Exception as exc:
            # Fallback to pending if store update fails
            doc["status"] = "pending"
            doc["approved_at"] = None
            doc["notes"] = f"{doc.get('notes', '')} | auto_store_failed: {exc}"
    status = doc.get("status", status)
    await db.ref_candidates.insert_one(doc)

    if status != "rejected":
        await _insert_bootstrap_embedding(
            plu_code=plu_code,
            embedding_vec=embedding_vec,
            status=status,
            filename=filename,
            validation_id=validation_id,
        )


def _schedule_bootstrap_update(
    plu_code: str,
    embedding_vec: np.ndarray,
    image_base64: str,
    filename: Optional[str],
    validation_id: Optional[str],
) -> None:
    async def _runner():
        try:
            await _bootstrap_update(plu_code, embedding_vec, image_base64, filename, validation_id)
        except Exception as exc:
            logging.error(f"Bootstrap update failed for PLU {plu_code}: {exc}")

    asyncio.create_task(_runner())


async def run_embedding_inference(image_base64: str, plu_product: PLUProduct) -> dict:
    """Run local embedding similarity and return uniform result dict."""
    engine = get_embedding_engine()
    store = get_embedding_store()
    img = decode_base64_to_pil(image_base64)
    query_embeddings = await asyncio.to_thread(engine.extract_embeddings, img, True)
    top_matches = await asyncio.to_thread(
        _build_embedding_top_matches, store, query_embeddings, EMBEDDING_TOP_K, 3
    )
    # Build a single vector for bootstrap usage
    vec = query_embeddings.mean(axis=0)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    try:
        result = await asyncio.to_thread(
            store.score_query,
            query_embeddings,
            plu_product.plu_code,
            EMBEDDING_TOP_K,
            EMBEDDING_MIN_SIM,
            EMBEDDING_MARGIN,
        )
    except Exception as exc:
        msg = str(exc)
        # If PLU is missing in store, still allow bootstrap to run.
        return {
            "analysis": f"Error: {msg}",
            "is_match": False,
            "confidence": 0.0,
            "embedding_vector": vec.astype("float32"),
            "top_matches": top_matches,
        }
    analysis_detail = {
        "selected_plu": str(plu_product.plu_code),
        "selected_score": result.get("selected_score"),
        "best_other_score": result.get("best_other_score"),
        "predicted_plu": result.get("predicted_plu"),
        "predicted_score": result.get("predicted_score"),
        "embedding_count": result.get("embedding_count"),
    }
    analysis = (
        "Embedding match check: "
        f"selected_plu={analysis_detail['selected_plu']}, "
        f"selected_score={analysis_detail['selected_score']:.3f}, "
        f"best_other={analysis_detail['best_other_score']:.3f}, "
        f"predicted_plu={analysis_detail['predicted_plu']}, "
        f"predicted_score={analysis_detail['predicted_score']:.3f}."
    )
    return {
        "analysis": analysis,
        "analysis_selected_plu": str(analysis_detail.get("selected_plu")),
        "analysis_selected_score": f"{analysis_detail.get('selected_score'):.3f}",
        "analysis_best_other_score": f"{analysis_detail.get('best_other_score'):.3f}",
        "analysis_predicted_plu": str(analysis_detail.get("predicted_plu")),
        "analysis_predicted_score": f"{analysis_detail.get('predicted_score'):.3f}",
        "analysis_embedding_count": str(analysis_detail.get("embedding_count")),
        "embedding_vector": vec.astype("float32"),
        "is_match": result["is_match"],
        "confidence": result["confidence"],
        "top_matches": top_matches,
    }

async def run_local_inference(image_base64: str, plu_product: PLUProduct, model_key: str = "local") -> dict:
    """Run offline ONNX model and return uniform result dict."""
    session, input_name, labels, image_size = await asyncio.to_thread(load_local_model, model_key)
    input_tensor = preprocess_for_local_model(image_base64, size=image_size)
    outputs = await asyncio.to_thread(session.run, None, {input_name: input_tensor})
    logits = outputs[0][0]  # assuming (1, num_classes)
    probs = softmax(logits)
    top_matches = _build_local_top_matches(labels, probs, top_n=3)
    top_idx = int(np.argmax(probs))
    pred_label = labels[top_idx]
    pred_plu = str(pred_label.get("plu_code"))
    pred_name = pred_label.get("name", pred_plu)
    selected_plu = str(plu_product.plu_code)
    # Confidence for selected PLU or top-1
    try:
        selected_idx = next(i for i, lbl in enumerate(labels) if str(lbl.get("plu_code")) == selected_plu)
        selected_conf = float(probs[selected_idx]) * 100.0
    except StopIteration:
        selected_conf = float(probs[top_idx]) * 100.0
    response_text = (
        f"Local model ({model_key}) prediction: {pred_name} (PLU {pred_plu}), "
        f"confidence {probs[top_idx]*100:.1f}%. "
        f"Selected PLU {selected_plu} confidence {selected_conf:.1f}%."
    )
    return {
        "analysis": response_text,
        "is_match": pred_plu == selected_plu,
        "confidence": round(selected_conf, 2),
        "top_matches": top_matches,
    }


async def run_butcher_resnet_inference(
    image_base64: str,
    plu_product: PLUProduct,
    top_k: Optional[int] = None,
) -> dict:
    """
    butcher-vision mantığıyla ResNet18 checkpoint inference.
    Config + best.pth + class_to_idx.json kullanır.
    """
    from butcher_runtime import predict_base64_image

    effective_top_k = BUTCHER_TOP_K
    if top_k is not None:
        try:
            effective_top_k = max(1, int(top_k))
        except Exception:
            effective_top_k = BUTCHER_TOP_K

    result = await asyncio.to_thread(
        predict_base64_image,
        image_base64,
        BUTCHER_CONFIG_PATH,
        effective_top_k,
    )
    top_matches = result.get("top_matches", [])
    class_probs = result.get("class_probs", {})
    predicted_class = str(result.get("predicted_class", ""))
    predicted_prob = float(result.get("predicted_prob", 0.0))

    selected_prob, selected_class = _resolve_selected_prob(class_probs, plu_product)
    if selected_prob is None:
        selected_prob = predicted_prob
        selected_class = predicted_class

    predicted_norm = _normalize_label(predicted_class)
    selected_norm_candidates = {
        _normalize_label(str(plu_product.plu_code)),
        _normalize_label(str(plu_product.name)),
    }
    is_match = predicted_norm in selected_norm_candidates
    selected_conf = float(selected_prob) * 100.0

    analysis = (
        f"Butcher ResNet18 prediction: {predicted_class}, "
        f"predicted_prob={predicted_prob:.3f}, "
        f"selected_class={selected_class}, selected_prob={float(selected_prob):.3f}."
    )
    return {
        "analysis": analysis,
        "analysis_selected_plu": str(selected_class or plu_product.plu_code),
        "analysis_selected_score": f"{float(selected_prob):.3f}",
        "analysis_best_other_score": "",
        "analysis_predicted_plu": predicted_class,
        "analysis_predicted_score": f"{predicted_prob:.3f}",
        "analysis_embedding_count": "",
        "is_match": is_match,
        "confidence": round(selected_conf, 2),
        "top_matches": top_matches,
    }

def build_gemini_prompt(plu_product: PLUProduct) -> str:
    """Construct a strict, deterministic prompt for Gemini."""
    return (
        "You are an expert grocery product verifier for scales.\n"
        "Expected Product:\n"
        f"- PLU: {plu_product.plu_code}\n"
        f"- Name: {plu_product.name}\n"
        f"- Description: {plu_product.description}\n\n"
        "Rules:\n"
        "- Decide match based on what is visible in the image(s).\n"
        "- Items are often in transparent plastic bags; ignore bag reflections, folds, or glare.\n"
        "- Focus on visible color/shape/texture; do NOT require seeing stem/calyx or full surface if the bag obscures it.\n"
        "- If visible cues (color, overall shape/size) are consistent with the expected product, answer Match: Yes.\n"
        "- For small clustered items (e.g., cherry tomatoes), individual stems need not be visible; small/oval red form is enough.\n"
        "- For long green produce (e.g., cucumbers/Çengelköy), stems need not be visible; long thin cylinder + green color is enough.\n"
        "- Respect size/shape cues: if expected is thin/pointed pepper but observed peppers are thick/stubby or bell-like, answer Match: No.\n"
        "- Only answer Match: No if visible cues clearly contradict the expected product (wrong color/shape/category).\n"
        "- If uncertain, answer Match: No.\n"
        "- Keep answers short and deterministic.\n\n"
        "Respond exactly in this format:\n"
        "Product Seen: <short description>\n"
        "Match: <Yes/No>\n"
        "Confidence: <0-100>%\n"
        "Reasoning: <one brief sentence>"
    )

def load_reference_images(plu_product: PLUProduct) -> List[bytes]:
    """Load up to REFERENCE_MAX_IMAGES reference images for the given PLU."""
    refs: List[bytes] = []
    candidates = [
        REFERENCE_IMAGE_DIR / str(plu_product.plu_code),
        REFERENCE_IMAGE_DIR / plu_product.name.replace(" ", "_"),
    ]
    seen = set()
    for folder in candidates:
        if not folder.exists() or not folder.is_dir():
            continue
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if img_path in seen:
                continue
            try:
                data = img_path.read_bytes()
                refs.append(data)
                seen.add(img_path)
                if len(refs) >= REFERENCE_MAX_IMAGES:
                    return refs
            except Exception as e:
                logging.warning(f"Failed to read reference image {img_path}: {e}")
    return refs

async def run_remote_inference(provider: str, model_name: str, image_base64: str, plu_product: PLUProduct) -> dict:
    """Run Gemini or OpenAI inference and return uniform result dict."""
    # Determine which API key to use and prepare client (remote providers)
    if provider == 'gemini':
        api_key = GOOGLE_API_KEY
        model = model_name.strip() if model_name and model_name.strip() else 'gemini-2.0-flash'
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set for Gemini provider")
        gemini_client = genai.Client(api_key=api_key)
    elif provider == 'openai':
        api_key = OPENAI_API_KEY
        model = model_name if model_name and model_name.startswith(('gpt', 'o')) else 'gpt-5'
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider")
        openai_client = AsyncOpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unsupported remote AI provider: {provider}")

    prompt_text = build_gemini_prompt(plu_product)

    if provider == 'gemini':
        image_bytes = base64.b64decode(image_base64)
        ref_bytes_list = load_reference_images(plu_product)
        parts = [Part.from_text(text=prompt_text)]
        for ref in ref_bytes_list:
            parts.append(Part.from_bytes(data=ref, mime_type="image/jpeg"))
        parts.append(Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
        user_content = Content(role="user", parts=parts)
        genai_response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=model,
            contents=[user_content],
        )
        response_text = getattr(genai_response, "text", str(genai_response))
    else:  # openai
        system_message = "You are an expert grocery product verifier for scales."
        openai_response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                },
            ],
        )
        message_content = openai_response.choices[0].message.content
        if isinstance(message_content, list):
            response_text = " ".join(
                part.get("text", "") for part in message_content if isinstance(part, dict)
            ).strip()
        else:
            response_text = message_content or ""

    logging.info(f"Using AI Provider: {provider}, Model: {model}")

    # Parse response
    response_lower = response_text.lower()
    is_match = "match: yes" in response_lower or "match:yes" in response_lower
    confidence = 50.0
    try:
        import re
        match = re.search(r"confidence[:\s]+([0-9]+\.?[0-9]*)", response_lower)
        if match:
            confidence = float(match.group(1))
    except Exception:
        confidence = 50.0

    return {
        "analysis": response_text,
        "is_match": is_match,
        "confidence": confidence,
        "reference_count": len(ref_bytes_list) if provider == 'gemini' else None,
    }

# AI Analysis function
async def analyze_image_with_ai(
    image_base64: str,
    plu_product: PLUProduct,
    resnet_top_k: Optional[int] = None,
) -> dict:
    """Analyze image using configured AI provider."""
    try:
        provider = AI_PROVIDER
        model_name = AI_MODEL

        # 0) Butcher-style ResNet18 classifier
        if provider == 'butcher_resnet':
            return await run_butcher_resnet_inference(
                image_base64,
                plu_product,
                top_k=resnet_top_k,
            )

        # 1) Only local (small)
        if provider == 'local':
            return await run_local_inference(image_base64, plu_product, model_key="local")

        # 2) Only local (large)
        if provider == 'local_large':
            return await run_local_inference(image_base64, plu_product, model_key="local_large")

        # 3) Local embedding store (offline similarity)
        if provider == 'local_embedding':
            return await run_embedding_inference(image_base64, plu_product)

        # 4) Local first (small), Gemini if mismatch
        if provider == 'local_gemini':
            local_res = await run_local_inference(image_base64, plu_product, model_key="local")
            if local_res["is_match"]:
                local_res["analysis"] += " | Gemini skipped because local matched."
                local_res["fallback_local_match"] = local_res["is_match"]
                local_res["fallback_local_confidence"] = local_res["confidence"]
                local_res["fallback_remote_provider"] = "gemini"
                local_res["fallback_remote_match"] = None
                local_res["fallback_remote_confidence"] = None
                local_res["reference_count"] = None
                return local_res
            gemini_res = await run_remote_inference('gemini', model_name, image_base64, plu_product)
            combined_analysis = (
                f"Local mismatch -> {local_res['analysis']} | Gemini -> {gemini_res['analysis']}"
            )
            return {
                "analysis": combined_analysis,
                "is_match": gemini_res["is_match"],
                "confidence": gemini_res["confidence"],
                "fallback_local_match": local_res["is_match"],
                "fallback_local_confidence": local_res["confidence"],
                "fallback_remote_provider": "gemini",
                "fallback_remote_match": gemini_res["is_match"],
                "fallback_remote_confidence": gemini_res["confidence"],
                "reference_count": gemini_res.get("reference_count"),
            }

        # 5) Local + Gemini consensus (always run both)
        if provider == 'local_gemini_consensus':
            local_key = "local_large" if has_local_model("local_large") else "local"
            local_res = await run_local_inference(image_base64, plu_product, model_key=local_key)
            gemini_res = await run_remote_inference('gemini', model_name, image_base64, plu_product)
            combined_analysis = (
                f"Consensus: local({local_key}) -> {local_res['analysis']} | Gemini -> {gemini_res['analysis']}"
            )
            consensus_match = local_res["is_match"] and gemini_res["is_match"]
            consensus_conf = round(min(local_res["confidence"], gemini_res["confidence"]), 2)
            return {
                "analysis": combined_analysis,
                "is_match": consensus_match,
                "confidence": consensus_conf,
                "fallback_local_match": local_res["is_match"],
                "fallback_local_confidence": local_res["confidence"],
                "fallback_remote_provider": "gemini",
                "fallback_remote_match": gemini_res["is_match"],
                "fallback_remote_confidence": gemini_res["confidence"],
                "reference_count": gemini_res.get("reference_count"),
            }

        # 6) Remote single provider (gemini or openai)
        return await run_remote_inference(provider, model_name, image_base64, plu_product)

    except Exception as e:
        logging.error(f"AI analysis error: {e}")
        return {
            "analysis": f"Error: {str(e)}",
            "is_match": False,
            "confidence": 0.0
        }

# Routes
@api_router.get("/")
async def root():
    return {"message": "Terazi AI System", "mode": SYSTEM_MODE}

@api_router.get("/roi/config")
async def get_roi_config():
    cfg = _load_tray_roi(force_reload=True)
    return {
        "enabled": TRAY_ROI_ENABLE,
        "strict": TRAY_ROI_STRICT,
        "path": str(TRAY_ROI_PATH),
        "exists": TRAY_ROI_PATH.exists(),
        "config": cfg,
    }

@api_router.post("/roi/config")
async def set_roi_config(payload: TrayROIConfigUpdate):
    try:
        cfg = _validate_tray_roi_config(payload.model_dump())
        _save_tray_roi(cfg)
        return {
            "enabled": TRAY_ROI_ENABLE,
            "strict": TRAY_ROI_STRICT,
            "path": str(TRAY_ROI_PATH),
            "exists": True,
            "config": cfg,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ROI config: {exc}")

# PLU Management
@api_router.post("/plu/create", response_model=PLUProduct)
async def create_plu(input: PLUProductCreate):
    plu_obj = PLUProduct(**input.model_dump())
    doc = plu_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.plu_products.insert_one(doc)
    return plu_obj

@api_router.get("/plu/list", response_model=List[PLUProduct])
async def get_plu_list():
    plus = await db.plu_products.find({}, {"_id": 0}).to_list(1000)
    for plu in plus:
        if isinstance(plu['created_at'], str):
            plu['created_at'] = datetime.fromisoformat(plu['created_at'])
    return plus

@api_router.delete("/plu/delete/{plu_code}")
async def delete_plu(plu_code: str):
    result = await db.plu_products.delete_one({"plu_code": plu_code})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="PLU not found")
    return {"message": "PLU deleted successfully"}

# PLU Selection & Image Capture
@api_router.post("/plu/select")
async def select_plu(selection: PLUSelection, background_tasks: BackgroundTasks):
    global SYSTEM_MODE
    
    # Get PLU product
    plu_product = await db.plu_products.find_one({"plu_code": selection.plu_code}, {"_id": 0})
    if not plu_product:
        raise HTTPException(status_code=404, detail="PLU not found")
    
    # Convert datetime if needed
    if isinstance(plu_product['created_at'], str):
        plu_product['created_at'] = datetime.fromisoformat(plu_product['created_at'])
    plu_obj = PLUProduct(**plu_product)
    
    # Capture image
    image_base64 = capture_image_from_camera()
    if not image_base64:
        raise HTTPException(status_code=500, detail="Camera not available or failed to capture image")
    
    # Save captured image
    captured_img = CapturedImage(
        plu_code=selection.plu_code,
        image_base64=image_base64,
        phase=SYSTEM_MODE,
        ai_provider=AI_PROVIDER,
        ai_model=AI_MODEL
    )
    doc = captured_img.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    await db.captured_images.insert_one(doc)
    
    # If in production mode, run AI analysis
    if SYSTEM_MODE == "production":
        # Run AI analysis in background
        async def run_analysis():
            processed_image_base64, _roi_meta = preprocess_image_for_ai(image_base64)
            start = time.monotonic()
            result = await analyze_image_with_ai(processed_image_base64, plu_obj)
            elapsed_ms = (time.monotonic() - start) * 1000.0
            validation = ValidationResult(
                plu_code=selection.plu_code,
                selected_plu_name=plu_obj.name,
                image_base64=image_base64,
                processed_image_base64=processed_image_base64 if _roi_meta.get("roi_applied") else None,
                roi_applied=bool(_roi_meta.get("roi_applied")),
                roi_reason=str(_roi_meta.get("roi_reason", "")),
                roi_width=int(_roi_meta["roi_width"]) if _roi_meta.get("roi_width") is not None else None,
                roi_height=int(_roi_meta["roi_height"]) if _roi_meta.get("roi_height") is not None else None,
                ai_analysis=result["analysis"],
                analysis_selected_plu=result.get("analysis_selected_plu"),
                analysis_selected_score=result.get("analysis_selected_score"),
                analysis_best_other_score=result.get("analysis_best_other_score"),
                analysis_predicted_plu=result.get("analysis_predicted_plu"),
                analysis_predicted_score=result.get("analysis_predicted_score"),
                analysis_embedding_count=result.get("analysis_embedding_count"),
                top_matches=result.get("top_matches"),
                is_match=result["is_match"],
                confidence=result["confidence"],
                ai_provider=AI_PROVIDER,
                ai_model=AI_MODEL,
                processing_ms=round(elapsed_ms, 2),
                fallback_local_match=result.get("fallback_local_match"),
                fallback_local_confidence=result.get("fallback_local_confidence"),
                fallback_remote_provider=result.get("fallback_remote_provider"),
                fallback_remote_match=result.get("fallback_remote_match"),
                fallback_remote_confidence=result.get("fallback_remote_confidence"),
                reference_count=result.get("reference_count"),
            )
            doc = validation.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            await db.validation_results.insert_one(doc)
            if BOOTSTRAP_ENABLE and result.get("embedding_vector") is not None:
                _schedule_bootstrap_update(
                    selection.plu_code,
                    result.get("embedding_vector"),
                    processed_image_base64,
                    None,
                    validation.id,
                )
        
        background_tasks.add_task(run_analysis)
        
        return {
            "message": "PLU selected, image captured, AI analysis started",
            "plu_code": selection.plu_code,
            "mode": SYSTEM_MODE
        }
    else:
        return {
            "message": "PLU selected, image captured (training mode)",
            "plu_code": selection.plu_code,
            "mode": SYSTEM_MODE,
            "images_collected": await db.captured_images.count_documents({"plu_code": selection.plu_code, "phase": "training"})
        }

# Manual camera test
@api_router.post("/camera/test")
async def test_camera():
    image_base64 = capture_image_from_camera()
    if not image_base64:
        raise HTTPException(status_code=500, detail="Camera not available or failed to capture image")
    return {"message": "Camera test successful", "image_preview": image_base64[:100] + "..."}

@api_router.post("/live/validate")
async def live_validate(payload: LiveValidateRequest):
    """Capture one live frame and return immediate AI validation output."""
    plu_product = await db.plu_products.find_one({"plu_code": payload.plu_code}, {"_id": 0})
    if not plu_product:
        raise HTTPException(status_code=404, detail="PLU not found")
    if isinstance(plu_product.get("created_at"), str):
        plu_product["created_at"] = datetime.fromisoformat(plu_product["created_at"])
    plu_obj = PLUProduct(**plu_product)

    image_base64 = capture_image_from_camera(camera_index=payload.camera_index, warmup_frames=0)
    if not image_base64:
        raise HTTPException(status_code=500, detail="Camera not available or failed to capture image")

    captured_image_id = None
    if payload.persist_capture:
        captured_img = CapturedImage(
            plu_code=payload.plu_code,
            image_base64=image_base64,
            phase=SYSTEM_MODE,
            ai_provider=AI_PROVIDER,
            ai_model=AI_MODEL,
        )
        captured_doc = captured_img.model_dump()
        captured_doc["timestamp"] = captured_doc["timestamp"].isoformat()
        await db.captured_images.insert_one(captured_doc)
        captured_image_id = captured_img.id

    processed_image_base64, _roi_meta = preprocess_image_for_ai(image_base64)

    start = time.monotonic()
    result = await analyze_image_with_ai(processed_image_base64, plu_obj)
    elapsed_ms = (time.monotonic() - start) * 1000.0

    validation_id = None
    if payload.persist_validation:
        validation = ValidationResult(
            plu_code=payload.plu_code,
            selected_plu_name=plu_obj.name,
            image_base64=image_base64,
            processed_image_base64=processed_image_base64 if _roi_meta.get("roi_applied") else None,
            roi_applied=bool(_roi_meta.get("roi_applied")),
            roi_reason=str(_roi_meta.get("roi_reason", "")),
            roi_width=int(_roi_meta["roi_width"]) if _roi_meta.get("roi_width") is not None else None,
            roi_height=int(_roi_meta["roi_height"]) if _roi_meta.get("roi_height") is not None else None,
            ai_analysis=result.get("analysis", ""),
            analysis_selected_plu=result.get("analysis_selected_plu"),
            analysis_selected_score=result.get("analysis_selected_score"),
            analysis_best_other_score=result.get("analysis_best_other_score"),
            analysis_predicted_plu=result.get("analysis_predicted_plu"),
            analysis_predicted_score=result.get("analysis_predicted_score"),
            analysis_embedding_count=result.get("analysis_embedding_count"),
            top_matches=result.get("top_matches"),
            is_match=bool(result.get("is_match", False)),
            confidence=float(result.get("confidence", 0.0)),
            ai_provider=AI_PROVIDER,
            ai_model=AI_MODEL,
            processing_ms=round(elapsed_ms, 2),
            fallback_local_match=result.get("fallback_local_match"),
            fallback_local_confidence=result.get("fallback_local_confidence"),
            fallback_remote_provider=result.get("fallback_remote_provider"),
            fallback_remote_match=result.get("fallback_remote_match"),
            fallback_remote_confidence=result.get("fallback_remote_confidence"),
            reference_count=result.get("reference_count"),
            source="live_validate",
        )
        validation_doc = validation.model_dump()
        validation_doc["timestamp"] = validation_doc["timestamp"].isoformat()
        await db.validation_results.insert_one(validation_doc)
        validation_id = validation.id

        if BOOTSTRAP_ENABLE and result.get("embedding_vector") is not None:
            _schedule_bootstrap_update(
                payload.plu_code,
                result.get("embedding_vector"),
                processed_image_base64,
                None,
                validation.id,
            )

    return {
        "plu_code": payload.plu_code,
        "plu_name": plu_obj.name,
        "is_match": bool(result.get("is_match", False)),
        "confidence": float(result.get("confidence", 0.0)),
        "analysis": result.get("analysis", ""),
        "analysis_selected_plu": result.get("analysis_selected_plu"),
        "analysis_selected_score": result.get("analysis_selected_score"),
        "analysis_best_other_score": result.get("analysis_best_other_score"),
        "analysis_predicted_plu": result.get("analysis_predicted_plu"),
        "analysis_predicted_score": result.get("analysis_predicted_score"),
        "analysis_embedding_count": result.get("analysis_embedding_count"),
        "top_matches": result.get("top_matches"),
        "ai_provider": AI_PROVIDER,
        "ai_model": AI_MODEL,
        "processing_ms": round(elapsed_ms, 2),
        "roi_applied": bool(_roi_meta.get("roi_applied")),
        "roi_reason": str(_roi_meta.get("roi_reason", "")),
        "roi_width": _roi_meta.get("roi_width"),
        "roi_height": _roi_meta.get("roi_height"),
        "captured_image_id": captured_image_id,
        "validation_id": validation_id,
        "image_base64": image_base64,
        "processed_image_base64": processed_image_base64 if _roi_meta.get("roi_applied") else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@api_router.post("/live/predict")
async def live_predict(payload: LivePredictRequest):
    """Live top-3 prediction without requiring a selected PLU."""
    supported_live_providers = {
        "local",
        "local_large",
        "local_embedding",
        "butcher_resnet",
    }
    if AI_PROVIDER not in supported_live_providers:
        raise HTTPException(
            status_code=400,
            detail=(
                "Live prediction supports only local/local_large/local_embedding/"
                "butcher_resnet providers. Current provider: "
                f"{AI_PROVIDER}"
            ),
        )

    image_base64 = payload.image_base64
    if not image_base64:
        image_base64 = capture_image_from_camera(camera_index=payload.camera_index, warmup_frames=0)
    if not image_base64:
        raise HTTPException(status_code=500, detail="Camera not available or failed to capture image")

    processed_image_base64, _roi_meta = preprocess_image_for_ai(image_base64)
    probe_plu = PLUProduct(
        plu_code="__live__",
        name="__live__",
        description="live prediction placeholder",
    )

    start = time.monotonic()
    result = await analyze_image_with_ai(processed_image_base64, probe_plu)
    elapsed_ms = (time.monotonic() - start) * 1000.0

    top_matches = result.get("top_matches")
    if not isinstance(top_matches, list):
        top_matches = []
    top_matches = top_matches[:3]

    predicted_plu = result.get("analysis_predicted_plu")
    predicted_score = result.get("analysis_predicted_score")
    if not predicted_plu and top_matches:
        predicted_plu = str(top_matches[0].get("plu_code") or "")
    if not predicted_score and top_matches:
        score = top_matches[0].get("score")
        predicted_score = str(score) if score is not None else None

    return {
        "analysis": result.get("analysis", ""),
        "top_matches": top_matches,
        "analysis_predicted_plu": predicted_plu,
        "analysis_predicted_score": predicted_score,
        "confidence": float(result.get("confidence", 0.0)),
        "ai_provider": AI_PROVIDER,
        "ai_model": AI_MODEL,
        "processing_ms": round(elapsed_ms, 2),
        "roi_applied": bool(_roi_meta.get("roi_applied")),
        "roi_reason": str(_roi_meta.get("roi_reason", "")),
        "roi_width": _roi_meta.get("roi_width"),
        "roi_height": _roi_meta.get("roi_height"),
        "image_base64": image_base64,
        "processed_image_base64": processed_image_base64 if _roi_meta.get("roi_applied") else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# Get captured images
@api_router.get("/images/captured")
async def get_captured_images(
    plu_code: Optional[str] = None, limit: int = 50, skip: int = 0
):
    query = {}
    if plu_code:
        query["plu_code"] = plu_code
    
    if limit < 1:
        limit = 1
    if skip < 0:
        skip = 0

    cursor = db.captured_images.find(query, {"_id": 0}).sort("timestamp", -1)
    if skip:
        cursor = cursor.skip(skip)
    images = await cursor.to_list(limit)
    for img in images:
        if isinstance(img['timestamp'], str):
            img['timestamp'] = datetime.fromisoformat(img['timestamp'])
        # Don't send full base64 in list view, just preview
        if 'image_base64' in img:
            img['has_image'] = True
            img['image_preview'] = img['image_base64'][:100]
            del img['image_base64']
        # Ensure AI metadata exists
        if 'ai_provider' not in img:
            img['ai_provider'] = None
        if 'ai_model' not in img:
            img['ai_model'] = None
    return images

# Get single image with full data
@api_router.get("/images/{image_id}")
async def get_image(image_id: str):
    image = await db.captured_images.find_one({"id": image_id}, {"_id": 0})
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    if isinstance(image['timestamp'], str):
        image['timestamp'] = datetime.fromisoformat(image['timestamp'])
    return image

# ResNet-only synchronous top-k prediction for VB file-based integration
@api_router.post("/resnet/topk-sync")
async def resnet_topk_sync(payload: ResnetTopKSyncRequest):
    image_base64 = None
    filename = payload.file_name or payload.filename

    if payload.file_path:
        file_path = Path(payload.file_path)
        if not file_path.is_absolute():
            file_path = ALLOWED_IMAGE_DIR / file_path
        safe_path = resolve_safe_path(file_path)
        if not safe_path.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {safe_path}")
        try:
            file_bytes = safe_path.read_bytes()
            image_base64 = base64.b64encode(file_bytes).decode("utf-8")
            if not filename:
                filename = safe_path.name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not image_base64 and payload.image_base64:
        image_base64 = payload.image_base64

    if not image_base64:
        if filename:
            file_path = ALLOWED_IMAGE_DIR / filename
            safe_path = resolve_safe_path(file_path)
            if not safe_path.exists():
                raise HTTPException(status_code=400, detail=f"File not found: {safe_path}")
            try:
                file_bytes = safe_path.read_bytes()
                image_base64 = base64.b64encode(file_bytes).decode("utf-8")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
        else:
            latest_path = get_latest_image_path(ALLOWED_IMAGE_DIR)
            try:
                file_bytes = latest_path.read_bytes()
                image_base64 = base64.b64encode(file_bytes).decode("utf-8")
                if not filename:
                    filename = latest_path.name
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read latest file: {e}")

    processed_image_base64, _roi_meta = preprocess_image_for_ai(image_base64)
    top_k = min(max(1, int(payload.top_k)), 20)

    from butcher_runtime import predict_base64_image

    start = time.monotonic()
    result = await asyncio.to_thread(
        predict_base64_image,
        processed_image_base64,
        BUTCHER_CONFIG_PATH,
        top_k,
    )
    elapsed_ms = (time.monotonic() - start) * 1000.0

    top_matches = result.get("top_matches")
    if not isinstance(top_matches, list):
        top_matches = []
    top_matches = top_matches[:top_k]

    top_plu_codes: List[str] = []
    for item in top_matches:
        code = str(item.get("plu_code") or "").strip()
        if code:
            top_plu_codes.append(code)

    predicted_plu = str(
        result.get("predicted_class")
        or (top_plu_codes[0] if top_plu_codes else "")
    )
    predicted_prob = float(result.get("predicted_prob", 0.0))

    return {
        "top_k": top_k,
        "top_plu_codes": top_plu_codes,
        "top_plu_codes_csv": ",".join(top_plu_codes),
        "top_matches": top_matches,
        "analysis_predicted_plu": predicted_plu,
        "analysis_predicted_score": f"{predicted_prob:.3f}",
        "ai_provider": "butcher_resnet",
        "ai_model": "",
        "processing_ms": round(elapsed_ms, 2),
        "roi_applied": bool(_roi_meta.get("roi_applied")),
        "roi_reason": str(_roi_meta.get("roi_reason", "")),
        "roi_width": _roi_meta.get("roi_width"),
        "roi_height": _roi_meta.get("roi_height"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
    }

# Synchronous validation: capture image, run AI, optionally save result
@api_router.post("/validate-sync")
async def validate_sync(payload: ValidateSyncRequest):
    global SYSTEM_MODE
    # Find PLU
    plu_product = await db.plu_products.find_one({"plu_code": payload.plu_code}, {"_id": 0})
    if not plu_product:
        if AI_PROVIDER != "butcher_resnet":
            raise HTTPException(status_code=404, detail="PLU not found")
        # ResNet top-k mode can run without a DB PLU row.
        plu_obj = PLUProduct(
            plu_code=str(payload.plu_code),
            name=str(payload.plu_code),
            description="resnet_topk_placeholder",
        )
    else:
        if isinstance(plu_product['created_at'], str):
            plu_product['created_at'] = datetime.fromisoformat(plu_product['created_at'])
        plu_obj = PLUProduct(**plu_product)

    # Determine image source: file_path -> image_base64 -> camera
    image_base64 = None
    filename = payload.file_name or payload.filename

    if payload.file_path:
        file_path = Path(payload.file_path)
        # relative ise ALLOWED_IMAGE_DIR altÄ±ndan Ã§Ã¶z
        if not file_path.is_absolute():
            file_path = ALLOWED_IMAGE_DIR / file_path
        safe_path = resolve_safe_path(file_path)
        if not safe_path.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {safe_path}")
        try:
            file_bytes = safe_path.read_bytes()
            image_base64 = base64.b64encode(file_bytes).decode("utf-8")
            if not filename:
                filename = safe_path.name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not image_base64 and payload.image_base64:
        image_base64 = payload.image_base64

    if not image_base64:
        if filename:
            # If filename provided, read that file from ALLOWED_IMAGE_DIR
            file_path = ALLOWED_IMAGE_DIR / filename
            safe_path = resolve_safe_path(file_path)
            if not safe_path.exists():
                raise HTTPException(status_code=400, detail=f"File not found: {safe_path}")
            try:
                file_bytes = safe_path.read_bytes()
                image_base64 = base64.b64encode(file_bytes).decode("utf-8")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
        else:
            # If file_path/filename not provided, use latest image from ALLOWED_IMAGE_DIR
            latest_path = get_latest_image_path(ALLOWED_IMAGE_DIR)
            try:
                file_bytes = latest_path.read_bytes()
                image_base64 = base64.b64encode(file_bytes).decode("utf-8")
                if not filename:
                    filename = latest_path.name
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read latest file: {e}")
    processed_image_base64, _roi_meta = preprocess_image_for_ai(image_base64)

    # ResNet-only response mode for VB compatibility on the same endpoint.
    if AI_PROVIDER == "butcher_resnet":
        from butcher_runtime import predict_base64_image

        top_k = 5
        start = time.monotonic()
        resnet_result = await asyncio.to_thread(
            predict_base64_image,
            processed_image_base64,
            BUTCHER_CONFIG_PATH,
            top_k,
        )
        elapsed_ms = (time.monotonic() - start) * 1000.0

        top_matches = resnet_result.get("top_matches")
        if not isinstance(top_matches, list):
            top_matches = []
        top_matches = top_matches[:top_k]

        top_plu_codes: List[str] = []
        for item in top_matches:
            code = str(item.get("plu_code") or "").strip()
            if code:
                top_plu_codes.append(code)

        predicted_plu = str(
            resnet_result.get("predicted_class")
            or (top_plu_codes[0] if top_plu_codes else "")
        )
        predicted_prob = float(resnet_result.get("predicted_prob", 0.0))
        predicted_prob_pct = round(predicted_prob * 100.0, 2)
        analysis_text = f"ResNet top-{top_k}: {','.join(top_plu_codes)}"

        validation = ValidationResult(
            plu_code=str(payload.plu_code),
            selected_plu_name=str(plu_obj.name),
            image_base64=image_base64,
            processed_image_base64=processed_image_base64 if _roi_meta.get("roi_applied") else None,
            roi_applied=bool(_roi_meta.get("roi_applied")),
            roi_reason=str(_roi_meta.get("roi_reason", "")),
            roi_width=int(_roi_meta["roi_width"]) if _roi_meta.get("roi_width") is not None else None,
            roi_height=int(_roi_meta["roi_height"]) if _roi_meta.get("roi_height") is not None else None,
            ai_analysis=analysis_text,
            analysis_selected_plu="",
            analysis_selected_score="",
            analysis_best_other_score="",
            analysis_predicted_plu=predicted_plu,
            analysis_predicted_score=f"{predicted_prob:.3f}",
            analysis_embedding_count="",
            top_matches=top_matches,
            is_match=False,
            confidence=predicted_prob_pct,
            ai_provider="butcher_resnet",
            ai_model="",
            processing_ms=round(elapsed_ms, 2),
            source="resnet_topk",
            original_filename=filename,
        )
        doc = validation.model_dump()
        doc["timestamp"] = doc["timestamp"].isoformat()
        await db.validation_results.insert_one(doc)

        response_payload = _stringify_response({
            "is_match": False,
            "confidence": predicted_prob_pct,
            "analysis": analysis_text,
            "analysis_selected_plu": "",
            "analysis_selected_score": "",
            "analysis_best_other_score": "",
            "analysis_predicted_plu": predicted_plu,
            "analysis_predicted_score": f"{predicted_prob:.3f}",
            "analysis_embedding_count": "",
            "top_matches": top_matches,
            "top_plu_codes": top_plu_codes,
            "top_plu_codes_csv": ",".join(top_plu_codes),
            "top1_plu": top_plu_codes[0] if len(top_plu_codes) > 0 else "",
            "top2_plu": top_plu_codes[1] if len(top_plu_codes) > 1 else "",
            "top3_plu": top_plu_codes[2] if len(top_plu_codes) > 2 else "",
            "top4_plu": top_plu_codes[3] if len(top_plu_codes) > 3 else "",
            "top5_plu": top_plu_codes[4] if len(top_plu_codes) > 4 else "",
            "ai_provider": "butcher_resnet",
            "ai_model": "",
            "processing_ms": round(elapsed_ms, 2),
            "roi_applied": bool(_roi_meta.get("roi_applied")),
            "roi_reason": str(_roi_meta.get("roi_reason", "")),
            "roi_width": _roi_meta.get("roi_width"),
            "roi_height": _roi_meta.get("roi_height"),
            "timestamp": doc["timestamp"],
            "validation_id": validation.id,
            "filename": filename,
        })
        # VB tarafi is_match alanini boolean olarak deserialize ediyor.
        # ResNet modunda uyumlu/uyumsuz anlami olmadigi icin her zaman false donuyoruz.
        response_payload["is_match"] = False
        return response_payload

    # Run AI for all non-ResNet providers (existing behavior).
    start = time.monotonic()
    result = await analyze_image_with_ai(processed_image_base64, plu_obj)
    elapsed_ms = (time.monotonic() - start) * 1000.0

    # Persist validation result for traceability
    validation = ValidationResult(
        plu_code=payload.plu_code,
        selected_plu_name=plu_obj.name,
        image_base64=image_base64,
        processed_image_base64=processed_image_base64 if _roi_meta.get("roi_applied") else None,
        roi_applied=bool(_roi_meta.get("roi_applied")),
        roi_reason=str(_roi_meta.get("roi_reason", "")),
        roi_width=int(_roi_meta["roi_width"]) if _roi_meta.get("roi_width") is not None else None,
        roi_height=int(_roi_meta["roi_height"]) if _roi_meta.get("roi_height") is not None else None,
        ai_analysis=result["analysis"],
        analysis_selected_plu=result.get("analysis_selected_plu"),
        analysis_selected_score=result.get("analysis_selected_score"),
        analysis_best_other_score=result.get("analysis_best_other_score"),
        analysis_predicted_plu=result.get("analysis_predicted_plu"),
        analysis_predicted_score=result.get("analysis_predicted_score"),
        analysis_embedding_count=result.get("analysis_embedding_count"),
        top_matches=result.get("top_matches"),
        is_match=result["is_match"],
        confidence=result["confidence"],
        ai_provider=AI_PROVIDER,
        ai_model=AI_MODEL,
        processing_ms=round(elapsed_ms, 2),
        fallback_local_match=result.get("fallback_local_match"),
        fallback_local_confidence=result.get("fallback_local_confidence"),
        fallback_remote_provider=result.get("fallback_remote_provider"),
        fallback_remote_match=result.get("fallback_remote_match"),
        fallback_remote_confidence=result.get("fallback_remote_confidence"),
        reference_count=result.get("reference_count"),
        original_filename=filename,
    )
    doc = validation.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    await db.validation_results.insert_one(doc)

    if BOOTSTRAP_ENABLE and result.get("embedding_vector") is not None:
        _schedule_bootstrap_update(
            payload.plu_code,
            result.get("embedding_vector"),
            processed_image_base64,
            filename,
            validation.id,
        )

    # Return concise response
    return _stringify_response({
        "is_match": result["is_match"],
        "confidence": result["confidence"],
        "analysis": result["analysis"],
        "analysis_selected_plu": result.get("analysis_selected_plu"),
        "analysis_selected_score": result.get("analysis_selected_score"),
        "analysis_best_other_score": result.get("analysis_best_other_score"),
        "analysis_predicted_plu": result.get("analysis_predicted_plu"),
        "analysis_predicted_score": result.get("analysis_predicted_score"),
        "analysis_embedding_count": result.get("analysis_embedding_count"),
        "top_matches": result.get("top_matches"),
        "ai_provider": AI_PROVIDER,
        "ai_model": AI_MODEL,
        "processing_ms": round(elapsed_ms, 2),
        "roi_applied": bool(_roi_meta.get("roi_applied")),
        "roi_reason": str(_roi_meta.get("roi_reason", "")),
        "roi_width": _roi_meta.get("roi_width"),
        "roi_height": _roi_meta.get("roi_height"),
        "timestamp": doc["timestamp"],
        "validation_id": validation.id,
        "filename": filename
    })

@api_router.post("/batch/validate")
async def batch_validate(
    metadata: str = Form(...),
    files: List[UploadFile] = File(...),
    foldering_enabled: bool = Form(False),
):
    """Validate a batch of uploaded photos against expected PLU codes."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    try:
        meta_raw = json.loads(metadata)
        meta_items = [BatchValidationMeta(**item) for item in meta_raw]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata: {e}")

    mapping = {Path(item.filename).name: item.plu_code for item in meta_items}
    batch_id = str(uuid.uuid4())
    results = []
    match_count = 0
    mismatch_count = 0
    error_count = 0
    folder_low_pct = max(0.0, min(BATCH_FOLDERING_LOW_PCT, 100.0))
    folder_high_pct = max(0.0, min(BATCH_FOLDERING_HIGH_PCT, 100.0))
    if folder_high_pct <= folder_low_pct:
        folder_high_pct = min(100.0, folder_low_pct + 1.0)
    foldering_info = {
        "enabled": bool(foldering_enabled),
        "base_dir": str((BATCH_FOLDERING_ROOT / batch_id).resolve()) if foldering_enabled else "",
        "low_threshold_pct": folder_low_pct,
        "high_threshold_pct": folder_high_pct,
        "created_files": 0,
        "failed_files": 0,
        "categories": {},
    }

    for upload in files:
        filename = Path(upload.filename or "uploaded").name
        expected_plu = mapping.get(filename) or mapping.get(upload.filename)
        if not expected_plu:
            results.append({
                "filename": filename,
                "error": "Expected PLU missing",
                "status": "error"
            })
            error_count += 1
            continue

        plu_product = await db.plu_products.find_one({"plu_code": expected_plu}, {"_id": 0})
        if not plu_product:
            results.append({
                "filename": filename,
                "plu_code": expected_plu,
                "error": "PLU not found in database",
                "status": "error"
            })
            error_count += 1
            continue
        if isinstance(plu_product['created_at'], str):
            plu_product['created_at'] = datetime.fromisoformat(plu_product['created_at'])

        file_bytes = await upload.read()
        if not file_bytes:
            results.append({
                "filename": filename,
                "plu_code": expected_plu,
                "error": "Empty file",
                "status": "error"
            })
            error_count += 1
            continue

        try:
            image_base64 = encode_uploaded_image(file_bytes)
        except ValueError as e:
            results.append({
                "filename": filename,
                "plu_code": expected_plu,
                "error": str(e),
                "status": "error"
            })
            error_count += 1
            continue

        processed_image_base64, _roi_meta = preprocess_image_for_ai(image_base64)
        start_t = time.monotonic()
        ai_result = await analyze_image_with_ai(
            processed_image_base64,
            PLUProduct(**plu_product),
            resnet_top_k=5 if AI_PROVIDER == "butcher_resnet" else None,
        )
        elapsed_ms = (time.monotonic() - start_t) * 1000.0

        validation = ValidationResult(
            plu_code=expected_plu,
            selected_plu_name=plu_product["name"],
            image_base64=image_base64,
            processed_image_base64=processed_image_base64 if _roi_meta.get("roi_applied") else None,
            roi_applied=bool(_roi_meta.get("roi_applied")),
            roi_reason=str(_roi_meta.get("roi_reason", "")),
            roi_width=int(_roi_meta["roi_width"]) if _roi_meta.get("roi_width") is not None else None,
            roi_height=int(_roi_meta["roi_height"]) if _roi_meta.get("roi_height") is not None else None,
            ai_analysis=ai_result.get("analysis", ""),
            analysis_selected_plu=ai_result.get("analysis_selected_plu"),
            analysis_selected_score=ai_result.get("analysis_selected_score"),
            analysis_best_other_score=ai_result.get("analysis_best_other_score"),
            analysis_predicted_plu=ai_result.get("analysis_predicted_plu"),
            analysis_predicted_score=ai_result.get("analysis_predicted_score"),
            analysis_embedding_count=ai_result.get("analysis_embedding_count"),
            top_matches=ai_result.get("top_matches"),
            is_match=ai_result.get("is_match", False),
            confidence=ai_result.get("confidence", 0.0),
            ai_provider=AI_PROVIDER,
            ai_model=AI_MODEL,
            processing_ms=round(elapsed_ms, 2),
            fallback_local_match=ai_result.get("fallback_local_match"),
            fallback_local_confidence=ai_result.get("fallback_local_confidence"),
            fallback_remote_provider=ai_result.get("fallback_remote_provider"),
            fallback_remote_match=ai_result.get("fallback_remote_match"),
            fallback_remote_confidence=ai_result.get("fallback_remote_confidence"),
            reference_count=ai_result.get("reference_count"),
            batch_id=batch_id,
            source="batch_upload",
            original_filename=filename
        )
        doc = validation.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.validation_results.insert_one(doc)

        if BOOTSTRAP_ENABLE and ai_result.get("embedding_vector") is not None:
            _schedule_bootstrap_update(
                expected_plu,
                ai_result.get("embedding_vector"),
                processed_image_base64,
                filename,
                validation.id,
            )

        result_payload = {
            "filename": filename,
            "plu_code": expected_plu,
            "is_match": ai_result.get("is_match", False),
            "confidence": ai_result.get("confidence", 0.0),
            "analysis": ai_result.get("analysis", ""),
            "analysis_selected_plu": ai_result.get("analysis_selected_plu"),
            "analysis_selected_score": ai_result.get("analysis_selected_score"),
            "analysis_best_other_score": ai_result.get("analysis_best_other_score"),
            "analysis_predicted_plu": ai_result.get("analysis_predicted_plu"),
            "analysis_predicted_score": ai_result.get("analysis_predicted_score"),
            "analysis_embedding_count": ai_result.get("analysis_embedding_count"),
            "top_matches": ai_result.get("top_matches"),
            "processing_ms": round(elapsed_ms, 2),
            "ai_provider": AI_PROVIDER,
            "ai_model": AI_MODEL,
            "validation_id": validation.id,
            "batch_id": batch_id,
            "status": "match" if ai_result.get("is_match") else "mismatch"
        }
        if foldering_enabled:
            try:
                category, folder_meta = _categorize_batch_result(
                    expected_plu=expected_plu,
                    ai_provider=AI_PROVIDER,
                    top_matches_raw=ai_result.get("top_matches"),
                    low_pct=folder_low_pct,
                    high_pct=folder_high_pct,
                )
                saved_path = _save_batch_file_for_foldering(
                    batch_id=batch_id,
                    expected_plu=expected_plu,
                    category=category,
                    filename=filename,
                    file_bytes=file_bytes,
                )
                foldering_info["created_files"] += 1
                foldering_info["categories"][category] = int(foldering_info["categories"].get(category, 0)) + 1
                result_payload["folder_category"] = category
                result_payload["folder_path"] = str(saved_path)
                result_payload["foldering_meta"] = folder_meta
            except Exception as exc:
                foldering_info["failed_files"] += 1
                logging.error(f"Batch foldering failed for {filename}: {exc}")
                result_payload["folder_category"] = "folder_error"
                result_payload["folder_path"] = ""
                result_payload["foldering_meta"] = {
                    "provider": AI_PROVIDER,
                    "error": str(exc),
                }
        if ai_result.get("fallback_local_match") is not None:
            result_payload["fallback_local_match"] = ai_result.get("fallback_local_match")
            result_payload["fallback_local_confidence"] = ai_result.get("fallback_local_confidence")
            result_payload["fallback_remote_provider"] = ai_result.get("fallback_remote_provider")
            result_payload["fallback_remote_match"] = ai_result.get("fallback_remote_match")
            result_payload["fallback_remote_confidence"] = ai_result.get("fallback_remote_confidence")

        results.append(result_payload)

        if ai_result.get("is_match"):
            match_count += 1
        else:
            mismatch_count += 1

    summary = {
        "total": len(files),
        "processed": len([r for r in results if r.get("status") in {"match", "mismatch"}]),
        "match_count": match_count,
        "mismatch_count": mismatch_count,
        "error_count": error_count,
        "foldering_enabled": bool(foldering_enabled),
    }

    return {
        "batch_id": batch_id,
        "summary": summary,
        "results": results,
        "foldering": foldering_info,
    }


def _update_centroid_rank_job(job_id: str, **fields) -> None:
    job = CENTROID_RANK_JOBS.get(job_id)
    if not job:
        return
    job.update(fields)
    job["updated_at"] = datetime.now(timezone.utc).isoformat()


def _run_centroid_rank_job(job_id: str, payload: dict) -> None:
    cancel_event = CENTROID_RANK_CANCEL_EVENTS.get(job_id)
    _update_centroid_rank_job(
        job_id,
        status="cancelling" if cancel_event and cancel_event.is_set() else "running",
        started_at=datetime.now(timezone.utc).isoformat(),
        error="",
    )
    try:
        from rank_butcher_centroid import run_centroid_ranking

        input_dir = Path(payload["input_dir"]).expanduser().resolve()
        out_dir_raw = str(payload.get("out_dir") or "").strip()
        out_dir = (
            Path(out_dir_raw).expanduser().resolve()
            if out_dir_raw
            else (ROOT_DIR / "centroid_rank_jobs" / job_id).resolve()
        )

        def _progress(processed: int, total: int) -> None:
            _update_centroid_rank_job(
                job_id,
                processed_count=int(processed),
                total_count=int(total),
            )

        def _should_stop() -> bool:
            return bool(cancel_event and cancel_event.is_set())

        summary = run_centroid_ranking(
            config_path=BUTCHER_CONFIG_PATH,
            input_dir=input_dir,
            out_dir=out_dir,
            recursive=bool(payload.get("recursive")),
            copy_mode=str(payload.get("copy_mode") or "bands"),
            bands=int(payload.get("bands") or 5),
            progress_callback=_progress,
            should_stop_callback=_should_stop,
        )
        _update_centroid_rank_job(
            job_id,
            status="complete",
            completed_at=datetime.now(timezone.utc).isoformat(),
            processed_count=int(summary.get("processed_count", 0)),
            total_count=int(summary.get("image_count", 0)),
            result=summary,
            output_dir=summary.get("output_dir"),
            ranking_csv=summary.get("ranking_csv"),
            summary_path=summary.get("summary_path"),
        )
    except Exception as exc:
        if exc.__class__.__name__ == "CentroidRankingCancelled":
            _update_centroid_rank_job(
                job_id,
                status="cancelled",
                completed_at=datetime.now(timezone.utc).isoformat(),
                cancelled_at=datetime.now(timezone.utc).isoformat(),
                error="",
            )
            return
        logging.exception(f"Centroid rank job failed: {job_id}")
        _update_centroid_rank_job(
            job_id,
            status="error",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
    finally:
        CENTROID_RANK_CANCEL_EVENTS.pop(job_id, None)


@api_router.get("/dataset/folders")
async def list_dataset_folders(path: Optional[str] = None):
    roots = _get_centroid_browser_roots()
    root_entries = [_format_folder_entry(root, is_root=True) for root in roots]

    if not str(path or "").strip():
        return {
            "current_path": "",
            "parent_path": None,
            "roots": root_entries,
            "entries": root_entries,
        }

    current_path, current_root = _resolve_centroid_browser_path(path)
    if not current_path.exists() or not current_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {current_path}")

    entries = []
    try:
        for child in current_path.iterdir():
            try:
                child_resolved = child.resolve()
                if not child_resolved.is_dir():
                    continue
                if not any(_path_is_same_or_child(child_resolved, root) for root in roots):
                    continue
                entries.append(_format_folder_entry(child_resolved))
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Folder is not accessible: {current_path}")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Folder cannot be read: {exc}")

    entries.sort(key=lambda item: item["name"].lower())
    parent_path = None
    try:
        parent = current_path.parent.resolve()
        if parent != current_path and _path_is_same_or_child(parent, current_root):
            parent_path = str(parent)
    except Exception:
        parent_path = None

    return {
        "current_path": str(current_path),
        "parent_path": parent_path,
        "roots": root_entries,
        "entries": entries,
    }


@api_router.post("/dataset/centroid-rank")
async def start_centroid_rank(payload: CentroidRankRequest):
    input_raw = str(payload.input_dir or "").strip()
    if not input_raw:
        raise HTTPException(status_code=400, detail="input_dir is required")

    copy_mode = str(payload.copy_mode or "bands").strip()
    if copy_mode not in {"bands", "ranked", "none"}:
        raise HTTPException(status_code=400, detail="copy_mode must be one of: bands, ranked, none")

    bands = max(1, min(int(payload.bands or 5), 20))
    input_dir, _ = _resolve_centroid_browser_path(input_raw)
    if not input_dir.exists() or not input_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Input dir not found: {input_dir}")

    out_dir = str(payload.out_dir or "").strip()
    if out_dir:
        _resolve_centroid_browser_path(out_dir)
    job_id = str(uuid.uuid4())
    request_payload = {
        "input_dir": str(input_dir),
        "out_dir": out_dir,
        "recursive": bool(payload.recursive),
        "copy_mode": copy_mode,
        "bands": bands,
    }
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "out_dir": out_dir,
        "recursive": bool(payload.recursive),
        "copy_mode": copy_mode,
        "bands": bands,
        "processed_count": 0,
        "total_count": 0,
        "error": "",
        "result": None,
    }
    CENTROID_RANK_JOBS[job_id] = job
    CENTROID_RANK_CANCEL_EVENTS[job_id] = threading.Event()
    asyncio.create_task(asyncio.to_thread(_run_centroid_rank_job, job_id, request_payload))
    return job


@api_router.get("/dataset/centroid-rank/{job_id}")
async def get_centroid_rank_job(job_id: str):
    job = CENTROID_RANK_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Centroid rank job not found")
    return job


@api_router.post("/dataset/centroid-rank/{job_id}/cancel")
async def cancel_centroid_rank_job(job_id: str):
    job = CENTROID_RANK_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Centroid rank job not found")

    if job.get("status") not in {"queued", "running", "cancelling"}:
        return job

    cancel_event = CENTROID_RANK_CANCEL_EVENTS.get(job_id)
    if cancel_event is not None:
        cancel_event.set()

    _update_centroid_rank_job(
        job_id,
        status="cancelling",
        cancelled_requested_at=datetime.now(timezone.utc).isoformat(),
    )
    return CENTROID_RANK_JOBS[job_id]

# Get validation results
@api_router.get("/validation/results")
async def get_validation_results(limit: int = 50, skip: int = 0):
    if limit < 1:
        limit = 1
    if skip < 0:
        skip = 0
    cursor = db.validation_results.find({}, {"_id": 0}).sort("timestamp", -1)
    if skip:
        cursor = cursor.skip(skip)
    results = await cursor.to_list(limit)
    for result in results:
        if isinstance(result['timestamp'], str):
            result['timestamp'] = datetime.fromisoformat(result['timestamp'])
        # Don't send full base64 in list view
        if 'image_base64' in result:
            result['has_image'] = True
            del result['image_base64']
        if 'processed_image_base64' in result:
            result['has_processed_image'] = bool(result.get('processed_image_base64'))
            del result['processed_image_base64']
        if 'has_processed_image' not in result:
            result['has_processed_image'] = False
        # Ensure AI metadata exists
        if 'ai_provider' not in result:
            result['ai_provider'] = None
        if 'ai_model' not in result:
            result['ai_model'] = None
        if 'processing_ms' not in result:
            result['processing_ms'] = None
        if 'fallback_local_match' not in result:
            result['fallback_local_match'] = None
        if 'fallback_local_confidence' not in result:
            result['fallback_local_confidence'] = None
        if 'fallback_remote_provider' not in result:
            result['fallback_remote_provider'] = None
        if 'fallback_remote_match' not in result:
            result['fallback_remote_match'] = None
        if 'fallback_remote_confidence' not in result:
            result['fallback_remote_confidence'] = None
        if 'reference_count' not in result:
            result['reference_count'] = None
        if 'top_matches' not in result:
            result['top_matches'] = None
    return results

# Get single validation result with full data
@api_router.get("/validation/{result_id}")
async def get_validation_result(result_id: str):
    result = await db.validation_results.find_one({"id": result_id}, {"_id": 0})
    if not result:
        raise HTTPException(status_code=404, detail="Validation result not found")
    if isinstance(result['timestamp'], str):
        result['timestamp'] = datetime.fromisoformat(result['timestamp'])
    return result

# Ref candidate pool
@api_router.post("/ref-candidates/from-validation", response_model=RefCandidate)
async def create_ref_candidate_from_validation(payload: RefCandidateCreateFromValidation):
    validation = await db.validation_results.find_one({"id": payload.validation_id}, {"_id": 0})
    if not validation:
        raise HTTPException(status_code=404, detail="Validation result not found")
    image_base64 = validation.get("image_base64")
    if not image_base64:
        raise HTTPException(status_code=400, detail="Validation result missing image")
    plu_code = payload.expected_plu or validation.get("plu_code")
    if not plu_code:
        raise HTTPException(status_code=400, detail="PLU code is required")
    candidate = RefCandidate(
        plu_code=str(plu_code),
        image_base64=image_base64,
        reason=payload.reason,
        notes=payload.notes,
        original_filename=validation.get("original_filename") or validation.get("filename"),
        source_validation_id=payload.validation_id,
    )
    doc = candidate.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.ref_candidates.insert_one(doc)
    return candidate


@api_router.post("/ref-candidates/from-file", response_model=RefCandidate)
async def create_ref_candidate_from_file(payload: RefCandidateCreateFromFile):
    if not payload.file_path and not payload.file_name:
        raise HTTPException(status_code=400, detail="file_path or file_name required")
    if payload.file_path:
        file_path = Path(payload.file_path)
        if not file_path.is_absolute():
            file_path = ALLOWED_IMAGE_DIR / file_path
    else:
        file_path = ALLOWED_IMAGE_DIR / payload.file_name
    safe_path = resolve_safe_path(file_path)
    if not safe_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {safe_path}")
    try:
        file_bytes = safe_path.read_bytes()
        image_base64 = base64.b64encode(file_bytes).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
    candidate = RefCandidate(
        plu_code=str(payload.plu_code),
        image_base64=image_base64,
        reason=payload.reason,
        notes=payload.notes,
        original_filename=safe_path.name,
    )
    doc = candidate.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.ref_candidates.insert_one(doc)
    return candidate


@api_router.get("/ref-candidates", response_model=List[RefCandidate])
async def list_ref_candidates(status: Optional[str] = None, limit: int = 50, skip: int = 0):
    query = {}
    if status:
        query["status"] = status
    if limit < 1:
        limit = 1
    if skip < 0:
        skip = 0
    cursor = db.ref_candidates.find(query, {"_id": 0}).sort("created_at", -1)
    if skip:
        cursor = cursor.skip(skip)
    candidates = await cursor.to_list(limit)
    for cand in candidates:
        if isinstance(cand.get("created_at"), str):
            cand["created_at"] = datetime.fromisoformat(cand["created_at"])
        if isinstance(cand.get("approved_at"), str):
            cand["approved_at"] = datetime.fromisoformat(cand["approved_at"])
        if isinstance(cand.get("rejected_at"), str):
            cand["rejected_at"] = datetime.fromisoformat(cand["rejected_at"])
    return [RefCandidate(**cand) for cand in candidates]


@api_router.get("/ref-candidates/stats")
async def ref_candidate_stats():
    pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}},
    ]
    stats = await db.ref_candidates.aggregate(pipeline).to_list(10)
    return {item["_id"]: item["count"] for item in stats}


@api_router.get("/ref-candidates/{candidate_id}", response_model=RefCandidate)
async def get_ref_candidate(candidate_id: str):
    cand = await db.ref_candidates.find_one({"id": candidate_id}, {"_id": 0})
    if not cand:
        raise HTTPException(status_code=404, detail="Ref candidate not found")
    if isinstance(cand.get("created_at"), str):
        cand["created_at"] = datetime.fromisoformat(cand["created_at"])
    if isinstance(cand.get("approved_at"), str):
        cand["approved_at"] = datetime.fromisoformat(cand["approved_at"])
    if isinstance(cand.get("rejected_at"), str):
        cand["rejected_at"] = datetime.fromisoformat(cand["rejected_at"])
    return RefCandidate(**cand)


@api_router.post("/ref-candidates/{candidate_id}/approve", response_model=RefCandidate)
async def approve_ref_candidate(candidate_id: str, payload: RefCandidateReview = RefCandidateReview()):
    cand = await db.ref_candidates.find_one({"id": candidate_id}, {"_id": 0})
    if not cand:
        raise HTTPException(status_code=404, detail="Ref candidate not found")
    if cand.get("status") == "approved":
        return RefCandidate(**cand)
    image_base64 = cand.get("image_base64")
    if not image_base64:
        raise HTTPException(status_code=400, detail="Candidate missing image")
    plu_code = str(cand.get("plu_code"))
    try:
        out_path, added, max_sim = await _append_embedding_to_store(
            plu_code,
            image_base64,
            cand.get("original_filename"),
            source_tag="manual_approve",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    update = {
        "status": "approved",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "approved_path": str(out_path),
        "rejected_at": None,
    }
    if not added:
        update["notes"] = f"{update.get('notes', '')} | redundant_max_sim={max_sim:.4f}".strip()
    if payload.notes:
        update["notes"] = payload.notes
    await db.ref_candidates.update_one({"id": candidate_id}, {"$set": update})
    cand.update(update)
    cand["approved_at"] = datetime.fromisoformat(cand["approved_at"])
    if BOOTSTRAP_ENABLE:
        try:
            img = decode_base64_to_pil(image_base64)
            engine = get_embedding_engine()
            vec = await asyncio.to_thread(engine.embed_image, img, True)
            vec = vec.astype("float32")
            await _insert_bootstrap_embedding(
                plu_code=plu_code,
                embedding_vec=vec,
                status="approved_manual",
                filename=cand.get("original_filename"),
                validation_id=cand.get("source_validation_id"),
                source_candidate_id=candidate_id,
            )
        except Exception as exc:
            logging.error(f"Failed to add approved candidate to bootstrap pool for PLU {plu_code}: {exc}")
    return RefCandidate(**cand)


@api_router.post("/ref-candidates/{candidate_id}/reject", response_model=RefCandidate)
async def reject_ref_candidate(candidate_id: str, payload: RefCandidateReview = RefCandidateReview()):
    cand = await db.ref_candidates.find_one({"id": candidate_id}, {"_id": 0})
    if not cand:
        raise HTTPException(status_code=404, detail="Ref candidate not found")
    if cand.get("status") == "rejected":
        return RefCandidate(**cand)
    update = {
        "status": "rejected",
        "rejected_at": datetime.now(timezone.utc).isoformat(),
    }
    if payload.notes:
        update["notes"] = payload.notes
    await db.ref_candidates.update_one({"id": candidate_id}, {"$set": update})
    cand.update(update)
    cand["rejected_at"] = datetime.fromisoformat(cand["rejected_at"])
    return RefCandidate(**cand)


# System mode management
@api_router.post("/system/mode")
async def update_system_mode(mode_update: SystemModeUpdate):
    global SYSTEM_MODE
    if mode_update.mode not in ["training", "production"]:
        raise HTTPException(status_code=400, detail="Mode must be 'training' or 'production'")
    SYSTEM_MODE = mode_update.mode
    await persist_system_mode(SYSTEM_MODE)
    return {"message": f"System mode updated to {SYSTEM_MODE}", "mode": SYSTEM_MODE}

@api_router.get("/system/mode")
async def get_system_mode():
    global SYSTEM_MODE
    # Ensure we reflect persisted mode if server restarted
    if SYSTEM_MODE not in ["training", "production"]:
        SYSTEM_MODE = await load_system_mode_from_db()
    return {"mode": SYSTEM_MODE}

# AI provider/model management
@api_router.get("/system/ai-config")
async def get_ai_config():
    global AI_PROVIDER, AI_MODEL
    return {"provider": AI_PROVIDER, "model": AI_MODEL}

@api_router.post("/system/ai-config")
async def update_ai_config(config: AIConfigUpdate):
    global AI_PROVIDER, AI_MODEL
    allowed = {
        "gemini",
        "openai",
        "local",
        "local_large",
        "local_embedding",
        "butcher_resnet",
        "local_gemini",
        "local_gemini_consensus",
    }
    if config.provider not in allowed:
        raise HTTPException(status_code=400, detail=f"Provider must be one of {allowed}")
    AI_PROVIDER = config.provider
    # Normalize model choice based on provider
    if AI_PROVIDER in {"local", "local_large", "local_embedding", "butcher_resnet"}:
        AI_MODEL = ""
    elif AI_PROVIDER in {"gemini", "local_gemini", "local_gemini_consensus"}:
        AI_MODEL = config.model if config.model is not None else "gemini-2.5-flash-lite"
    else:  # openai
        if config.model is not None:
            AI_MODEL = config.model
    await persist_ai_config(AI_PROVIDER, AI_MODEL)
    return {"provider": AI_PROVIDER, "model": AI_MODEL}

# Dashboard statistics
@api_router.get("/stats/dashboard", response_model=DashboardStats)
async def get_dashboard_stats():
    total_images = await db.captured_images.count_documents({})
    total_validations = await db.validation_results.count_documents({})
    dino_metrics_filter = {"source": {"$ne": "resnet_topk"}}
    dino_total_validations = await db.validation_results.count_documents(dino_metrics_filter)
    match_query = dict(dino_metrics_filter)
    match_query["is_match"] = True
    mismatch_query = dict(dino_metrics_filter)
    mismatch_query["is_match"] = False
    match_count = await db.validation_results.count_documents(match_query)
    mismatch_count = await db.validation_results.count_documents(mismatch_query)
    
    match_percentage = (match_count / dino_total_validations * 100) if dino_total_validations > 0 else 0
    
    # Images by PLU
    pipeline = [
        {"$group": {"_id": "$plu_code", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    images_by_plu_cursor = db.captured_images.aggregate(pipeline)
    images_by_plu = {}
    async for item in images_by_plu_cursor:
        images_by_plu[item["_id"]] = item["count"]
    
    return DashboardStats(
        total_images=total_images,
        total_validations=total_validations,
        match_count=match_count,
        mismatch_count=mismatch_count,
        match_percentage=round(match_percentage, 2),
        images_by_plu=images_by_plu,
        current_mode=SYSTEM_MODE
    )


@api_router.get("/stats/validation-kpi")
async def get_validation_kpi(top_pairs: int = 10):
    per_plu_pipeline = [
        {"$match": {"source": {"$ne": "resnet_topk"}}},
        {"$group": {
            "_id": "$plu_code",
            "total": {"$sum": 1},
            "match_count": {"$sum": {"$cond": ["$is_match", 1, 0]}},
            "mismatch_count": {"$sum": {"$cond": ["$is_match", 0, 1]}},
        }},
        {"$sort": {"total": -1}},
    ]
    per_plu = await db.validation_results.aggregate(per_plu_pipeline).to_list(10000)
    per_plu_stats = []
    for item in per_plu:
        total = item.get("total", 0) or 0
        match_count = item.get("match_count", 0) or 0
        mismatch_count = item.get("mismatch_count", 0) or 0
        match_rate = (match_count / total) * 100.0 if total else 0.0
        per_plu_stats.append({
            "plu_code": item["_id"],
            "total": total,
            "match_count": match_count,
            "mismatch_count": mismatch_count,
            "match_rate": round(match_rate, 2),
        })

    confusion_pipeline = [
        {"$match": {
            "source": {"$ne": "resnet_topk"},
            "is_match": False,
            "analysis_predicted_plu": {"$nin": [None, ""]},
        }},
        {"$group": {
            "_id": {"selected": "$plu_code", "predicted": "$analysis_predicted_plu"},
            "count": {"$sum": 1},
        }},
        {"$sort": {"count": -1}},
        {"$limit": max(1, min(top_pairs, 100))},
    ]
    confusion = await db.validation_results.aggregate(confusion_pipeline).to_list(1000)
    confusion_pairs = [
        {"selected": c["_id"]["selected"], "predicted": c["_id"]["predicted"], "count": c["count"]}
        for c in confusion
    ]

    return {
        "per_plu": per_plu_stats,
        "top_confusions": confusion_pairs,
    }


@api_router.get("/stats/resnet-top5-analysis")
async def get_resnet_top5_analysis(
    plu_code: Optional[str] = None,
    low_conf_threshold_pct: float = 35.0,
    sample_limit: int = 60,
    include_only_low_conf: bool = False,
    include_overview: bool = True,
):
    threshold = max(0.0, min(float(low_conf_threshold_pct), 100.0))
    sample_limit = max(1, min(int(sample_limit), max(1, RESNET_ANALYSIS_MAX_SAMPLE_LIMIT)))
    selected_code = str(plu_code or "").strip()
    base_match = {"ai_provider": "butcher_resnet"}

    metric_projection = {
        "_id": 0,
        "id": 1,
        "timestamp": 1,
        "plu_code": {"$toString": {"$ifNull": ["$plu_code", ""]}},
        "analysis_predicted_plu": 1,
        "original_filename": 1,
        "filename": 1,
        "top_matches": {
            "$cond": [
                {"$isArray": "$top_matches"},
                {"$slice": ["$top_matches", 5]},
                [],
            ]
        },
    }
    metric_add_fields_1 = {
        "top5_codes": {
            "$map": {
                "input": "$top_matches",
                "as": "m",
                "in": {"$toString": {"$ifNull": ["$$m.plu_code", ""]}},
            }
        },
        "top1_match": {"$arrayElemAt": ["$top_matches", 0]},
        "selected_match": {
            "$first": {
                "$filter": {
                    "input": "$top_matches",
                    "as": "m",
                    "cond": {
                        "$eq": [
                            {"$toString": {"$ifNull": ["$$m.plu_code", ""]}},
                            {"$toString": {"$ifNull": ["$plu_code", ""]}},
                        ]
                    },
                }
            }
        },
    }
    metric_add_fields_2 = {
        "top1_code": {"$arrayElemAt": ["$top5_codes", 0]},
        "is_top5_match": {"$in": ["$plu_code", "$top5_codes"]},
        "top1_score_raw": {
            "$convert": {
                "input": "$top1_match.score",
                "to": "double",
                "onError": None,
                "onNull": None,
            }
        },
        "top1_prob_raw": {
            "$convert": {
                "input": "$top1_match.prob",
                "to": "double",
                "onError": None,
                "onNull": None,
            }
        },
        "selected_score_raw": {
            "$convert": {
                "input": "$selected_match.score",
                "to": "double",
                "onError": None,
                "onNull": None,
            }
        },
        "selected_prob_raw": {
            "$convert": {
                "input": "$selected_match.prob",
                "to": "double",
                "onError": None,
                "onNull": None,
            }
        },
    }
    metric_add_fields_3 = {
        "is_top1_match": {"$eq": ["$plu_code", "$top1_code"]},
        "top1_score_pct": {
            "$ifNull": [
                "$top1_score_raw",
                {
                    "$cond": [
                        {"$ne": ["$top1_prob_raw", None]},
                        {"$multiply": ["$top1_prob_raw", 100]},
                        None,
                    ]
                },
            ]
        },
        "selected_score_pct": {
            "$ifNull": [
                "$selected_score_raw",
                {
                    "$cond": [
                        {"$ne": ["$selected_prob_raw", None]},
                        {"$multiply": ["$selected_prob_raw", 100]},
                        None,
                    ]
                },
            ]
        },
    }
    metric_add_fields_4 = {
        "is_low_conf_top5": {
            "$and": [
                "$is_top5_match",
                {"$ne": ["$selected_score_pct", None]},
                {"$lt": ["$selected_score_pct", threshold]},
            ]
        },
    }

    summary = None
    per_plu_stats: List[dict] = []

    if include_overview:
        per_plu_pipeline = [
            {"$match": base_match},
            {"$project": metric_projection},
            {"$match": {"plu_code": {"$ne": ""}}},
            {"$addFields": metric_add_fields_1},
            {"$addFields": metric_add_fields_2},
            {"$addFields": metric_add_fields_3},
            {"$addFields": metric_add_fields_4},
            {"$group": {
                "_id": "$plu_code",
                "total": {"$sum": 1},
                "top5_match_count": {"$sum": {"$cond": ["$is_top5_match", 1, 0]}},
                "top1_match_count": {"$sum": {"$cond": ["$is_top1_match", 1, 0]}},
                "low_conf_top5_count": {"$sum": {"$cond": ["$is_low_conf_top5", 1, 0]}},
                "avg_top1_score_pct": {"$avg": "$top1_score_pct"},
                "avg_selected_score_pct": {"$avg": "$selected_score_pct"},
                "top1_score_sum": {"$sum": {"$ifNull": ["$top1_score_pct", 0]}},
                "top1_score_count": {"$sum": {"$cond": [{"$ne": ["$top1_score_pct", None]}, 1, 0]}},
                "last_seen": {"$max": "$timestamp"},
            }},
            {"$sort": {"total": -1}},
        ]
        per_plu_docs = await db.validation_results.aggregate(per_plu_pipeline, allowDiskUse=True).to_list(10000)

        total = 0
        top5_match_total = 0
        top1_match_total = 0
        low_conf_top5_total = 0
        top1_score_sum = 0.0
        top1_score_count = 0

        for item in per_plu_docs:
            plu = str(item.get("_id") or "").strip()
            plu_total = int(item.get("total") or 0)
            top5_match_count = int(item.get("top5_match_count") or 0)
            top1_match_count = int(item.get("top1_match_count") or 0)
            low_conf_count = int(item.get("low_conf_top5_count") or 0)
            top5_mismatch_count = max(0, plu_total - top5_match_count)

            avg_top1_raw = item.get("avg_top1_score_pct")
            avg_selected_raw = item.get("avg_selected_score_pct")
            avg_top1 = round(float(avg_top1_raw), 2) if avg_top1_raw is not None else None
            avg_selected = round(float(avg_selected_raw), 2) if avg_selected_raw is not None else None

            per_plu_stats.append({
                "plu_code": plu,
                "total": plu_total,
                "top5_match_count": top5_match_count,
                "top5_mismatch_count": top5_mismatch_count,
                "top5_match_rate": round((top5_match_count / plu_total) * 100.0, 2) if plu_total else 0.0,
                "top1_match_count": top1_match_count,
                "top1_match_rate": round((top1_match_count / plu_total) * 100.0, 2) if plu_total else 0.0,
                "low_conf_top5_count": low_conf_count,
                "low_conf_top5_rate": round((low_conf_count / plu_total) * 100.0, 2) if plu_total else 0.0,
                "avg_top1_score_pct": avg_top1,
                "avg_selected_score_pct": avg_selected,
                "last_seen": item.get("last_seen"),
            })

            total += plu_total
            top5_match_total += top5_match_count
            top1_match_total += top1_match_count
            low_conf_top5_total += low_conf_count
            top1_score_sum += float(item.get("top1_score_sum") or 0.0)
            top1_score_count += int(item.get("top1_score_count") or 0)

        top5_mismatch_total = max(0, total - top5_match_total)
        summary = {
            "total": total,
            "top5_match_count": top5_match_total,
            "top5_mismatch_count": top5_mismatch_total,
            "top5_match_rate": round((top5_match_total / total) * 100.0, 2) if total else 0.0,
            "top1_match_count": top1_match_total,
            "top1_match_rate": round((top1_match_total / total) * 100.0, 2) if total else 0.0,
            "low_conf_top5_count": low_conf_top5_total,
            "low_conf_top5_rate": round((low_conf_top5_total / total) * 100.0, 2) if total else 0.0,
            "avg_top1_score_pct": round(top1_score_sum / top1_score_count, 2) if top1_score_count else None,
            "distinct_plu_count": len(per_plu_stats),
        }

    selected_stats = None
    samples: List[dict] = []

    if selected_code:
        selected_match = {
            "ai_provider": "butcher_resnet",
            "plu_code": selected_code,
        }
        selected_stats_pipeline = [
            {"$match": selected_match},
            {"$project": metric_projection},
            {"$match": {"plu_code": {"$ne": ""}}},
            {"$addFields": metric_add_fields_1},
            {"$addFields": metric_add_fields_2},
            {"$addFields": metric_add_fields_3},
            {"$addFields": metric_add_fields_4},
            {"$group": {
                "_id": "$plu_code",
                "total": {"$sum": 1},
                "top5_match_count": {"$sum": {"$cond": ["$is_top5_match", 1, 0]}},
                "top1_match_count": {"$sum": {"$cond": ["$is_top1_match", 1, 0]}},
                "low_conf_top5_count": {"$sum": {"$cond": ["$is_low_conf_top5", 1, 0]}},
                "avg_selected_score_pct": {"$avg": "$selected_score_pct"},
            }},
        ]
        selected_docs = await db.validation_results.aggregate(selected_stats_pipeline).to_list(1)
        if selected_docs:
            s = selected_docs[0]
            s_total = int(s.get("total") or 0)
            s_top5_match = int(s.get("top5_match_count") or 0)
            s_top1_match = int(s.get("top1_match_count") or 0)
            s_low_conf_top5 = int(s.get("low_conf_top5_count") or 0)
            avg_selected_raw = s.get("avg_selected_score_pct")
            selected_stats = {
                "plu_code": selected_code,
                "total": s_total,
                "top5_match_count": s_top5_match,
                "top5_mismatch_count": max(0, s_total - s_top5_match),
                "top5_match_rate": round((s_top5_match / s_total) * 100.0, 2) if s_total else 0.0,
                "top1_match_count": s_top1_match,
                "top1_match_rate": round((s_top1_match / s_total) * 100.0, 2) if s_total else 0.0,
                "low_conf_top5_count": s_low_conf_top5,
                "low_conf_top5_rate": round((s_low_conf_top5 / s_total) * 100.0, 2) if s_total else 0.0,
                "avg_selected_score_pct": round(float(avg_selected_raw), 2) if avg_selected_raw is not None else None,
                "sample_count": 0,
            }

        sample_output_projection = {
            "_id": 0,
            "validation_id": "$id",
            "timestamp": 1,
            "plu_code": 1,
            "filename": {"$ifNull": ["$original_filename", "$filename"]},
            "predicted_plu": {"$ifNull": ["$analysis_predicted_plu", "$top1_code"]},
            "top5_codes": 1,
            "top1_score_pct": {"$round": ["$top1_score_pct", 2]},
            "selected_score_pct": {"$round": ["$selected_score_pct", 2]},
            "is_top5_match": 1,
            "is_top1_match": 1,
            "is_low_conf_top5": 1,
        }
        if include_only_low_conf:
            sample_pipeline = [
                {"$match": selected_match},
                {"$match": {"top_matches.plu_code": selected_code}},
                {"$project": metric_projection},
                {"$match": {"plu_code": {"$ne": ""}}},
                {"$addFields": metric_add_fields_1},
                {"$addFields": metric_add_fields_2},
                {"$addFields": metric_add_fields_3},
                {"$addFields": metric_add_fields_4},
                {"$match": {"is_low_conf_top5": True}},
                {"$sort": {"timestamp": -1}},
                {"$limit": sample_limit},
                {"$project": sample_output_projection},
            ]
        else:
            # Fast path: sort/limit first so Mongo can leverage
            # (ai_provider, plu_code, timestamp) index.
            sample_pipeline = [
                {"$match": selected_match},
                {"$sort": {"timestamp": -1}},
                {"$limit": sample_limit},
                {"$project": metric_projection},
                {"$match": {"plu_code": {"$ne": ""}}},
                {"$addFields": metric_add_fields_1},
                {"$addFields": metric_add_fields_2},
                {"$addFields": metric_add_fields_3},
                {"$addFields": metric_add_fields_4},
                {"$project": sample_output_projection},
            ]
        samples = await db.validation_results.aggregate(sample_pipeline).to_list(sample_limit)
        if selected_stats is not None:
            selected_stats["sample_count"] = len(samples)

    return {
        "query": {
            "plu_code": selected_code or None,
            "low_conf_threshold_pct": threshold,
            "sample_limit": sample_limit,
            "include_only_low_conf": bool(include_only_low_conf),
            "include_overview": bool(include_overview),
        },
        "summary": summary,
        "per_plu": per_plu_stats,
        "selected_plu": selected_stats,
        "samples": samples,
    }


@api_router.get("/health", response_model=HealthStatus)
async def health_check():
    # Mongo
    mongo_ok = True
    try:
        await db.command('ping')
    except Exception as e:
        logging.error(f"Mongo health error: {e}")
        mongo_ok = False
    # Camera quick check (without capturing full image)
    cam_ok = False
    try:
        cap = cv2.VideoCapture(0)
        cam_ok = cap.isOpened()
        cap.release()
    except Exception:
        cam_ok = False
    # System resources
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage(str(ROOT_DIR)).percent
    return HealthStatus(
        mongo_connected=mongo_ok,
        camera_available=cam_ok,
        ai_provider=AI_PROVIDER,
        ai_model=AI_MODEL,
        system_mode=SYSTEM_MODE,
        cpu_percent=cpu,
        mem_percent=mem,
        disk_percent=disk
    )

# Include the router in the main app
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=parse_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

@app.on_event("startup")
async def load_mode_on_startup():
    global SYSTEM_MODE
    global AI_PROVIDER, AI_MODEL
    await ensure_database_indexes()
    SYSTEM_MODE = await load_system_mode_from_db()
    provider, model = await load_ai_config_from_db()
    AI_PROVIDER, AI_MODEL = provider, model
    logging.info(f"System mode loaded from DB: {SYSTEM_MODE}")
    logging.info(f"AI config loaded from DB: provider={AI_PROVIDER}, model={AI_MODEL}")







