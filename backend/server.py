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
# Supported: 'gemini', 'openai', 'local', 'local_large', 'local_embedding', 'local_gemini', 'local_gemini_consensus'
AI_PROVIDER = os.environ.get('AI_PROVIDER', 'gemini')
AI_MODEL = os.environ.get('AI_MODEL', 'gemini-2.0-flash')  # Model name
LOCAL_MODEL_PATH = Path(os.environ.get('LOCAL_MODEL_PATH', ROOT_DIR / "models" / "local_model.onnx"))
LOCAL_LABELS_PATH = Path(os.environ.get('LOCAL_LABELS_PATH', ROOT_DIR / "models" / "local_labels.json"))
LOCAL_LARGE_MODEL_PATH = Path(os.environ.get('LOCAL_LARGE_MODEL_PATH', ROOT_DIR / "models" / "local_model_large.onnx"))
LOCAL_LARGE_LABELS_PATH = Path(os.environ.get('LOCAL_LARGE_LABELS_PATH', ROOT_DIR / "models" / "local_labels_large.json"))
LOCAL_SMALL_IMAGE_SIZE = int(os.environ.get('LOCAL_SMALL_IMAGE_SIZE', '224'))
LOCAL_LARGE_IMAGE_SIZE = int(os.environ.get('LOCAL_LARGE_IMAGE_SIZE', '300'))
REFERENCE_IMAGE_DIR = Path(os.environ.get('REFERENCE_IMAGE_DIR', ROOT_DIR / "reference_images")).resolve()
REFERENCE_MAX_IMAGES = int(os.environ.get('REFERENCE_MAX_IMAGES', '2'))
PROMPT_VERSION = os.environ.get('PROMPT_VERSION', 'dense_v1')
ALLOWED_IMAGE_DIR = Path(os.environ.get('ALLOWED_IMAGE_DIR', ROOT_DIR / "incoming")).resolve()
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
    ai_analysis: str
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

class PLUSelection(BaseModel):
    plu_code: str

class SystemModeUpdate(BaseModel):
    mode: str  # "training" or "production"

class ValidateSyncRequest(BaseModel):
    plu_code: str
    image_base64: Optional[str] = None
    filename: Optional[str] = None
    file_path: Optional[str] = None

class BatchValidationMeta(BaseModel):
    filename: str
    plu_code: str

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
def capture_image_from_camera() -> Optional[str]:
    """Capture image from default camera and return as base64 string"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Cannot open camera")
            return None
        
        # Wait a bit for camera to initialize
        for _ in range(5):
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


def decode_base64_to_pil(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img


async def run_embedding_inference(image_base64: str, plu_product: PLUProduct) -> dict:
    """Run local embedding similarity and return uniform result dict."""
    engine = get_embedding_engine()
    store = get_embedding_store()
    img = decode_base64_to_pil(image_base64)
    query_embeddings = await asyncio.to_thread(engine.extract_embeddings, img, True)
    result = await asyncio.to_thread(
        store.score_query,
        query_embeddings,
        plu_product.plu_code,
        EMBEDDING_TOP_K,
        EMBEDDING_MIN_SIM,
        EMBEDDING_MARGIN,
    )
    return {
        "analysis": result["analysis"],
        "is_match": result["is_match"],
        "confidence": result["confidence"],
    }
async def run_local_inference(image_base64: str, plu_product: PLUProduct, model_key: str = "local") -> dict:
    """Run offline ONNX model and return uniform result dict."""
    session, input_name, labels, image_size = await asyncio.to_thread(load_local_model, model_key)
    input_tensor = preprocess_for_local_model(image_base64, size=image_size)
    outputs = await asyncio.to_thread(session.run, None, {input_name: input_tensor})
    logits = outputs[0][0]  # assuming (1, num_classes)
    probs = softmax(logits)
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
        "confidence": round(selected_conf, 2)
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
async def analyze_image_with_ai(image_base64: str, plu_product: PLUProduct) -> dict:
    """Analyze image using configured AI provider (local, gemini, openai, or local+gemini fallback)."""
    try:
        provider = AI_PROVIDER
        model_name = AI_MODEL

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
            start = time.monotonic()
            result = await analyze_image_with_ai(image_base64, plu_obj)
            elapsed_ms = (time.monotonic() - start) * 1000.0
            validation = ValidationResult(
                plu_code=selection.plu_code,
                selected_plu_name=plu_obj.name,
                image_base64=image_base64,
                ai_analysis=result["analysis"],
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

# Synchronous validation: capture image, run AI, optionally save result
@api_router.post("/validate-sync")
async def validate_sync(payload: ValidateSyncRequest):
    global SYSTEM_MODE
    # Find PLU
    plu_product = await db.plu_products.find_one({"plu_code": payload.plu_code}, {"_id": 0})
    if not plu_product:
        raise HTTPException(status_code=404, detail="PLU not found")
    if isinstance(plu_product['created_at'], str):
        plu_product['created_at'] = datetime.fromisoformat(plu_product['created_at'])
    plu_obj = PLUProduct(**plu_product)

    # Determine image source: file_path -> image_base64 -> camera
    image_base64 = None
    filename = payload.filename

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
        image_base64 = capture_image_from_camera()
        if not image_base64:
            raise HTTPException(status_code=500, detail="Camera not available or failed to capture image")

    # Run AI
    start = time.monotonic()
    result = await analyze_image_with_ai(image_base64, plu_obj)
    elapsed_ms = (time.monotonic() - start) * 1000.0

    # Persist validation result for traceability
    validation = ValidationResult(
        plu_code=payload.plu_code,
        selected_plu_name=plu_obj.name,
        image_base64=image_base64,
        ai_analysis=result["analysis"],
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

    # Return concise response
    return {
        "is_match": result["is_match"],
        "confidence": result["confidence"],
        "analysis": result["analysis"],
        "ai_provider": AI_PROVIDER,
        "ai_model": AI_MODEL,
        "processing_ms": round(elapsed_ms, 2),
        "timestamp": doc["timestamp"],
        "validation_id": validation.id,
        "filename": filename
    }

@api_router.post("/batch/validate")
async def batch_validate(metadata: str = Form(...), files: List[UploadFile] = File(...)):
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

        start_t = time.monotonic()
        ai_result = await analyze_image_with_ai(image_base64, PLUProduct(**plu_product))
        elapsed_ms = (time.monotonic() - start_t) * 1000.0

        validation = ValidationResult(
            plu_code=expected_plu,
            selected_plu_name=plu_product["name"],
            image_base64=image_base64,
            ai_analysis=ai_result.get("analysis", ""),
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

        result_payload = {
            "filename": filename,
            "plu_code": expected_plu,
            "is_match": ai_result.get("is_match", False),
            "confidence": ai_result.get("confidence", 0.0),
            "analysis": ai_result.get("analysis", ""),
            "processing_ms": round(elapsed_ms, 2),
            "ai_provider": AI_PROVIDER,
            "ai_model": AI_MODEL,
            "validation_id": validation.id,
            "batch_id": batch_id,
            "status": "match" if ai_result.get("is_match") else "mismatch"
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
        "error_count": error_count
    }

    return {"batch_id": batch_id, "summary": summary, "results": results}

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
        "local_gemini",
        "local_gemini_consensus",
    }
    if config.provider not in allowed:
        raise HTTPException(status_code=400, detail=f"Provider must be one of {allowed}")
    AI_PROVIDER = config.provider
    # Normalize model choice based on provider
    if AI_PROVIDER in {"local", "local_large", "local_embedding"}:
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
    match_count = await db.validation_results.count_documents({"is_match": True})
    mismatch_count = await db.validation_results.count_documents({"is_match": False})
    
    match_percentage = (match_count / total_validations * 100) if total_validations > 0 else 0
    
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
    SYSTEM_MODE = await load_system_mode_from_db()
    provider, model = await load_ai_config_from_db()
    AI_PROVIDER, AI_MODEL = provider, model
    logging.info(f"System mode loaded from DB: {SYSTEM_MODE}")
    logging.info(f"AI config loaded from DB: provider={AI_PROVIDER}, model={AI_MODEL}")




