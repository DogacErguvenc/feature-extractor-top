import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import uvicorn


def _runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _set_default_path_env(name: str, value: Path) -> None:
    if not os.environ.get(name):
        os.environ[name] = str(value)


def configure_runtime_environment() -> Path:
    runtime_root = _runtime_root()

    # Load customer-provided runtime config first.
    load_dotenv(runtime_root / ".env", override=False)

    # Default runtime paths for packaged distribution.
    _set_default_path_env("ALLOWED_IMAGE_DIR", runtime_root / "incoming")
    _set_default_path_env("BUTCHER_CONFIG_PATH", runtime_root / "butcher_config.yaml")
    _set_default_path_env("TRAY_ROI_PATH", runtime_root / "tray_roi.json")
    _set_default_path_env("PLU_BUDGETS_PATH", runtime_root / "plu_budgets.json")
    _set_default_path_env("LOCAL_MODEL_PATH", runtime_root / "models" / "local_model.onnx")
    _set_default_path_env("LOCAL_LABELS_PATH", runtime_root / "models" / "local_labels.json")
    _set_default_path_env("LOCAL_LARGE_MODEL_PATH", runtime_root / "models" / "local_model_large.onnx")
    _set_default_path_env("LOCAL_LARGE_LABELS_PATH", runtime_root / "models" / "local_labels_large.json")
    _set_default_path_env("REFERENCE_IMAGE_DIR", runtime_root / "reference_images")
    _set_default_path_env("EMBEDDING_STORE_DIR", runtime_root / "embedding_store")

    return runtime_root


def main() -> None:
    configure_runtime_environment()

    # Import after env setup so server.py reads correct runtime paths.
    from server import app

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
