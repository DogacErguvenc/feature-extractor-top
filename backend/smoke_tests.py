"""
Basit backend smoke testleri.
- Sağlık kontrolü
- Sistem modu get/set
- AI config get/set (local)
- Opsiyonel: local model ONNX forward/perf (varsa)
Kullanım:
  .\\.venv\\Scripts\\python.exe smoke_tests.py
"""

import os
import time
import json
import base64
from pathlib import Path

import requests

BASE_URL = os.environ.get("BACKEND_URL", "http://localhost:8001")
API = f"{BASE_URL}/api"
HEADERS = {}\r\n

def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_health():    resp = requests.get(f"{API}/health", timeout=5, headers=HEADERS)
    ok(resp.status_code == 200, f"health status {resp.status_code}")
    data = resp.json()
    print("health:", data)


def test_system_mode_toggle():
    # get
    resp = requests.get(f"{API}/system/mode", timeout=5, headers=HEADERS)
    ok(resp.status_code == 200, "mode get failed")
    current = resp.json()["mode"]
    target = "production" if current == "training" else "training"
    # set
    resp = requests.post(f"{API}/system/mode", json={"mode": target}, timeout=5, headers=HEADERS)
    ok(resp.status_code == 200, "mode set failed")
    # reset
    requests.post(f"{API}/system/mode", json={"mode": current}, timeout=5, headers=HEADERS)


def test_ai_config():
    resp = requests.get(f"{API}/system/ai-config", timeout=5, headers=HEADERS)
    ok(resp.status_code == 200, "ai-config get failed")
    cfg = resp.json()
    # set local with empty model
    resp = requests.post(
        f"{API}/system/ai-config",
        json={"provider": "local", "model": "", "version": ""},
        timeout=5,
        headers=HEADERS,
    )
    ok(resp.status_code == 200, "ai-config set failed")
    # restore previous
    requests.post(
        f"{API}/system/ai-config",
        json={"provider": cfg.get("provider", "local"), "model": cfg.get("model", ""), "version": cfg.get("version", "")},
        timeout=5,
        headers=HEADERS,
    )


def test_dashboard_stats():
    resp = requests.get(f"{API}/stats/dashboard", timeout=5, headers=HEADERS)
    ok(resp.status_code == 200, f"stats status {resp.status_code}")


def test_local_onnx_forward():
    base_dir = Path(__file__).parent
    model_path = base_dir / "models" / "local_model.onnx"
    if not model_path.exists():
        print(f"local_model.onnx yok ({model_path}), ONNX testi atlandı.")
        return
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.random.rand(1, 3, 224, 224).astype("float32")
    t0 = time.time()
    outputs = sess.run(None, {input_name: dummy})
    dt = (time.time() - t0) * 1000
    print(f"ONNX forward ok ({model_path}), output shape {outputs[0].shape}, {dt:.1f} ms")


def main():
    test_health()
    test_system_mode_toggle()
    test_ai_config()
    test_dashboard_stats()
    test_local_onnx_forward()
    print("Smoke tests OK")


if __name__ == "__main__":
    main()

