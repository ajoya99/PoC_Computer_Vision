"""
Local inference server for the Single Image Detection UI with adaptive scaling.
Font sizes and box thickness scale based on input image dimensions.

Usage (from project root):
    python ui_single_image_review/server_v2.py

Serves the UI at http://127.0.0.1:8765/ and handles POST /detect.
"""

import base64
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
UI_DIR = Path(__file__).resolve().parent

# Available models: key -> ordered list of candidate paths
MODEL_CANDIDATES = {
    "rtdetr": [
        ROOT / "runs" / "detect" / "runs" / "detect" / "models" / "rtdetr_test_full_e18_f100_b4" / "weights" / "best.pt",
        ROOT / "runs" / "detect" / "models" / "rtdetr_test_full_e18_f100_b4" / "weights" / "best.pt",
        ROOT / "runs" / "detect" / "runs" / "detect" / "models" / "rtdetr_test_production_e16_f50_b4" / "weights" / "best.pt",
        ROOT / "runs" / "detect" / "models" / "rtdetr_test_production_e16_f50_b4" / "weights" / "best.pt",
    ],
    "yolo": [
        ROOT / "runs" / "detect" / "models" / "yolo26s_aggressive_s_e30_f75_i832_b2" / "weights" / "best.pt",
        ROOT / "runs" / "detect" / "models" / "yolo26s_high_s_e18_f50" / "weights" / "best.pt",
        ROOT / "runs" / "detect" / "models" / "yolo26n_high_n_e18_f50" / "weights" / "best.pt",
    ],
}

MODEL_PATHS = {
    key: next((p for p in paths if p.exists()), None)
    for key, paths in MODEL_CANDIDATES.items()
}
MODEL_LABELS = {
    "rtdetr": "RT-DETR-L",
    "yolo": "YOLO26s",
}

SERVER_VERSION = "ui-single-review-adaptive-scaling-v4"
_models = {}  # cache: key -> loaded model


def normalize_class_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def resolve_threshold(class_key: str, class_conf_norm: dict, default_conf: float) -> float:
    if not class_conf_norm:
        return default_conf

    if class_key in class_conf_norm:
        return float(class_conf_norm[class_key])

    aliases = {
        "pallets": "pallet",
        "wheelchair": "wheelchair",
        "wheelchairs": "wheelchair",
        "wheelchairperson": "wheelchair",
        "wheelchairuser": "wheelchair",
    }

    alias_key = aliases.get(class_key)
    if alias_key and alias_key in class_conf_norm:
        return float(class_conf_norm[alias_key])

    if class_key.endswith("s"):
        singular = class_key[:-1]
        if singular in class_conf_norm:
            return float(class_conf_norm[singular])

    return default_conf


def get_class_color(class_name: str) -> tuple:
    """Return BGR color tuple for each class."""
    color_map = {
        "box": (200, 100, 50),        # blue-ish
        "pallet": (0, 165, 255),      # orange
        "person": (0, 255, 0),        # green
        "forklift": (0, 0, 255),      # red
        "cart": (255, 0, 255),        # magenta
        "wheelchair": (255, 255, 0),  # cyan
    }
    normalized = normalize_class_name(class_name)
    return color_map.get(normalized, (88, 180, 255))  # default light blue


def get_model(model_key: str = "rtdetr"):
    global _models
    if model_key not in _models:
        path = MODEL_PATHS.get(model_key)
        if path is None:
            raise RuntimeError(f"No weights found for model '{model_key}'.")
        from ultralytics import YOLO
        _models[model_key] = YOLO(str(path))
        print(f"[server] Model loaded ({model_key}): {path.parent.parent.name}", flush=True)
    return _models[model_key]


# ---------------------------------------------------------------------------
# Inference with adaptive scaling
# ---------------------------------------------------------------------------

def run_inference(image_bytes: bytes, confidence: float = 0.05,
                  class_conf: dict | None = None,
                  model_key: str = "rtdetr") -> dict:
    """
    Run detection with per-class threshold filtering.
    Annotations scale automatically based on image size.
    """
    from PIL import Image as PILImage
    import cv2
    import numpy as np

    print(f"[inference] Starting (model={model_key})...", flush=True)
    model = get_model(model_key)
    print("[inference] Model loaded, converting image...", flush=True)
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_width, img_height = img.size
    print(f"[inference] Image size: {img_width}x{img_height}", flush=True)

    print("[inference] Running model.predict()...", flush=True)
    results = model.predict(
        source=img,
        conf=confidence,
        device=0,
        verbose=False,
        save=False,
        stream=False,
    )
    print("[inference] Model prediction complete", flush=True)

    result = results[0]
    model_names = result.names if hasattr(result, "names") and result.names is not None else {}
    boxes = result.boxes

    class_conf_norm = {}
    if class_conf:
        class_conf_norm = {
            normalize_class_name(k): float(v)
            for k, v in class_conf.items()
        }

    kept = []

    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf_val = float(boxes.conf[i].item())
            cls_name = str(model_names.get(cls_id, cls_id))
            cls_key = normalize_class_name(cls_name)
            threshold = resolve_threshold(cls_key, class_conf_norm, confidence)

            if conf_val >= threshold:
                kept.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf_val, 3),
                    "threshold": round(float(threshold), 3),
                    "box": [round(v, 1) for v in boxes.xyxy[i].tolist()],
                })

    # Calculate adaptive scaling based on image dimensions
    # Smaller images -> larger relative text, larger images -> smaller relative text
    avg_dim = (img_width + img_height) / 2.0
    # Reference: 1000px gets scale ~1.0
    base_scale = 1000.0 / avg_dim
    scale = max(0.4, min(4.0, base_scale))
    
    print(f"[inference] Adaptive scale: {scale:.2f} (avg_dim={avg_dim:.0f})", flush=True)
    
    # Dynamically sized parameters
    font_size = max(0.7, 3.5 * scale)
    font_thickness = max(1, int(5 * scale))
    box_thickness = max(2, int(8 * scale))
    text_bg_pad = max(4, int(12 * scale))
    text_vertical_offset = max(2, int(8 * scale))

    # Build annotated image
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    for det in kept:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        color = get_class_color(det['class_name'])
        
        # Draw detection box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, box_thickness)
        
        # Draw label background and text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)
        label_y_top = y1 - text_h - text_bg_pad
        label_x_right = x1 + text_w + text_bg_pad
        
        cv2.rectangle(img_bgr, (x1, label_y_top), (label_x_right, y1), color, -1)
        cv2.putText(img_bgr, label, (x1 + int(text_bg_pad/2), y1 - int(text_vertical_offset/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    annotated_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "detections": len(kept),
        "items": kept,
        "thresholds": class_conf_norm,
        "annotated_image": "data:image/jpeg;base64," + annotated_b64,
        "model": WEIGHTS_PATH.parent.parent.name if WEIGHTS_PATH else "unknown",
        "server_version": SERVER_VERSION,
        "scaling_info": {
            "image_width": img_width,
            "image_height": img_height,
            "scale_factor": round(scale, 2),
        }
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"[server] {self.address_string()} - {format % args}", flush=True)

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(body)

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/ui_single_image_review/"):
            self._serve_file(UI_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path.endswith(".css"):
            self._serve_file(UI_DIR / Path(self.path).name, "text/css; charset=utf-8")
        elif self.path.endswith(".js"):
            self._serve_file(UI_DIR / Path(self.path).name, "application/javascript; charset=utf-8")
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        print(f"[server] POST request received at {self.path}", flush=True)
        if self.path != "/detect":
            self.send_json({"error": "Not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        print(f"[server] POST body size: {len(body)} bytes", flush=True)

        try:
            payload = json.loads(body.decode("utf-8"))
            image_b64 = payload.get("image_b64", "")
            confidence = float(payload.get("confidence", 0.05))
            class_conf = payload.get("class_conf", None)

            print(f"[server] Confidence: {confidence}, Class conf keys: {list(class_conf.keys()) if class_conf else 'None'}", flush=True)

            if not isinstance(class_conf, dict) or len(class_conf) == 0:
                self.send_json(
                    {
                        "error": "Missing class_conf payload from UI. Hard refresh page (Ctrl+F5) and retry.",
                        "hint": "UI must send class_conf map with per-class slider thresholds.",
                    },
                    status=400,
                )
                return

            model_key = payload.get("model_key", "rtdetr")

            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]

            image_bytes = base64.b64decode(image_b64)
            print(f"[server] Image decoded: {len(image_bytes)} bytes", flush=True)
            result = run_inference(image_bytes, confidence, class_conf, model_key)
            print(f"[server] Inference complete: {result.get('detections')} detections", flush=True)
            self.send_json(result)
        except Exception as exc:
            print(f"[server] ERROR: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            self.send_json({"error": str(exc)}, status=500)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = 8765
    host = "127.0.0.1"
    try:
        print(f"[server] Single Image Detection server starting...")
        print(f"[server] Source  : {Path(__file__).resolve()}")
        print(f"[server] Version : {SERVER_VERSION}")
        weights_label = " | ".join(
                f"{k}: {p.parent.parent.name}" if p else f"{k}: NOT FOUND"
                for k, p in MODEL_PATHS.items()
            )
        print(f"[server] Weights : {weights_label}")
        print(f"[server] Creating HTTPServer on {host}:{port}...", flush=True)
        server = HTTPServer((host, port), Handler)
        print(f"[server] Open    : http://{host}:{port}/", flush=True)
        print(f"[server] Press Ctrl+C to stop.", flush=True)
        print(f"[server] Starting serve_forever()...", flush=True)
        server.serve_forever()
    except Exception as e:
        print(f"[server] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
