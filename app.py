"""
Laparoscopy Image Defogging - Flask Web Interface
Run:  python app.py
Open: http://127.0.0.1:5000
"""

import os
import sys
import base64
import traceback
import time
import zipfile
import io

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "web_ui"))
CORS(app)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(BASE_DIR, "scripts", "checkpoints",
                            "pix2pix_laparoscopy_dc", "best_net_G.pth")
MAX_BYTES    = 20 * 1024 * 1024   # 20 MB upload limit
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

# ── Pipeline parameter definitions ───────────────────────────────────────────
PIPELINE_PARAMS = {
    "dc_kernel_size":      {"type": "int",   "min": 3,      "max": 51,   "default": 15,     "step": 2,     "label": "Kernel Size",         "group": "dark_channel", "desc": "Erosion kernel for dark channel computation (odd values only)"},
    "dc_filter_radius":    {"type": "int",   "min": 1,      "max": 50,   "default": 15,     "step": 1,     "label": "Guided Filter Radius", "group": "dark_channel", "desc": "Smoothing radius of the guided filter"},
    "dc_filter_eps":       {"type": "float", "min": 0.00001,"max": 0.1,  "default": 0.0001, "step": 0.00001,"label": "Filter Epsilon",      "group": "dark_channel", "desc": "Regularization term — lower = sharper edges, higher = smoother"},
    "bypass_dark_channel": {"type": "bool",  "default": False,                               "label": "Bypass Dark Channel",  "group": "dark_channel", "desc": "Skip dark channel prior entirely (feed zero channel to model)"},
    "defog_strength":      {"type": "float", "min": 0.0,    "max": 1.0,  "default": 1.0,    "step": 0.05,  "label": "Defog Strength",      "group": "post_processing", "desc": "Blend ratio: strength × defogged + (1-strength) × original"},
    "sharpen_amount":      {"type": "float", "min": 0.0,    "max": 3.0,  "default": 0.0,    "step": 0.1,   "label": "Sharpen Amount",      "group": "post_processing", "desc": "Unsharp mask strength applied to the output"},
    "brightness":          {"type": "int",   "min": -100,   "max": 100,  "default": 0,      "step": 1,     "label": "Brightness",          "group": "post_processing", "desc": "Brightness offset applied to the output"},
    "contrast":            {"type": "float", "min": 0.5,    "max": 2.0,  "default": 1.0,    "step": 0.05,  "label": "Contrast",            "group": "post_processing", "desc": "Contrast multiplier applied to the output"},
    "clahe_enabled":       {"type": "bool",  "default": False,                               "label": "Enable CLAHE",         "group": "clahe", "desc": "Apply Contrast Limited Adaptive Histogram Equalization"},
    "clahe_clip":          {"type": "float", "min": 1.0,    "max": 10.0, "default": 2.0,    "step": 0.5,   "label": "CLAHE Clip Limit",    "group": "clahe", "desc": "Threshold for contrast limiting"},
    "clahe_grid":          {"type": "int",   "min": 2,      "max": 16,   "default": 8,      "step": 1,     "label": "CLAHE Grid Size",     "group": "clahe", "desc": "Size of the grid for histogram equalization"},
    "interpolation":       {"type": "str",   "options": ["lanczos", "cubic", "linear", "nearest"], "default": "lanczos", "label": "Interpolation", "group": "output", "desc": "Upscale interpolation method for restoring original resolution"},
    "output_fmt":          {"type": "str",   "options": ["png", "jpg"], "default": "png",    "label": "Output Format",       "group": "output", "desc": "Encoding format for the output image"},
    "jpeg_quality":        {"type": "int",   "min": 50,     "max": 100,  "default": 92,     "step": 1,     "label": "JPEG Quality",        "group": "output", "desc": "Quality setting when output format is JPEG"},
}

PRESETS = {
    "default":    {"label": "Default",          "desc": "Original pipeline settings — no modifications",                  "params": {}},
    "aggressive": {"label": "Aggressive Defog", "desc": "Maximum defogging with sharpening and CLAHE",                    "params": {"dc_kernel_size": 21, "dc_filter_radius": 20, "dc_filter_eps": 0.00005, "sharpen_amount": 1.2, "clahe_enabled": True, "clahe_clip": 3.0, "contrast": 1.15}},
    "light":      {"label": "Light Touch",      "desc": "Subtle correction — preserves natural appearance",               "params": {"defog_strength": 0.6, "dc_kernel_size": 11, "dc_filter_eps": 0.001, "brightness": 5}},
    "max_clarity":{"label": "Max Clarity",       "desc": "High contrast + sharpening for maximum detail",                  "params": {"sharpen_amount": 2.0, "contrast": 1.4, "clahe_enabled": True, "clahe_clip": 4.0, "clahe_grid": 4, "brightness": 10}},
    "no_dc":      {"label": "No Dark Channel",  "desc": "Bypass the dark channel prior — model-only output",              "params": {"bypass_dark_channel": True}},
}

INTERP_MAP = {
    "lanczos": cv2.INTER_LANCZOS4,
    "cubic":   cv2.INTER_CUBIC,
    "linear":  cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}

# ── Model cache ───────────────────────────────────────────────────────────────
_desmoker   = None
MODEL_ERROR = None

# ── Session history ───────────────────────────────────────────────────────────
session_history = []   # list of dicts: {filename, timestamp, elapsed_s}
_total_time   = 0.0
_total_count  = 0


def get_model():
    global _desmoker, MODEL_ERROR
    if _desmoker is not None:
        return _desmoker, None
    if not os.path.isfile(MODEL_PATH):
        MODEL_ERROR = (
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Please place best_net_G.pth in the checkpoints folder."
        )
        return None, MODEL_ERROR
    try:
        from main import desmoker
        _desmoker   = desmoker(MODEL_PATH)
        MODEL_ERROR = None
        return _desmoker, None
    except Exception as exc:
        MODEL_ERROR = f"Failed to load model: {exc}\n{traceback.format_exc()}"
        return None, MODEL_ERROR


def _encode(img_bgr: np.ndarray, fmt: str = "jpg", quality: int = 92) -> str:
    """Encode a BGR numpy image to a base64 data-URI (JPEG or PNG)."""
    if fmt == "png":
        ext, mime = ".png", "image/png"
        params = []
    else:
        ext, mime = ".jpg", "image/jpeg"
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buf = cv2.imencode(ext, img_bgr, params)
    return f"data:{mime};base64," + base64.b64encode(buf).decode("utf-8")


def _encode_gray(img_gray: np.ndarray) -> str:
    """Encode a grayscale image to base64 PNG data-URI."""
    _, buf = cv2.imencode(".png", img_gray)
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


def _parse_param(name: str, form_data) -> any:
    """Parse a single pipeline parameter from form data, with validation."""
    spec = PIPELINE_PARAMS[name]
    raw = form_data.get(name)
    if raw is None:
        return spec["default"]

    ptype = spec["type"]
    try:
        if ptype == "bool":
            return raw.lower() in ("true", "1", "yes", "on")
        elif ptype == "int":
            val = int(raw)
            return max(spec["min"], min(spec["max"], val))
        elif ptype == "float":
            val = float(raw)
            return max(spec["min"], min(spec["max"], val))
        elif ptype == "str":
            return raw if raw in spec.get("options", [raw]) else spec["default"]
    except (ValueError, TypeError):
        return spec["default"]
    
    return spec["default"]


def _apply_post_processing(original_bgr, result_bgr, params):
    """Apply post-processing effects based on user parameters."""
    out = result_bgr.copy()

    # ── Defog strength blend ──
    strength = params["defog_strength"]
    if strength < 1.0:
        # Resize original to match result if needed
        if original_bgr.shape[:2] != out.shape[:2]:
            orig_resized = cv2.resize(original_bgr, (out.shape[1], out.shape[0]),
                                      interpolation=cv2.INTER_LANCZOS4)
        else:
            orig_resized = original_bgr
        out = cv2.addWeighted(out, strength, orig_resized, 1.0 - strength, 0)

    # ── Brightness & Contrast ──
    brightness = params["brightness"]
    contrast = params["contrast"]
    if brightness != 0 or contrast != 1.0:
        out = np.clip(out.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

    # ── Sharpening (unsharp mask) ──
    sharpen = params["sharpen_amount"]
    if sharpen > 0:
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=3)
        out = cv2.addWeighted(out, 1.0 + sharpen, blurred, -sharpen, 0)
        out = np.clip(out, 0, 255).astype(np.uint8)

    # ── CLAHE ──
    if params["clahe_enabled"]:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=params["clahe_clip"],
            tileGridSize=(params["clahe_grid"], params["clahe_grid"])
        )
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a, b])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return out


def _run_inference(img_bgr: np.ndarray, params: dict):
    """
    Runs the defogging pipeline with user-specified parameters.
    Returns (result_bgr, dc_vis, elapsed_seconds, h, w) or raises on error.
    """
    model, err = get_model()
    if err:
        raise RuntimeError(err)

    h, w = img_bgr.shape[:2]
    img_256 = cv2.resize(img_bgr, (256, 256))

    # Ensure odd kernel size
    ks = params["dc_kernel_size"]
    if ks % 2 == 0:
        ks += 1

    t0 = time.time()
    result_rgb, dc_vis = model.apply(
        img_256,
        dc_kernel_size=ks,
        dc_filter_radius=params["dc_filter_radius"],
        dc_filter_eps=params["dc_filter_eps"],
        bypass_dark_channel=params["bypass_dark_channel"],
    )
    elapsed = time.time() - t0

    # Convert to BGR
    result_bgr = cv2.cvtColor(result_rgb[:, :, :3], cv2.COLOR_RGB2BGR)

    # Upscale result back to original resolution
    interp = INTERP_MAP.get(params["interpolation"], cv2.INTER_LANCZOS4)
    if (h, w) != (256, 256):
        result_bgr = cv2.resize(result_bgr, (w, h), interpolation=interp)

    # Apply post-processing
    result_bgr = _apply_post_processing(img_bgr, result_bgr, params)

    return result_bgr, dc_vis, elapsed, h, w


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve any static asset (CSS, JS, images) from the web_ui folder."""
    return send_from_directory(app.static_folder, filename)


@app.route("/status")
def status():
    """Model readiness + hardware info."""
    mod, stat_err = get_model()
    if torch.cuda.is_available():
        device_label = f"GPU ({torch.cuda.get_device_name(0)})"
        try:
            mem_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            mem_alloc = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_info = {
                "total_vram_gb": round(mem_total, 2),
                "used_vram_gb": round(mem_alloc, 2),
            }
        except Exception:
            gpu_info = {}
    else:
        device_label = "CPU"
        gpu_info = {}

    avg_time = round(_total_time / _total_count, 2) if _total_count else None

    if stat_err:
        return jsonify({"ready": False, "error": stat_err,
                        "device": device_label, "avg_time": avg_time,
                        "gpu_info": gpu_info})
    return jsonify({"ready": True, "device": device_label, "avg_time": avg_time,
                    "gpu_info": gpu_info})


@app.route("/config")
def config():
    """Return pipeline parameter definitions, defaults, ranges, and presets."""
    return jsonify({
        "params": PIPELINE_PARAMS,
        "presets": PRESETS,
    })


@app.route("/process", methods=["POST"])
def process():
    """
    Single image defogging with full pipeline control.
    Form fields: image (file), plus any pipeline parameter names.
    Returns JSON: {original, processed, dark_channel, message, elapsed,
                   width, height, params_used}
    """
    global _total_time, _total_count

    if "image" not in request.files:
        return jsonify({"error": "No image file in request"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Parse all pipeline parameters from form data
    params = {}
    for name in PIPELINE_PARAMS:
        params[name] = _parse_param(name, request.form)

    raw = file.read()
    if len(raw) > MAX_BYTES:
        return jsonify({"error": "File too large. Max allowed: 20 MB"}), 413

    np_arr  = np.frombuffer(raw, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Cannot decode image — unsupported format?"}), 400

    try:
        result_bgr, dc_vis, elapsed, h, w = _run_inference(img_bgr, params)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    fmt = params["output_fmt"]
    quality = params["jpeg_quality"]

    # Update session history
    _total_time  += elapsed
    _total_count += 1
    session_history.append({
        "filename":  file.filename,
        "timestamp": time.strftime("%H:%M:%S"),
        "elapsed_s": round(elapsed, 2),
        "width":     w,
        "height":    h,
        "thumbnail": _encode(cv2.resize(result_bgr, (80, 80)), "jpg"),
        "params":    {k: v for k, v in params.items()
                      if v != PIPELINE_PARAMS[k]["default"]},
    })

    return jsonify({
        "original":      _encode(img_bgr, fmt, quality),
        "processed":     _encode(result_bgr, fmt, quality),
        "dark_channel":  _encode_gray(dc_vis),
        "message":       "Defogging complete",
        "elapsed":       round(elapsed, 2),
        "width":         w,
        "height":        h,
        "params_used":   params,
    })


@app.route("/process_batch", methods=["POST"])
def process_batch():
    """
    Batch defogging: accepts multiple image files.
    Returns a ZIP file containing all defogged images.
    """
    global _total_time, _total_count

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    # Parse pipeline parameters (shared across the batch)
    params = {}
    for name in PIPELINE_PARAMS:
        params[name] = _parse_param(name, request.form)

    zip_buf = io.BytesIO()
    errors  = []

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            raw = file.read()
            if len(raw) > MAX_BYTES:
                errors.append(f"{file.filename}: too large (>20MB)")
                continue
            np_arr  = np.frombuffer(raw, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                errors.append(f"{file.filename}: cannot decode")
                continue
            try:
                result_bgr, dc_vis, elapsed, h, w = _run_inference(img_bgr, params)
                _total_time  += elapsed
                _total_count += 1
                session_history.append({
                    "filename":  file.filename,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "elapsed_s": round(elapsed, 2),
                    "width": w, "height": h,
                    "thumbnail": _encode(cv2.resize(result_bgr, (80, 80)), "jpg"),
                    "params":    {k: v for k, v in params.items()
                                  if v != PIPELINE_PARAMS[k]["default"]},
                })
                # Save as PNG inside ZIP
                _, buf = cv2.imencode(".png", result_bgr)
                zf.writestr(f"defogged_{file.filename}.png", buf.tobytes())
            except Exception as exc:
                errors.append(f"{file.filename}: {exc}")

    zip_buf.seek(0)
    from flask import Response
    resp = Response(zip_buf.read(), mimetype="application/zip")
    resp.headers["Content-Disposition"] = "attachment; filename=defogged_results.zip"
    if errors:
        resp.headers["X-Errors"] = " | ".join(errors)
    return resp


@app.route("/history")
def history():
    """Return session processing history and aggregate stats."""
    return jsonify({
        "history":    session_history,
        "total":      _total_count,
        "avg_time":   round(_total_time / _total_count, 2) if _total_count else 0,
        "total_time": round(_total_time, 2),
    })


@app.route("/history/clear", methods=["POST"])
def history_clear():
    global session_history, _total_time, _total_count
    session_history = []
    _total_time  = 0.0
    _total_count = 0
    return jsonify({"ok": True})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 60)
    print("  Laparoscopy Defogging Web Interface")
    print("  Open -> http://127.0.0.1:5000")
    print("=" * 60)
    model_inst, init_err = get_model()
    if init_err:
        print(f"\n  WARNING: {init_err}\n")
    else:
        print("  Model loaded on GPU (CUDA)")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
