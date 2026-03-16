"""
Centralised person identification using ONNX face embedding (ArcFace/FaceNet).
Matches detected persons to registered employees by face. Uses optional face detection
or head crop from person bbox, then ONNX embedding and cosine similarity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from config.settings import settings
from src.services.person_detector import Detection

# ONNX session and caches
_onnx_session = None
_face_detector_net = None
_encodings_cache: list[tuple[int, str, np.ndarray]] = []  # (employee_id, path, embedding)

# ArcFace default: input (1, 112, 112, 3) RGB, (x - 127.5) / 128
EMBED_INPUT_H = 112
EMBED_INPUT_W = 112


def _get_embedding_session():
    global _onnx_session
    if _onnx_session is not None:
        return _onnx_session
    path = Path(settings.face_embedding_model_path).resolve()
    if not path.exists():
        _maybe_download_arcface(path)
    if not path.exists():
        logger.warning(
            f"Face embedding model not found at {path}. "
            "Run: python -m scripts.download_arcface"
        )
        return None
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if getattr(settings, "use_gpu", False) else ["CPUExecutionProvider"]
        _onnx_session = ort.InferenceSession(str(path), opts, providers=providers)
        logger.info(f"Loaded ONNX face embedding model: {path}")
        return _onnx_session
    except Exception as e:
        logger.error(f"Failed to load ONNX face model: {e}")
        return None


def _maybe_download_arcface(target_path: Path) -> None:
    """Download ArcFace ONNX model if missing (so you only need pip install -r requirements.txt)."""
    if target_path.exists():
        return
    url = "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx"
    try:
        import urllib.request
        target_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading ArcFace ONNX to {target_path} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "VisionAI"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            target_path.write_bytes(resp.read())
        logger.info("ArcFace model downloaded successfully.")
    except Exception as e:
        logger.warning(f"Auto-download failed: {e}. Run: python -m scripts.download_arcface")


def _get_face_detector():
    """Optional OpenCV DNN face detector (Caffe). Returns None if not configured."""
    global _face_detector_net
    if _face_detector_net is not None:
        return _face_detector_net
    proto = getattr(settings, "face_detector_prototxt", "") or ""
    caffe = getattr(settings, "face_detector_caffemodel", "") or ""
    if not proto or not caffe or not Path(proto).exists() or not Path(caffe).exists():
        return None
    try:
        _face_detector_net = cv2.dnn.readNetFromCaffe(proto, caffe)
        logger.info("Loaded OpenCV DNN face detector")
        return _face_detector_net
    except Exception as e:
        logger.warning(f"Face detector load failed: {e}")
        return None


def _detect_face_in_crop(crop_bgr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) face bbox in crop, or None."""
    net = _get_face_detector()
    if net is None:
        return None
    h, w = crop_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(crop_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets = net.forward()
    for i in range(dets.shape[2]):
        conf = dets[0, 0, i, 2]
        if conf < 0.5:
            continue
        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
    return None


def _get_face_region_from_person_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Use top ~40% of person crop as head/face region when no face detector."""
    h, w = crop_bgr.shape[:2]
    top = int(h * 0.15)
    bottom = int(h * 0.55)
    return crop_bgr[top:bottom, :].copy()


def _preprocess_face_for_embedding(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Resize to 112x112, RGB, normalize. Shape (1, 112, 112, 3)."""
    if face_bgr.size == 0:
        return None
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(face_rgb, (EMBED_INPUT_W, EMBED_INPUT_H), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) - 127.5) / 128.0
    return normalized[np.newaxis, ...]  # (1, 112, 112, 3)


def _embed_face(face_input: np.ndarray) -> Optional[np.ndarray]:
    """Run ONNX model. Returns 512-d unit vector or None."""
    sess = _get_embedding_session()
    if sess is None:
        return None
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: face_input})[0]
    emb = out[0]
    norm = np.linalg.norm(emb)
    if norm < 1e-8:
        return None
    return (emb / norm).astype(np.float32)


def _load_encodings(employee_photos_dir: Path) -> list[tuple[int, str, np.ndarray]]:
    """Load face embeddings from employee photos. Photo filenames must be {employee_id}.jpg."""
    global _encodings_cache
    if _get_embedding_session() is None:
        _encodings_cache = []
        return []
    if not employee_photos_dir.exists():
        _encodings_cache = []
        return []
    encodings: list[tuple[int, str, np.ndarray]] = []
    for path in sorted(employee_photos_dir.glob("*.jpg")):
        try:
            emp_id = int(path.stem)
        except ValueError:
            continue
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Could not read employee photo {path}")
            continue
        face_rect = _detect_face_in_crop(img)
        if face_rect is not None:
            x1, y1, x2, y2 = face_rect
            face_crop = img[y1:y2, x1:x2]
        else:
            face_crop = _get_face_region_from_person_crop(img)
        preprocessed = _preprocess_face_for_embedding(face_crop)
        if preprocessed is None:
            continue
        emb = _embed_face(preprocessed)
        if emb is None:
            logger.warning(f"No embedding for employee photo {path}")
            continue
        encodings.append((emp_id, str(path), emb))
    _encodings_cache = encodings
    logger.info(f"Loaded {len(encodings)} employee face embeddings (ONNX) from {employee_photos_dir}")
    return encodings


def identify_persons(
    frame_rgb: np.ndarray,
    detections: list[Detection],
    employee_photos_dir: Optional[Path] = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    allowed_employee_ids: Optional[list[int]] = None,
) -> list[tuple[int, Optional[int], float]]:
    """
    For each detection (person bbox), extract face region, run ONNX embedding, match to employees.
    If allowed_employee_ids is set, only those employees (e.g. on current shift) are considered;
    others are treated as unknown.
    Returns list of (detection_index, employee_id or None, confidence/similarity).
    """
    enabled = getattr(settings, "face_embedding_enabled", True)
    if not enabled:
        return [(i, None, 0.0) for i in range(len(detections))]
    if _get_embedding_session() is None:
        return [(i, None, 0.0) for i in range(len(detections))]
    dir_path = employee_photos_dir or Path(settings.employee_photos_path)
    if not _encodings_cache:
        _load_encodings(dir_path)
    if not _encodings_cache:
        return [(i, None, 0.0) for i in range(len(detections))]

    # Restrict to employees on current shift when provided
    if allowed_employee_ids is not None:
        enc_list = [e[2] for e in _encodings_cache if e[0] in allowed_employee_ids]
        emp_ids = [e[0] for e in _encodings_cache if e[0] in allowed_employee_ids]
    else:
        enc_list = [e[2] for e in _encodings_cache]
        emp_ids = [e[0] for e in _encodings_cache]
    if not enc_list:
        return [(i, None, 0.0) for i in range(len(detections))]

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]
    threshold = getattr(settings, "face_match_threshold", 0.5)
    results: list[tuple[int, Optional[int], float]] = []

    for idx, det in enumerate(detections):
        x1 = max(0, int(det.x1 * scale_x))
        y1 = max(0, int(det.y1 * scale_y))
        x2 = min(w, int(det.x2 * scale_x))
        y2 = min(h, int(det.y2 * scale_y))
        if x2 <= x1 or y2 <= y1:
            results.append((idx, None, 0.0))
            continue
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            results.append((idx, None, 0.0))
            continue
        try:
            face_rect = _detect_face_in_crop(crop)
            if face_rect is not None:
                fx1, fy1, fx2, fy2 = face_rect
                face_crop = crop[fy1:fy2, fx1:fx2]
            else:
                face_crop = _get_face_region_from_person_crop(crop)
            preprocessed = _preprocess_face_for_embedding(face_crop)
            if preprocessed is None:
                results.append((idx, None, 0.0))
                continue
            emb = _embed_face(preprocessed)
            if emb is None:
                results.append((idx, None, 0.0))
                continue
            sims = np.array([float(np.dot(emb, e)) for e in enc_list])
            best_idx = int(np.argmax(sims))
            similarity = float(sims[best_idx])
            best_emp_id = emp_ids[best_idx]

            # Optional debug logging to understand why a person is (not) matching
            if getattr(settings, "log_face_matches", False):
                # Log top-3 candidates for this detection
                top_k = min(3, len(sims))
                top_indices = np.argsort(-sims)[:top_k]
                top = [(int(emp_ids[i]), float(sims[i])) for i in top_indices]
                logger.info(
                    f"[face-match] det={idx} best_emp={best_emp_id} sim={similarity:.3f} "
                    f"threshold={threshold:.3f} top={top}"
                )

            if similarity >= threshold:
                results.append((idx, best_emp_id, similarity))
            else:
                results.append((idx, None, similarity))
        except Exception as e:
            logger.debug(f"Face embedding error for det {idx}: {e}")
            results.append((idx, None, 0.0))

    return results


def reload_encodings() -> int:
    """Reload employee embeddings (call after adding/updating employees)."""
    dir_path = Path(settings.employee_photos_path)
    return len(_load_encodings(dir_path))
