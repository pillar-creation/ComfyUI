"""MediaPipe landmarks + semantic skin segmentation → beauty masks."""

from __future__ import annotations

import os
import shutil
import ssl
import urllib.request
from pathlib import Path

import cv2
import numpy as np

_MODEL_DIR = Path(__file__).resolve().parent / "models"
_LANDMARKER_PATH = _MODEL_DIR / "face_landmarker.task"
_SEGMENTER_PATH = _MODEL_DIR / "selfie_multiclass_256x256.tflite"
_MODEL_MIN_BYTES = 800_000

_LANDMARKER_URLS = (
    "https://hf-mirror.com/spacepxl/FLAME/resolve/main/SMIRK/face_landmarker.task",
    "https://huggingface.co/spacepxl/FLAME/resolve/main/SMIRK/face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task",
)

_SEGMENTER_URLS = (
    "https://hf-mirror.com/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite",
    "https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite",
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
)

# selfie_multiclass: 0=bg 1=hair 2=body-skin 3=face-skin 4=clothes 5=others
_CAT_BACKGROUND = 0
_CAT_HAIR = 1
_CAT_BODY_SKIN = 2
_CAT_FACE_SKIN = 3

_LANDMARKER_CACHE: dict[tuple[float, float], object] = {}
_SEGMENTER = None

# Workflows often mix 0.5 (detect) and 0.35 (slim/eye); warm both at startup.
_DEFAULT_LANDMARKER_CONFIDENCES = ((0.5, 0.5), (0.35, 0.5))


def sanitize_landmark_confidence(value, *, default: float = 0.5) -> float:
    """Clamp to MediaPipe range; reject misaligned widget values (e.g. 5 from mask_edge_blur)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v > 1.0 or v < 0.0:
        return default
    return max(0.1, min(0.9, v))


def sanitize_brow_thickness(value, *, default: float = 1.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v > 2.5:
        return default
    return max(0.5, min(2.0, v))


def sanitize_mask_edge_blur(value, *, default: int = 5) -> int:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    if v > 21:
        return default
    return max(0, min(21, v))

SKIN_MASK_MODES = ("semantic", "landmarks")


def _indices_from_connections(connections) -> list[int]:
    idx: set[int] = set()
    for a, b in connections:
        idx.add(int(a))
        idx.add(int(b))
    return sorted(idx)


def _load_region_index_sets() -> dict[str, list[int]]:
    _cheeks = {
        "left_cheek": [50, 101, 36, 205, 187, 123, 116, 147, 213, 192, 214, 204, 203, 142, 126],
        "right_cheek": [280, 330, 371, 266, 411, 352, 345, 376, 433, 416, 434, 432, 427],
        "nasolabial_left": [266, 426, 436, 416, 352, 347, 330, 423, 391, 322, 410],
        "nasolabial_right": [36, 206, 216, 192, 147, 123, 117, 118, 101, 205, 187],
    }
    try:
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as FLC

        def _conn_idx(conns) -> list[int]:
            idx: set[int] = set()
            for c in conns:
                idx.add(int(c.start))
                idx.add(int(c.end))
            return sorted(idx)

        return {
            "face_oval": _conn_idx(FLC.FACE_LANDMARKS_FACE_OVAL),
            "left_eye": _conn_idx(FLC.FACE_LANDMARKS_LEFT_EYE),
            "right_eye": _conn_idx(FLC.FACE_LANDMARKS_RIGHT_EYE),
            "left_eyebrow": _conn_idx(FLC.FACE_LANDMARKS_LEFT_EYEBROW),
            "right_eyebrow": _conn_idx(FLC.FACE_LANDMARKS_RIGHT_EYEBROW),
            "nose": _conn_idx(FLC.FACE_LANDMARKS_NOSE),
            "lips": _conn_idx(FLC.FACE_LANDMARKS_LIPS),
            **_cheeks,
        }
    except Exception:
        pass
    try:
        from mediapipe.python.solutions import face_mesh_connections as fmc

        return {
            "face_oval": _indices_from_connections(fmc.FACEMESH_FACE_OVAL),
            "left_eye": _indices_from_connections(fmc.FACEMESH_LEFT_EYE),
            "right_eye": _indices_from_connections(fmc.FACEMESH_RIGHT_EYE),
            "left_eyebrow": _indices_from_connections(fmc.FACEMESH_LEFT_EYEBROW),
            "right_eyebrow": _indices_from_connections(fmc.FACEMESH_RIGHT_EYEBROW),
            "nose": _indices_from_connections(fmc.FACEMESH_NOSE),
            "lips": _indices_from_connections(fmc.FACEMESH_LIPS),
            **_cheeks,
        }
    except Exception:
        return {
            "face_oval": [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
                379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                234, 127, 162, 21, 54, 103, 67, 109,
            ],
            "left_eye": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
            "right_eye": [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
            "left_eyebrow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
            "right_eyebrow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
            "nose": [
                168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 294, 278,
                344, 440, 275, 45, 220, 115, 48, 64,
            ],
            "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
            **_cheeks,
        }


_REGION_INDEX_SETS = _load_region_index_sets()

# MediaPipe Face Landmarker (478 pts) — official eye contours (not legacy mesh names)
_FACE_LANDMARKER_LEFT_EYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
]
_FACE_LANDMARKER_RIGHT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
]

# Closed eyelid loop order (MediaPipe FACE_LANDMARKS_*_EYE connection walk)
_EYE_CONTOUR_ORDER = {
    "left_eye": [
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        398, 384, 385, 386, 387, 388, 466,
    ],
    "right_eye": [
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        173, 157, 158, 159, 160, 161, 246,
    ],
}


def _order_contour_from_connections(connections) -> list[int]:
    """Walk eyelid edges into one closed polygon (outer corner → lids → outer corner)."""
    edges: list[tuple[int, int]] = []
    for c in connections:
        edges.append((int(c.start), int(c.end)))
    if not edges:
        return []

    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    start = edges[0][0]
    order = [start]
    prev, cur = -1, start
    for _ in range(len(adj) * 4):
        nbrs = [n for n in adj.get(cur, []) if n != prev]
        if not nbrs:
            break
        nxt = nbrs[0]
        if nxt == start and len(order) >= 3:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    return order


_EYE_CORNERS: dict[str, tuple[int, int]] = {
    "left_eye": (263, 362),
    "right_eye": (33, 133),
}

# Upper eyelid crease / lash ridge (Face Landmarker)
_UPPER_LID_RIDGE: dict[str, list[int]] = {
    "left_eye": [466, 388, 387, 386, 385, 384, 398],
    "right_eye": [246, 161, 160, 159, 158, 157, 173],
}


def get_eye_contour_polygon_order() -> dict[str, list[int]]:
    """Eyelid indices in contour order (not sorted by index — required for fillPoly)."""
    try:
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as FLC

        return {
            "left_eye": _order_contour_from_connections(FLC.FACE_LANDMARKS_LEFT_EYE),
            "right_eye": _order_contour_from_connections(FLC.FACE_LANDMARKS_RIGHT_EYE),
        }
    except Exception:
        return {k: list(v) for k, v in _EYE_CONTOUR_ORDER.items()}


def get_face_oval_polygon_order() -> list[int]:
    """Face oval indices in contour order (fillPoly — not convex hull / sorted index)."""
    try:
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as FLC

        order = _order_contour_from_connections(FLC.FACE_LANDMARKS_FACE_OVAL)
        if len(order) >= 3:
            return order
    except Exception:
        pass
    return list(_REGION_INDEX_SETS["face_oval"])


def get_eye_region_index_sets() -> dict[str, list[int]]:
    """Eye index sets for Face Landmarker task (subject left / right eye)."""
    try:
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as FLC

        def _conn_idx(conns) -> list[int]:
            idx: set[int] = set()
            for c in conns:
                idx.add(int(c.start))
                idx.add(int(c.end))
            return sorted(idx)

        return {
            "left_eye": _conn_idx(FLC.FACE_LANDMARKS_LEFT_EYE),
            "right_eye": _conn_idx(FLC.FACE_LANDMARKS_RIGHT_EYE),
        }
    except Exception:
        return {
            "left_eye": list(_FACE_LANDMARKER_LEFT_EYE),
            "right_eye": list(_FACE_LANDMARKER_RIGHT_EYE),
        }


def image_to_bgr_uint8(frame_rgb: np.ndarray) -> np.ndarray:
    if frame_rgb.ndim == 2:
        return frame_rgb
    return cv2.cvtColor(frame_rgb[:, :, :3], cv2.COLOR_RGB2BGR)


def bgr_to_rgb_float01(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _download_via_urllib(url: str, dest: Path, *, insecure: bool = False) -> None:
    ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Simple-Face-Mask/2.1"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(dest)


def _download_via_requests(url: str, dest: Path) -> None:
    import requests

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "ComfyUI-Simple-Face-Mask/2.1"}) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def _download_model_file(url: str, dest: Path) -> None:
    errors: list[str] = []
    for fn, kwargs in (
        (_download_via_requests, {"url": url, "dest": dest}),
        (_download_via_urllib, {"url": url, "dest": dest, "insecure": False}),
        (_download_via_urllib, {"url": url, "dest": dest, "insecure": True}),
    ):
        try:
            fn(**kwargs)
            if dest.is_file() and dest.stat().st_size >= _MODEL_MIN_BYTES:
                return
            if dest.is_file():
                dest.unlink(missing_ok=True)
            errors.append(f"{fn.__name__}: file too small")
        except Exception as exc:
            errors.append(f"{fn.__name__}: {exc}")
            dest.unlink(missing_ok=True)
    raise RuntimeError("; ".join(errors))


def _ensure_model(path: Path, urls: tuple[str, ...], env_key: str, label: str) -> Path:
    override = os.environ.get(env_key, "").strip()
    if override:
        custom = Path(override)
        if custom.is_file() and custom.stat().st_size >= _MODEL_MIN_BYTES:
            return custom
        raise FileNotFoundError(f"{env_key} 无效: {custom}")

    if path.is_file() and path.stat().st_size >= _MODEL_MIN_BYTES:
        return path

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    if label == "face_landmarker":
        try:
            from huggingface_hub import hf_hub_download

            cached = hf_hub_download(
                repo_id="spacepxl/FLAME",
                filename="SMIRK/face_landmarker.task",
                endpoint="https://hf-mirror.com",
            )
            shutil.copy2(cached, path)
            if path.stat().st_size >= _MODEL_MIN_BYTES:
                return path
        except Exception as exc:
            failures.append(f"hf-mirror landmarker -> {exc}")
            path.unlink(missing_ok=True)

    for url in urls:
        try:
            _download_model_file(url, path)
            return path
        except Exception as exc:
            failures.append(f"{url} -> {exc}")

    raise RuntimeError(
        f"无法下载 {label}，请手动放到 {path}\n" + "\n".join(failures[:3])
    )


def ensure_landmarker_model() -> Path:
    return _ensure_model(
        _LANDMARKER_PATH,
        _LANDMARKER_URLS,
        "COMFYUI_FACE_LANDMARKER_MODEL",
        "face_landmarker",
    )


def ensure_segmenter_model() -> Path:
    return _ensure_model(
        _SEGMENTER_PATH,
        _SEGMENTER_URLS,
        "COMFYUI_SELFIE_SEGMENTER_MODEL",
        "selfie_multiclass_256x256.tflite",
    )


def _get_landmarker(min_detection_confidence: float, min_presence_confidence: float):
    det = sanitize_landmark_confidence(min_detection_confidence)
    pres = sanitize_landmark_confidence(min_presence_confidence)
    params = (det, pres)
    cached = _LANDMARKER_CACHE.get(params)
    if cached is not None:
        return cached

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_landmarker_model())),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=3,
        min_face_detection_confidence=det,
        min_face_presence_confidence=pres,
        min_tracking_confidence=pres,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    _LANDMARKER_CACHE[params] = landmarker
    return landmarker


def _get_segmenter():
    global _SEGMENTER
    if _SEGMENTER is not None:
        return _SEGMENTER

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_segmenter_model())),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=False,
    )
    _SEGMENTER = vision.ImageSegmenter.create_from_options(options)
    return _SEGMENTER


def preload_mediapipe(
    *,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    load_segmenter: bool = True,
    extra_landmarker_confidences: tuple[tuple[float, float], ...] = _DEFAULT_LANDMARKER_CONFIDENCES,
) -> None:
    """Eagerly download models and create MediaPipe task runners (startup warmup)."""
    import time

    import mediapipe as mp  # noqa: F401

    t0 = time.perf_counter()
    ensure_landmarker_model()
    seen: set[tuple[float, float]] = set()
    for det, pres in ((min_detection_confidence, min_presence_confidence), *extra_landmarker_confidences):
        key = (
            sanitize_landmark_confidence(det),
            sanitize_landmark_confidence(pres),
        )
        if key in seen:
            continue
        seen.add(key)
        _get_landmarker(key[0], key[1])
    if load_segmenter:
        ensure_segmenter_model()
        _get_segmenter()
    elapsed = time.perf_counter() - t0
    print(
        f"[ComfyUI-Simple-Face-Mask] MediaPipe preloaded in {elapsed:.1f}s "
        f"({len(_LANDMARKER_CACHE)} landmarker, segmenter={'yes' if _SEGMENTER else 'no'})"
    )


def _landmark_xy(landmarks, index: int, width: int, height: int) -> tuple[int, int]:
    lm = landmarks[index]
    return int(lm.x * width), int(lm.y * height)


def _points_for_indices(landmarks, width: int, height: int, indices: list[int]) -> np.ndarray:
    pts = []
    n = len(landmarks)
    for i in indices:
        if 0 <= i < n:
            pts.append(_landmark_xy(landmarks, i, width, height))
    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(pts, dtype=np.int32)


def _polyline_points_for_indices(
    landmarks, width: int, height: int, indices: list[int]
) -> np.ndarray:
    """At least 2 points — for open polylines (nasolabial paths)."""
    pts = []
    n = len(landmarks)
    for i in indices:
        if 0 <= i < n:
            pts.append(_landmark_xy(landmarks, i, width, height))
    if len(pts) < 2:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(pts, dtype=np.int32)


def _fill_polygon_mask(height: int, width: int, pts: np.ndarray, blur: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if pts.shape[0] >= 3:
        cv2.fillConvexPoly(mask, pts, 255)
    if blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        k = max(3, min(k, 51))
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0


def _fill_poly_mask(height: int, width: int, pts: np.ndarray, blur: int = 0) -> np.ndarray:
    """Fill closed contour (non-convex eyelid loop)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    if blur > 0:
        k = max(3, min(blur | 1, 51))
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0


def _mean_point_spacing(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 3.0
    dists = np.linalg.norm(np.diff(pts.astype(np.float32), axis=0), axis=1)
    return max(2.0, float(np.mean(dists)))


def _arc_indices_on_loop(order: list[int], a: int, b: int) -> tuple[list[int], list[int]]:
    ia, ib = order.index(a), order.index(b)
    n = len(order)
    forward: list[int] = []
    i = ia
    while True:
        forward.append(order[i])
        if order[i] == b:
            break
        i = (i + 1) % n
    backward: list[int] = []
    i = ia
    while True:
        backward.append(order[i])
        if order[i] == b:
            break
        i = (i - 1) % n
    return forward, backward


def _split_eye_upper_lower(
    order: list[int],
    outer: int,
    inner: int,
    landmarks,
    width: int,
    height: int,
) -> tuple[list[int], list[int]]:
    path_a, path_b = _arc_indices_on_loop(order, outer, inner)

    def _mean_y(indices: list[int]) -> float:
        pts = _points_for_indices(landmarks, width, height, indices)
        return float(pts[:, 1].mean()) if pts.shape[0] else 0.0

    if _mean_y(path_a) <= _mean_y(path_b):
        return path_a, path_b
    return path_b, path_a


def _resample_polyline(pts: np.ndarray, n_pts: int) -> np.ndarray:
    if pts.shape[0] < 2 or n_pts < 2:
        return pts
    pts_f = pts.astype(np.float32)
    seg = np.linalg.norm(np.diff(pts_f, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-3:
        return np.repeat(pts_f[:1], n_pts, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, n_pts)
    out = np.zeros((n_pts, 2), dtype=np.float32)
    j = 0
    for i, t in enumerate(targets):
        while j < len(seg) - 1 and cum[j + 1] < t:
            j += 1
        seg_len = max(seg[j], 1e-6)
        alpha = (t - cum[j]) / seg_len
        out[i] = pts_f[j] * (1.0 - alpha) + pts_f[j + 1] * alpha
    return out


def _fill_band_between_polylines(
    height: int,
    width: int,
    lower_line: np.ndarray,
    upper_line: np.ndarray,
) -> np.ndarray:
    """Fill the strip between two polylines (same outer→inner direction)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if lower_line.shape[0] < 2 or upper_line.shape[0] < 2:
        return mask.astype(np.float32) / 255.0
    n_pts = max(int(max(lower_line.shape[0], upper_line.shape[0])), 4)
    low = _resample_polyline(lower_line, n_pts)
    up = _resample_polyline(upper_line, n_pts)
    for i in range(n_pts - 1):
        quad = np.array([low[i], low[i + 1], up[i + 1], up[i]], dtype=np.int32)
        cv2.fillConvexPoly(mask, quad, 255)
    return mask.astype(np.float32) / 255.0


def _canthus_pad_mask(
    landmarks,
    width: int,
    height: int,
    outer: int,
    radius: float,
) -> np.ndarray:
    """Pad at outer canthus only (inner pad bleeds onto nose bridge)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    r = max(2, int(round(radius)))
    if outer < len(landmarks):
        x, y = _landmark_xy(landmarks, outer, width, height)
        cv2.circle(mask, (x, y), r, 255, -1, lineType=cv2.LINE_AA)
    if r >= 2:
        k = r | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0


def _offset_polyline_from_center(
    pts: np.ndarray,
    center: np.ndarray,
    distance: float,
    *,
    inner_taper: int = 4,
) -> np.ndarray:
    """Push lid points away from eye center; taper to 0 near inner canthus (nose side)."""
    if pts.shape[0] < 1 or distance <= 0:
        return pts.astype(np.float32)
    n = len(pts)
    out = np.zeros_like(pts, dtype=np.float32)
    taper = max(1, inner_taper)
    for i, p in enumerate(pts.astype(np.float32)):
        if i >= n - taper:
            w = (n - 1 - i) / taper
        else:
            w = 1.0
        dist_i = distance * max(0.0, min(1.0, w))
        if dist_i <= 0:
            out[i] = p
            continue
        v = p - center
        length = max(float(np.hypot(v[0], v[1])), 1e-6)
        out[i] = p + dist_i * (v / length)
    return out


def _offset_polyline_downward(
    pts: np.ndarray,
    distance: float,
    *,
    inner_end_taper: int = 3,
) -> np.ndarray:
    """Extend lower-lid polyline downward (+Y); taper only at inner canthus (polyline end)."""
    if pts.shape[0] < 1 or distance <= 0:
        return pts.astype(np.float32)
    n = len(pts)
    out = pts.astype(np.float32).copy()
    taper = max(0, inner_end_taper)
    for i in range(n):
        if taper > 0 and i >= n - taper:
            w = (n - 1 - i) / taper
        else:
            w = 1.0
        out[i, 1] += distance * w
    return out


def _outer_canthus_bridge_mask(
    height: int,
    width: int,
    upper_pts: np.ndarray,
    lower_pts: np.ndarray,
    center: np.ndarray,
    landmarks,
    outer: int,
    inner: int,
    spacing: float,
) -> np.ndarray:
    """
    Fill the outer-canthus wedge: arc between the lateral tips of upper/lower lid bands.
    Paths are outer → inner; index 0 is the outer canthus.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if upper_pts.shape[0] < 2 or lower_pts.shape[0] < 2:
        return mask.astype(np.float32) / 255.0

    dist_up = max(3.0, spacing * 2.8)
    dist_low_rad = max(3.0, spacing * 2.5)
    dist_low_down = max(5.0, spacing * 3.6)
    upper_ext = _offset_polyline_from_center(
        upper_pts, center, dist_up, inner_taper=0
    )
    lower_rad = _offset_polyline_from_center(
        lower_pts, center, dist_low_rad, inner_taper=0
    )
    lower_down = _offset_polyline_downward(lower_pts, dist_low_down, inner_end_taper=0)
    lower_ext = np.maximum(lower_rad, lower_down)

    pt_up = upper_ext[0].astype(np.float32)
    pt_lo = lower_ext[0].astype(np.float32)
    ox, oy = _landmark_xy(landmarks, outer, width, height)
    ix, iy = _landmark_xy(landmarks, inner, width, height)
    corner = np.array([float(ox), float(oy)], dtype=np.float32)

    lateral = np.array([float(ox - ix), float(oy - iy)], dtype=np.float32)
    lat_len = max(float(np.hypot(lateral[0], lateral[1])), 1e-6)
    lat_unit = lateral / lat_len
    bulge = max(spacing * 0.9, 4.0)
    ctrl = (pt_up + pt_lo) * 0.5 + lat_unit * bulge

    n_arc = max(10, int(spacing * 1.2))
    ts = np.linspace(0.0, 1.0, n_arc, dtype=np.float32)
    arc = (
        (1.0 - ts)[:, np.newaxis] ** 2 * pt_up
        + 2.0 * (1.0 - ts)[:, np.newaxis] * ts[:, np.newaxis] * ctrl
        + ts[:, np.newaxis] ** 2 * pt_lo
    )
    poly = np.vstack([pt_up, arc, pt_lo, corner]).astype(np.int32)
    cv2.fillConvexPoly(mask, poly, 255, lineType=cv2.LINE_AA)
    return mask.astype(np.float32) / 255.0


def _clip_eye_mask_from_nose(
    mask: np.ndarray,
    landmarks,
    width: int,
    outer: int,
    inner: int,
    spacing: float,
) -> np.ndarray:
    """Remove mask pixels on the nose side of the inner canthus."""
    if mask.max() <= 0:
        return mask
    ix, _ = _landmark_xy(landmarks, inner, width, mask.shape[0])
    ox, _ = _landmark_xy(landmarks, outer, width, mask.shape[0])
    allow = max(1, int(spacing * 0.2))
    out = mask.copy()
    if ox > ix:
        out[:, : max(0, ix - allow)] = 0.0
    else:
        out[:, min(width, ix + allow) :] = 0.0
    return out


def _lid_skin_band_mask(
    height: int,
    width: int,
    lid_pts: np.ndarray,
    center: np.ndarray,
    spacing: float,
    *,
    upper: bool,
) -> np.ndarray:
    """Eyelid skin: band from lid arc outward (distance = mesh spacing, not brow)."""
    if lid_pts.shape[0] < 2:
        return np.zeros((height, width), dtype=np.float32)
    if upper:
        dist = max(3.0, spacing * 2.8)
        outer = _offset_polyline_from_center(lid_pts, center, dist)
        return _fill_band_between_polylines(height, width, lid_pts, outer)
    # Lower lid: radial + downward so palpebral skin below lash line is fully covered.
    dist_rad = max(3.0, spacing * 2.5)
    dist_down = max(5.0, spacing * 3.6)
    outer_rad = _offset_polyline_from_center(
        lid_pts, center, dist_rad, inner_taper=2
    )
    outer_down = _offset_polyline_downward(lid_pts, dist_down, inner_end_taper=3)
    outer = np.maximum(outer_rad, outer_down)
    return _fill_band_between_polylines(height, width, lid_pts, outer)


def _ridge_ordered_indices(
    ridge: list[int],
    landmarks,
    width: int,
    height: int,
    outer: int,
    inner: int,
) -> list[int]:
    """Order upper-lid ridge landmarks outer → inner."""
    if not ridge:
        return []
    ox, _ = _landmark_xy(landmarks, outer, width, height)
    ix, _ = _landmark_xy(landmarks, inner, width, height)
    items: list[tuple[int, int]] = []
    for idx in ridge:
        if idx < len(landmarks):
            items.append((idx, _landmark_xy(landmarks, idx, width, height)[0]))
    if not items:
        return []
    if ox > ix:
        items.sort(key=lambda t: -t[1])
    else:
        items.sort(key=lambda t: t[1])
    return [idx for idx, _ in items]


def _polyline_stroke_mask(
    height: int,
    width: int,
    pts: np.ndarray,
    thickness: float,
) -> np.ndarray:
    """Lash / lid line: stroke width from landmark spacing, not image-scale expand."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if pts.shape[0] < 2:
        return mask.astype(np.float32) / 255.0
    thick = max(1, int(round(thickness)))
    cv2.polylines(
        mask,
        [pts.reshape(-1, 1, 2).astype(np.int32)],
        False,
        255,
        thick,
        cv2.LINE_AA,
    )
    if thick >= 2:
        k = thick | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0


def _one_periocular_eye_mask(
    landmarks,
    width: int,
    height: int,
    eye_key: str,
    eye_order: list[int],
    outer: int,
    inner: int,
) -> np.ndarray:
    """
    Periocular mask: eye opening + lid/lash bands (radial from eye center, landmark spacing).
    """
    union = np.zeros((height, width), dtype=np.float32)
    if len(eye_order) < 3:
        return union

    opening_pts = _points_for_indices(landmarks, width, height, eye_order)
    if opening_pts.shape[0] >= 3:
        union = np.maximum(union, _fill_poly_mask(height, width, opening_pts, blur=0))

    upper_idx, lower_idx = _split_eye_upper_lower(
        eye_order, outer, inner, landmarks, width, height
    )
    upper_pts = _points_for_indices(landmarks, width, height, upper_idx)
    lower_pts = _points_for_indices(landmarks, width, height, lower_idx)
    spacing = _mean_point_spacing(upper_pts if upper_pts.shape[0] >= 2 else lower_pts)
    center = opening_pts.astype(np.float32).mean(axis=0)

    if upper_pts.shape[0] >= 2:
        union = np.maximum(
            union,
            _lid_skin_band_mask(height, width, upper_pts, center, spacing, upper=True),
        )
        union = np.maximum(
            union, _polyline_stroke_mask(height, width, upper_pts, spacing * 1.35)
        )
        ridge_idx = _ridge_ordered_indices(
            _UPPER_LID_RIDGE.get(eye_key, []), landmarks, width, height, outer, inner
        )
        ridge_pts = _points_for_indices(landmarks, width, height, ridge_idx)
        if ridge_pts.shape[0] >= 2:
            union = np.maximum(
                union,
                _polyline_stroke_mask(height, width, ridge_pts, spacing * 1.25),
            )
            union = np.maximum(
                union,
                _lid_skin_band_mask(height, width, ridge_pts, center, spacing * 0.85, upper=True),
            )

    if lower_pts.shape[0] >= 2:
        lower_spacing = _mean_point_spacing(lower_pts)
        union = np.maximum(
            union,
            _lid_skin_band_mask(
                height, width, lower_pts, center, lower_spacing, upper=False
            ),
        )
        union = np.maximum(
            union,
            _polyline_stroke_mask(height, width, lower_pts, lower_spacing * 1.75),
        )

    if upper_pts.shape[0] >= 2 and lower_pts.shape[0] >= 2:
        union = np.maximum(
            union,
            _outer_canthus_bridge_mask(
                height,
                width,
                upper_pts,
                lower_pts,
                center,
                landmarks,
                outer,
                inner,
                spacing,
            ),
        )
    union = np.maximum(
        union,
        _canthus_pad_mask(landmarks, width, height, outer, spacing * 1.15),
    )
    return _clip_eye_mask_from_nose(union, landmarks, width, outer, inner, spacing)


def build_periocular_eye_mask(
    landmarks,
    height: int,
    width: int,
    mask_edge_blur: int = 9,
) -> np.ndarray:
    """Eye opening + eyelid/lash bands (radial from eye center)."""
    eye_orders = get_eye_contour_polygon_order()
    union = np.zeros((height, width), dtype=np.float32)
    for key, (outer, inner) in _EYE_CORNERS.items():
        order = eye_orders.get(key, _EYE_CONTOUR_ORDER.get(key, []))
        union = np.maximum(
            union,
            _one_periocular_eye_mask(
                landmarks, width, height, key, order, outer, inner
            ),
        )
    if mask_edge_blur > 0 and union.max() > 0:
        k = max(3, min(mask_edge_blur | 1, 15))
        union = cv2.GaussianBlur(union, (k, k), 0)
    return union


def build_full_eye_union_mask(
    landmarks,
    height: int,
    width: int,
    mask_edge_blur: int = 9,
) -> np.ndarray:
    """Periocular mask: opening + eyelid skin + lash bands (landmark-based, no global scale)."""
    return build_periocular_eye_mask(landmarks, height, width, mask_edge_blur)


def _soften_mask_edges(mask: np.ndarray, blur: int) -> np.ndarray:
    if blur <= 0:
        return mask
    k = blur if blur % 2 == 1 else blur + 1
    k = max(3, min(k, 51))
    return cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)


def _dilate_mask(mask: np.ndarray, padding_percent: float, height: int, width: int) -> np.ndarray:
    if padding_percent <= 0:
        return mask
    px = max(1, int(min(height, width) * padding_percent * 0.25))
    k = px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate((mask * 255).astype(np.uint8), kernel, iterations=1)
    return dilated.astype(np.float32) / 255.0


def _subtract_clipped(base: np.ndarray, *subs: np.ndarray) -> np.ndarray:
    out = base.copy()
    for sub in subs:
        out = np.clip(out - sub, 0.0, 1.0)
    return out


def _landmark_bbox(
    landmarks,
    width: int,
    height: int,
    padding_percent: float,
) -> tuple[int, int, int, int]:
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    fh = max(12.0, y1 - y0)
    fw = max(12.0, x1 - x0)
    pad = float(np.clip(padding_percent, 0.0, 0.35))
    mx = fw * (0.18 + pad * 0.45)
    my_top = fh * (0.55 + pad * 0.85)
    my_bot = fh * (0.42 + pad * 0.75)
    left = int(max(0, x0 - mx))
    right = int(min(width - 1, x1 + mx))
    top = int(max(0, y0 - my_top))
    bottom = int(min(height - 1, y1 + my_bot))
    return left, top, right, bottom


def _roi_mask_from_bbox(height: int, width: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = bbox
    roi = np.zeros((height, width), dtype=np.float32)
    roi[top : bottom + 1, left : right + 1] = 1.0
    return roi


def _category_mask_numpy(segmentation_result, height: int, width: int) -> np.ndarray:
    cat = segmentation_result.category_mask
    if cat is None:
        raise RuntimeError("ImageSegmenter returned no category_mask")
    arr = np.asarray(cat.numpy_view(), dtype=np.uint8)
    # MediaPipe may return (H, W) or (H, W, 1)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        arr = arr.reshape(height, width)
    if arr.shape[0] != height or arr.shape[1] != width:
        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST)
    return arr


def _semantic_skin_mask(
    frame_rgb: np.ndarray,
    landmarks,
    padding_percent: float,
) -> np.ndarray | None:
    """Pixel-wise skin from MediaPipe selfie multiclass (follows real edges)."""
    import mediapipe as mp

    h, w = frame_rgb.shape[:2]
    bbox = _landmark_bbox(landmarks, w, h, padding_percent)
    roi = _roi_mask_from_bbox(h, w, bbox)

    segmenter = _get_segmenter()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb[:, :, :3]))
    result = segmenter.segment(mp_image)
    cats = _category_mask_numpy(result, h, w)

    roi_bool = roi.astype(bool)
    face_skin = cats == _CAT_FACE_SKIN
    body_skin = (cats == _CAT_BODY_SKIN) & roi_bool
    hair = cats == _CAT_HAIR
    skin = (face_skin | body_skin) & ~hair
    skin = skin.astype(np.uint8) * 255
    k = max(3, int(min(h, w) * 0.012) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=1)
    skin = cv2.dilate(skin, kernel, iterations=1)
    skin = skin.astype(np.float32) / 255.0
    skin = np.clip(skin * roi, 0.0, 1.0)
    if float(skin.mean()) < 0.001:
        return None
    return skin


def _grabcut_refine_skin(bgr: np.ndarray, init_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Refine mask along color/edge boundaries (similar spirit to magnetic lasso + refine edge)."""
    h, w = init_mask.shape
    left, top, right, bottom = bbox
    gc = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc[init_mask > 0.35] = cv2.GC_PR_FGD
    gc[init_mask > 0.72] = cv2.GC_FGD
    gc[:top, :] = cv2.GC_BGD
    gc[bottom + 1 :, :] = cv2.GC_BGD
    gc[:, :left] = cv2.GC_BGD
    gc[:, right + 1 :] = cv2.GC_BGD

    rect = (left, top, max(1, right - left), max(1, bottom - top))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return init_mask

    refined = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    roi = _roi_mask_from_bbox(h, w, bbox)
    out = np.clip(refined * roi, 0.0, 1.0)
    if float(out.mean()) > 0.003:
        return np.clip(np.maximum(out, init_mask * 0.55), 0.0, 1.0)
    return init_mask


def _joint_bilateral_smooth(mask: np.ndarray, bgr: np.ndarray, blur: int) -> np.ndarray:
    if blur <= 0:
        return mask
    m8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    d = max(3, min(blur | 1, 31))
    sigma = max(12, blur * 2)
    try:
        smooth = cv2.ximgproc.jointBilateralFilter(bgr, m8, d, sigma, sigma)
    except (AttributeError, cv2.error):
        smooth = cv2.bilateralFilter(m8, d, sigma, sigma)
    return smooth.astype(np.float32) / 255.0


def _landmark_oval_skin_mask(
    landmarks,
    height: int,
    width: int,
    padding_percent: float,
    mask_edge_blur: int,
) -> np.ndarray:
    idx = _REGION_INDEX_SETS
    oval = _fill_polygon_mask(
        height, width, _points_for_indices(landmarks, width, height, idx["face_oval"]), mask_edge_blur
    )
    left_eye = _fill_polygon_mask(
        height, width, _points_for_indices(landmarks, width, height, idx["left_eye"]), mask_edge_blur
    )
    right_eye = _fill_polygon_mask(
        height, width, _points_for_indices(landmarks, width, height, idx["right_eye"]), mask_edge_blur
    )
    lips = _fill_polygon_mask(
        height, width, _points_for_indices(landmarks, width, height, idx["lips"]), mask_edge_blur
    )
    skin = _subtract_clipped(oval, left_eye, right_eye, lips)
    return _dilate_mask(skin, padding_percent, height, width)


def _expand_convex_mask(
    mask: np.ndarray,
    height: int,
    width: int,
    expand: float,
) -> np.ndarray:
    if expand <= 1.001 or mask.max() <= 0:
        return mask
    px = max(1, int(min(height, width) * (expand - 1.0) * 0.08))
    k = px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate((mask * 255).astype(np.uint8), kernel, iterations=1)
    return dilated.astype(np.float32) / 255.0


def _brow_region_mask(
    landmarks,
    width: int,
    height: int,
    indices: list[int],
    blur: int,
    thickness_scale: float = 1.0,
) -> np.ndarray:
    pts = _points_for_indices(landmarks, width, height, indices)
    if pts.shape[0] < 2:
        return np.zeros((height, width), dtype=np.float32)
    spacing = _mean_point_spacing(pts)
    thickness = max(2, int(spacing * 1.35 * thickness_scale))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(mask, [pts.astype(np.int32)], False, 255, thickness)
    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    if blur > 0:
        k = max(3, min(blur | 1, 15))
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0


def _nose_region_mask(
    landmarks,
    width: int,
    height: int,
    indices: list[int],
    blur: int,
    expand: float = 1.12,
) -> np.ndarray:
    pts = _points_for_indices(landmarks, width, height, indices)
    if pts.shape[0] < 3:
        return np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(pts.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    out = mask.astype(np.float32) / 255.0
    out = _expand_convex_mask(out, height, width, expand)
    if blur > 0:
        k = max(3, min(blur | 1, 15))
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def _mouth_region_mask(
    landmarks,
    width: int,
    height: int,
    indices: list[int],
    blur: int,
    expand: float = 1.08,
) -> np.ndarray:
    pts = _points_for_indices(landmarks, width, height, indices)
    if pts.shape[0] < 3:
        return np.zeros((height, width), dtype=np.float32)
    out = _fill_poly_mask(height, width, pts, blur)
    return _expand_convex_mask(out, height, width, expand)


def _refine_face_oval_points(
    pts: np.ndarray,
    landmarks,
    width: int,
    height: int,
    *,
    forehead_trim: float = 0.0,
    face_inset: float = 0.0,
    temple_trim: float = 0.0,
) -> np.ndarray:
    """Pull upper oval down toward brows, shrink temples — keeps hair out of face region."""
    if pts.shape[0] < 3:
        return pts
    out = pts.astype(np.float32).copy()
    center = out.mean(axis=0)

    inset_s = float(np.clip(face_inset, 0.0, 100.0)) / 100.0
    if inset_s > 0:
        scale = 1.0 - inset_s * 0.28
        out = center + (out - center) * scale

    trim_s = float(np.clip(forehead_trim, 0.0, 100.0)) / 100.0
    temple_s = float(np.clip(temple_trim, 0.0, 100.0)) / 100.0
    if temple_s <= 0 and trim_s > 0:
        temple_s = trim_s * 0.85

    idx = _REGION_INDEX_SETS
    brow_pts = np.vstack(
        [
            _points_for_indices(landmarks, width, height, idx["left_eyebrow"]),
            _points_for_indices(landmarks, width, height, idx["right_eyebrow"]),
        ]
    )
    if brow_pts.shape[0] >= 2:
        brow_top = float(brow_pts[:, 1].min())
        brow_spacing = _mean_point_spacing(brow_pts)
        if trim_s > 0:
            cap_y = brow_top - brow_spacing * (0.12 + trim_s * 0.22)
            y_min = float(out[:, 1].min())
            face_h = max(float(out[:, 1].max()) - y_min, 1.0)
            for i in range(len(out)):
                if out[i, 1] < cap_y:
                    t = trim_s * np.clip((cap_y - out[i, 1]) / (face_h * 0.32), 0.0, 1.0)
                    out[i, 1] = out[i, 1] * (1.0 - t) + cap_y * t
                elif out[i, 1] < brow_top + brow_spacing * 0.6:
                    out[i, 0] = center[0] + (out[i, 0] - center[0]) * (1.0 - trim_s * 0.12)

        if temple_s > 0:
            outer_rx, _ = _landmark_xy(landmarks, _EYE_CORNERS["right_eye"][0], width, height)
            outer_lx, _ = _landmark_xy(landmarks, _EYE_CORNERS["left_eye"][0], width, height)
            margin = brow_spacing * 0.35
            temple_y_max = brow_top + brow_spacing * 0.95
            for i in range(len(out)):
                x, y = out[i, 0], out[i, 1]
                if y >= temple_y_max:
                    continue
                at_right_temple = x < outer_rx - margin
                at_left_temple = x > outer_lx + margin
                if not (at_right_temple or at_left_temple):
                    continue
                lateral = temple_s * (0.22 + 0.18 * np.clip((temple_y_max - y) / max(temple_y_max, 1.0), 0.0, 1.0))
                out[i, 0] = center[0] + (x - center[0]) * (1.0 - lateral)
                if y < brow_top + brow_spacing * 0.25:
                    out[i, 1] = y + brow_spacing * temple_s * 0.14

    return out


def _exclude_hair_from_mask(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    dilate_px: int | None = None,
) -> np.ndarray:
    """Subtract semantic hair pixels inside the face region (dilate hair to catch hairline wisps)."""
    try:
        import mediapipe as mp

        h, w = frame_rgb.shape[:2]
        segmenter = _get_segmenter()
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb[:, :, :3]),
        )
        result = segmenter.segment(mp_image)
        cats = _category_mask_numpy(result, h, w)
        hair = (cats == _CAT_HAIR).astype(np.uint8)
        if hair.max() <= 0:
            return mask
        px = dilate_px if dilate_px is not None else max(2, int(min(h, w) * 0.018))
        k = px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        hair = cv2.dilate(hair, kernel, iterations=1)
        return np.clip(mask * (1.0 - hair.astype(np.float32)), 0.0, 1.0)
    except Exception:
        return mask


def build_face_region_mask(
    landmarks,
    height: int,
    width: int,
    mask_edge_blur: int = 5,
    padding_percent: float = 0.0,
    *,
    forehead_trim: float = 35.0,
    face_inset: float = 8.0,
    temple_trim: float = 38.0,
    frame_rgb: np.ndarray | None = None,
    exclude_hair: bool = True,
) -> np.ndarray:
    """Filled face-oval region — upper bound for feature masks."""
    order = get_face_oval_polygon_order()
    pts = _points_for_indices(landmarks, width, height, order)
    if pts.shape[0] < 3:
        return np.zeros((height, width), dtype=np.float32)
    pts = _refine_face_oval_points(
        pts,
        landmarks,
        width,
        height,
        forehead_trim=forehead_trim,
        face_inset=face_inset,
        temple_trim=temple_trim,
    )
    mask = _fill_poly_mask(height, width, pts, mask_edge_blur)
    if padding_percent > 0:
        mask = _dilate_mask(mask, padding_percent, height, width)
    if exclude_hair and frame_rgb is not None:
        mask = _exclude_hair_from_mask(frame_rgb, mask)
    return np.clip(mask, 0.0, 1.0)


def face_region_is_valid(
    face_region: np.ndarray | None,
    *,
    min_coverage: float = 0.02,
) -> bool:
    """True when a non-trivial face oval was detected."""
    if face_region is None or face_region.max() <= 0:
        return False
    return float(face_region.mean()) >= min_coverage


def _clip_mask_to_region(mask: np.ndarray, region: np.ndarray) -> np.ndarray:
    if region is None or region.max() <= 0:
        return mask
    return np.clip(mask * np.clip(region, 0.0, 1.0), 0.0, 1.0)


# 脸部三角区：内眦为顶、嘴角+下巴为底，并 union 左右法令纹窄带
_FACE_TRIANGLE_OUTLINE = (362, 133, 291, 152, 61)

# 法令纹：嘴角锚定 + 鼻尖定左右 + 沟槽点控弧线
_NASOLABIAL_SIDE_DEFS: tuple[tuple[int, tuple[int, ...], str], ...] = (
    (61, (48, 219, 220, 98), "nasolabial_left"),
    (291, (278, 437, 326, 327), "nasolabial_right"),
)
_NASOLABIAL_FOLD_TRIM = 7


def _nasolabial_midline_x(landmarks, width: int) -> float:
    for idx in (1, 4, 2, 94, 19):
        if 0 <= idx < len(landmarks):
            return float(landmarks[idx].x * width)
    left, _, right, _ = _landmark_bbox(landmarks, width, 1, 0.0)
    return 0.5 * (left + right)


def _outer_nostril_for_mouth(
    landmarks, width: int, height: int, mouth_idx: int, candidates: tuple[int, ...]
) -> np.ndarray | None:
    if not (0 <= mouth_idx < len(landmarks)):
        return None
    mx, _ = _landmark_xy(landmarks, mouth_idx, width, height)
    mid_x = _nasolabial_midline_x(landmarks, width)
    mouth_on_left = mx < mid_x
    best: np.ndarray | None = None
    best_dist = -1.0
    for idx in candidates:
        if not (0 <= idx < len(landmarks)):
            continue
        x, y = _landmark_xy(landmarks, idx, width, height)
        side_dist = (mid_x - x) if mouth_on_left else (x - mid_x)
        if side_dist <= 0:
            continue
        if side_dist > best_dist:
            best_dist = side_dist
            best = np.array([x, y], dtype=np.float32)
    if best is not None:
        return best
    for idx in candidates:
        if not (0 <= idx < len(landmarks)):
            continue
        x, y = _landmark_xy(landmarks, idx, width, height)
        side_dist = abs((mid_x - x) if mouth_on_left else (x - mid_x))
        if side_dist > best_dist:
            best_dist = side_dist
            best = np.array([x, y], dtype=np.float32)
    return best


def _perp_dist_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1.0:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _fold_control_for_side(
    landmarks,
    width: int,
    height: int,
    nostril: np.ndarray,
    mouth: np.ndarray,
    fold_indices: list[int],
) -> np.ndarray:
    mx, my = int(mouth[0]), int(mouth[1])
    mid_x = _nasolabial_midline_x(landmarks, width)
    mouth_on_left = mx < mid_x
    y0, y1 = sorted((int(nostril[1]), my))
    face_left, _, face_right, _ = _landmark_bbox(landmarks, width, height, 0.0)
    face_w = max(40.0, float(face_right - face_left))

    best_pt: np.ndarray | None = None
    best_score = -1.0
    for idx in fold_indices[:24]:
        if not (0 <= idx < len(landmarks)):
            continue
        x, y = _landmark_xy(landmarks, idx, width, height)
        if y < y0 - 4 or y > y1 + 8:
            continue
        side = (mid_x - x) if mouth_on_left else (x - mid_x)
        if side <= face_w * 0.01:
            continue
        p = np.array([x, y], dtype=np.float32)
        score = _perp_dist_to_segment(p, nostril, mouth) + side * 0.15
        if score > best_score:
            best_score = score
            best_pt = p

    if best_pt is not None:
        return best_pt

    mid = nostril + 0.5 * (mouth - nostril)
    outward = -face_w * 0.085 if mouth_on_left else face_w * 0.085
    return mid + np.array([outward, face_w * 0.02], dtype=np.float32)


def _nasolabial_fold_endpoints(
    landmarks,
    width: int,
    height: int,
    nostril: np.ndarray,
    mouth: np.ndarray,
    mouth_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Start at outer ala; end above mouth corner (do not cover nostril hole or lips)."""
    face_left, _, face_right, _ = _landmark_bbox(landmarks, width, height, 0.0)
    face_w = max(40.0, float(face_right - face_left))
    mid_x = _nasolabial_midline_x(landmarks, width)
    mouth_on_left = float(mouth[0]) < mid_x
    cheek_sign = -1.0 if mouth_on_left else 1.0

    chord = mouth - nostril
    # 起点：贴鼻翼外侧 landmark，仅微向颊部外移（不沿沟槽下移）
    start = nostril.astype(np.float32).copy()
    start[0] += cheek_sign * face_w * 0.012

    # 终点：不到嘴角，停在沟槽上段并略上提
    end = nostril + 0.70 * chord
    end[1] -= face_w * 0.018

    upper_lip = {61: 78, 291: 308}.get(mouth_idx)
    if upper_lip is not None and 0 <= upper_lip < len(landmarks):
        ux, uy = _landmark_xy(landmarks, upper_lip, width, height)
        end[0] = 0.55 * end[0] + 0.45 * float(ux)
        end[1] = min(float(end[1]), float(uy) - face_w * 0.008)

    return start, end


def _clip_mask_cheek_side_of_nostril(
    mask: np.ndarray,
    nostril: np.ndarray,
    mouth_on_left: bool,
    *,
    margin_px: int = 2,
) -> np.ndarray:
    """Keep ribbon on cheek side of outer ala — drop medial bleed into nostril."""
    if mask.max() <= 0:
        return mask
    out = mask.copy()
    nx = int(round(float(nostril[0])))
    m = max(0, int(margin_px))
    if mouth_on_left:
        out[:, min(out.shape[1], nx + m) :] = 0.0
    else:
        out[:, : max(0, nx - m)] = 0.0
    return out


def _suppress_band_endpoint_caps(
    band: np.ndarray,
    pts: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Distance-band round caps at polyline ends → remove endpoint blobs."""
    if band.max() <= 0 or pts.shape[0] < 2:
        return band
    out = band.copy()
    r = max(2, int(round(radius)))
    for pt in (pts[0], pts[-1]):
        cx, cy = int(pt[0]), int(pt[1])
        cap = np.zeros_like(out)
        cv2.circle(cap, (cx, cy), r, 1.0, -1, lineType=cv2.LINE_AA)
        out = np.clip(out * (1.0 - cap), 0.0, 1.0)
    return out


def _nasolabial_side_curve(
    landmarks,
    width: int,
    height: int,
    mouth_idx: int,
    nostril_candidates: tuple[int, ...],
    fold_path_key: str,
    *,
    samples: int = 24,
) -> np.ndarray:
    if not (0 <= mouth_idx < len(landmarks)):
        return np.zeros((0, 2), dtype=np.int32)

    nostril = _outer_nostril_for_mouth(landmarks, width, height, mouth_idx, nostril_candidates)
    if nostril is None:
        return np.zeros((0, 2), dtype=np.int32)

    mouth = np.array(_landmark_xy(landmarks, mouth_idx, width, height), dtype=np.float32)
    start, end = _nasolabial_fold_endpoints(landmarks, width, height, nostril, mouth, mouth_idx)
    fold_indices = list(_REGION_INDEX_SETS.get(fold_path_key, ()))[: _NASOLABIAL_FOLD_TRIM * 3]
    p1 = _fold_control_for_side(landmarks, width, height, nostril, mouth, fold_indices)

    pts: list[np.ndarray] = []
    for t in np.linspace(0.0, 1.0, max(12, samples)):
        q = (1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * p1 + t**2 * end
        pts.append(q)
    return np.round(np.array(pts, dtype=np.float32)).astype(np.int32)


def _side_mask_from_curve(
    height: int,
    width: int,
    pts: np.ndarray,
    ribbon: float,
    stroke: float,
) -> np.ndarray:
    if pts.shape[0] < 2:
        return np.zeros((height, width), dtype=np.float32)
    line = _polyline_stroke_mask_tight(height, width, pts, stroke)
    band = _distance_band_from_polyline(height, width, pts, ribbon)
    band = _suppress_band_endpoint_caps(band, pts, ribbon * 0.9)
    return np.clip(line + band * 0.48, 0.0, 1.0)


def _subtract_philtrum_overlap(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    landmarks,
    width: int,
    height: int,
) -> np.ndarray:
    if left_mask.max() <= 0 and right_mask.max() <= 0:
        return np.zeros_like(left_mask)
    face_left, _, face_right, _ = _landmark_bbox(landmarks, width, height, 0.0)
    face_w = max(40.0, float(face_right - face_left))
    mid_x = _nasolabial_midline_x(landmarks, width)
    half = max(4.0, face_w * 0.03)
    x0 = max(0, int(mid_x - half))
    x1 = min(width, int(mid_x + half) + 1)
    cut = np.ones((height, width), dtype=np.float32)
    cut[:, x0:x1] = 0.0
    return np.clip(left_mask * cut + right_mask * cut, 0.0, 1.0)


def _landmark_xy_mean(
    landmarks, width: int, height: int, indices: tuple[int, ...]
) -> np.ndarray | None:
    pts: list[np.ndarray] = []
    n = len(landmarks)
    for idx in indices:
        if 0 <= idx < n:
            x, y = _landmark_xy(landmarks, idx, width, height)
            pts.append(np.array([x, y], dtype=np.float32))
    if not pts:
        return None
    return np.mean(pts, axis=0)


def _clip_mask_below_mouth(
    mask: np.ndarray,
    landmarks,
    width: int,
    height: int,
    margin_px: int = 10,
) -> np.ndarray:
    """Drop spill below each mouth corner (per side), not global min-y."""
    if mask.max() <= 0:
        return mask
    out = mask.copy()
    for mouth_idx in (61, 291):
        if not (0 <= mouth_idx < len(landmarks)):
            continue
        mx, my = _landmark_xy(landmarks, mouth_idx, width, height)
        y_max = min(height - 1, my + margin_px)
        x_half = max(12, int(width * 0.06))
        x0 = max(0, mx - x_half)
        x1 = min(width, mx + x_half + 1)
        out[y_max + 1 :, x0:x1] = 0.0
    return out


# 面颊丰盈区：小椭圆中心点（非整块凸包）
_CHEEK_PLUMP_CENTERS = (205, 425)


def _landmark_face_oval_clip(
    landmarks,
    height: int,
    width: int,
    *,
    inset_ratio: float = 0.025,
) -> np.ndarray:
    """Tight face-oval boundary (landmarks), slightly eroded — avoids semantic skin bleed."""
    idx = _REGION_INDEX_SETS["face_oval"]
    oval = _fill_polygon_mask(
        height, width, _points_for_indices(landmarks, width, height, idx), blur=0
    )
    if oval.max() <= 0:
        return oval
    left, top, right, bottom = _landmark_bbox(landmarks, width, height, 0.0)
    erode_px = max(2, int(min(right - left, bottom - top) * inset_ratio))
    k = erode_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode((oval * 255).astype(np.uint8), kernel, iterations=1)
    return eroded.astype(np.float32) / 255.0


def _distance_band_from_polyline(
    height: int,
    width: int,
    pts: np.ndarray,
    max_dist: float,
) -> np.ndarray:
    """Pixels within max_dist of the fold centerline."""
    if pts.shape[0] < 2:
        return np.zeros((height, width), dtype=np.float32)
    line = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(line, [pts.reshape(-1, 1, 2).astype(np.int32)], False, 255, 1, cv2.LINE_8)
    dist = cv2.distanceTransform(255 - line, cv2.DIST_L2, 3)
    return (dist <= max(1.5, max_dist)).astype(np.float32)


def _polyline_stroke_mask_tight(
    height: int,
    width: int,
    pts: np.ndarray,
    thickness: float,
) -> np.ndarray:
    """Thin fold ribbon without extra feather blur."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if pts.shape[0] < 2:
        return mask.astype(np.float32) / 255.0
    thick = max(1, int(round(thickness)))
    cv2.polylines(
        mask,
        [pts.reshape(-1, 1, 2).astype(np.int32)],
        False,
        255,
        thick,
        cv2.LINE_AA,
    )
    return mask.astype(np.float32) / 255.0


def _snap_nasolabial_to_crease(
    bgr: np.ndarray,
    guide: np.ndarray,
    clip: np.ndarray,
    search_px: int,
) -> np.ndarray:
    """Expand guide to cover visible dark crease pixels near the landmark path."""
    guide = np.clip(guide, 0.0, 1.0)
    clip = np.clip(clip, 0.0, 1.0)
    if guide.max() <= 0:
        return guide

    g8 = (guide * 255).astype(np.uint8)
    k = max(5, int(search_px) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    search = cv2.dilate(g8, kernel, iterations=1)
    search = cv2.bitwise_and(search, (clip * 255).astype(np.uint8))
    if search.max() == 0:
        return guide

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), 3.0)
    dark = np.clip(local - gray, 0.0, 255.0)

    vals = dark[search > 16]
    if vals.size < 12:
        return guide

    thr = float(np.percentile(vals, 55))
    crease = ((dark >= thr).astype(np.uint8) * 255)
    crease = cv2.bitwise_and(crease, search)
    crease = cv2.morphologyEx(
        crease, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(crease, connectivity=8)
    if num <= 1:
        return guide

    kept = np.zeros_like(crease)
    max_area = max(32, int(search.sum() / 255.0 * 0.45))
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) <= max_area:
            kept[labels == i] = 255

    if kept.max() == 0:
        return guide

    crease_f = kept.astype(np.float32) / 255.0
    out = np.clip(np.maximum(guide, crease_f * 0.92), 0.0, 1.0)
    return np.clip(out * clip, 0.0, 1.0)


def _refine_nasolabial_crease_mask(
    bgr: np.ndarray,
    corridor: np.ndarray,
    clip_mask: np.ndarray,
    *,
    centerline: np.ndarray | None = None,
) -> np.ndarray:
    """Keep thin dark crease pixels near fold centerline; never expand to cheek blobs."""
    corridor = np.clip(corridor, 0.0, 1.0)
    clip_mask = np.clip(clip_mask, 0.0, 1.0)
    guide = np.clip(corridor * clip_mask, 0.0, 1.0)
    if guide.max() <= 0:
        return guide

    skeleton = centerline if centerline is not None else corridor
    sk8 = (np.clip(skeleton, 0, 1) * 255).astype(np.uint8)
    near = cv2.dilate(sk8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    near_f = near.astype(np.float32) / 255.0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), 2.5)
    dark = np.clip(local - gray, 0.0, 255.0)

    g8 = (guide * 255).astype(np.uint8)
    vals = dark[(near_f > 0.5) & (g8 > 16)]
    if vals.size < 6:
        return guide

    thr = float(np.percentile(vals, 78))
    crease = ((dark >= thr).astype(np.uint8) * 255)
    crease = cv2.bitwise_and(crease, g8)
    crease = cv2.bitwise_and(crease, near)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    crease = cv2.morphologyEx(crease, cv2.MORPH_OPEN, k, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(crease, connectivity=8)
    if num <= 1:
        return guide

    kept = np.zeros_like(crease)
    max_area = max(12, min(96, int(guide.sum() / 255.0 * 0.06)))
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= max_area:
            kept[labels == i] = 255

    if kept.max() == 0:
        return guide

    crease_f = kept.astype(np.float32) / 255.0
    refined = cv2.GaussianBlur((crease_f * 255).astype(np.uint8), (3, 3), 0).astype(np.float32) / 255.0
    refined = np.clip(refined * clip_mask, 0.0, 1.0)
    if refined.mean() > max(0.004, guide.mean() * 2.5):
        return guide
    return refined


def build_cheek_plump_mask(
    landmarks,
    height: int,
    width: int,
    face_skin: np.ndarray,
    mask_edge_blur: int,
) -> np.ndarray:
    """Small cheek ellipses for plump — avoids huge convex-hull blobs in mask preview."""
    left, top, right, bottom = _landmark_bbox(landmarks, width, height, 0.0)
    fw = max(12.0, float(right - left))
    fh = max(12.0, float(bottom - top))
    oval = _landmark_face_oval_clip(landmarks, height, width, inset_ratio=0.02)
    face_skin = np.clip(face_skin, 0.0, 1.0)
    clip = np.clip(np.minimum(face_skin, oval), 0.0, 1.0)
    union = np.zeros((height, width), dtype=np.float32)
    rx = max(3, int(fw * 0.055))
    ry = max(3, int(fh * 0.048))

    for center_idx in _CHEEK_PLUMP_CENTERS:
        if center_idx >= len(landmarks):
            continue
        cx, cy = _landmark_xy(landmarks, center_idx, width, height)
        m = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        union = np.maximum(union, m.astype(np.float32) / 255.0)

    blur = int(mask_edge_blur)
    if blur > 0 and union.max() > 0:
        k = max(3, min(blur | 1, 9))
        union = cv2.GaussianBlur((union * 255).astype(np.uint8), (k, k), 0).astype(np.float32) / 255.0
    return np.clip(union * clip, 0.0, 1.0)


def _face_interior_clip_mask(
    landmarks,
    height: int,
    width: int,
) -> np.ndarray:
    """Face interior for clipping; falls back to bbox ellipse if oval landmarks collapse."""
    idx = _REGION_INDEX_SETS
    oval = _fill_polygon_mask(
        height,
        width,
        _points_for_indices(landmarks, width, height, idx["face_oval"]),
        blur=0,
    )
    if oval.mean() >= 0.03:
        return np.clip(oval, 0.0, 1.0)

    left, top, right, bottom = _landmark_bbox(landmarks, width, height, 0.04)
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    rx = max(1, (right - left) // 2)
    ry = max(1, (bottom - top) // 2)
    m = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return m.astype(np.float32) / 255.0


def build_nasolabial_fold_mask(
    landmarks,
    height: int,
    width: int,
    mask_edge_blur: int,
    face_skin: np.ndarray,
    frame_rgb: np.ndarray | None = None,
    *,
    strip_scale: float = 0.38,
) -> np.ndarray:
    """Per-side arcs: outer nostril → fold bulge → mouth corner (61 / 291)."""
    left, top, right, bottom = _landmark_bbox(landmarks, width, height, 0.0)
    face_w = max(40.0, float(right - left))
    scale = max(0.85, float(strip_scale))
    ribbon = max(5, int(face_w * 0.015 * scale))
    stroke = max(3, int(ribbon * 0.42))

    idx = _REGION_INDEX_SETS
    oval = _face_interior_clip_mask(landmarks, height, width)
    lips = _fill_polygon_mask(
        height,
        width,
        _points_for_indices(landmarks, width, height, idx["lips"]),
        blur=3,
    )
    nose = _fill_polygon_mask(
        height,
        width,
        _points_for_indices(landmarks, width, height, idx["nose"]),
        blur=2,
    )
    lip_keep = np.clip(1.0 - np.clip(lips, 0.0, 1.0) * 0.92, 0.0, 1.0)
    nose_keep = np.clip(1.0 - np.clip(nose, 0.0, 1.0) * 0.75, 0.0, 1.0)
    clip_mask = np.clip(oval * lip_keep * nose_keep, 0.0, 1.0)

    side_masks: list[np.ndarray] = []
    mid_x = _nasolabial_midline_x(landmarks, width)
    for mouth_idx, nostril_ids, fold_key in _NASOLABIAL_SIDE_DEFS:
        pts = _nasolabial_side_curve(
            landmarks, width, height, mouth_idx, nostril_ids, fold_key
        )
        if pts.shape[0] < 2:
            continue
        side = _side_mask_from_curve(height, width, pts, ribbon, stroke)
        nostril = _outer_nostril_for_mouth(landmarks, width, height, mouth_idx, nostril_ids)
        if nostril is not None:
            mx, _ = _landmark_xy(landmarks, mouth_idx, width, height)
            side = _clip_mask_cheek_side_of_nostril(side, nostril, mx < mid_x, margin_px=2)
        side_masks.append(side)

    if not side_masks:
        return np.zeros((height, width), dtype=np.float32)

    if len(side_masks) == 2:
        union = _subtract_philtrum_overlap(
            side_masks[0], side_masks[1], landmarks, width, height
        )
    else:
        union = side_masks[0]

    union = np.clip(union * clip_mask, 0.0, 1.0)
    if union.max() <= 0:
        union = np.clip(np.maximum.reduce(side_masks), 0.0, 1.0)

    union = _clip_mask_below_mouth(union, landmarks, width, height)

    blur = int(mask_edge_blur)
    if blur > 0 and union.max() > 0:
        k = max(3, min(blur | 1, 5))
        blurred = cv2.GaussianBlur((union * 255).astype(np.uint8), (k, k), 0).astype(np.float32) / 255.0
        reclipped = np.clip(blurred * clip_mask, 0.0, 1.0)
        union = reclipped if reclipped.max() > 0 else blurred
        union = _clip_mask_below_mouth(union, landmarks, width, height)

    return union


def build_face_triangle_mask(
    landmarks,
    height: int,
    width: int,
    mask_edge_blur: int = 5,
) -> np.ndarray:
    """Mid-face triangle extended to nasolabial folds (inner eyes → mouth corners → chin)."""
    blur = int(mask_edge_blur)
    union = np.zeros((height, width), dtype=np.float32)

    outline = _points_for_indices(landmarks, width, height, list(_FACE_TRIANGLE_OUTLINE))
    if outline.shape[0] >= 3:
        union = np.maximum(union, _fill_polygon_mask(height, width, outline, blur))

    for mouth_idx, nostril_ids, fold_key in _NASOLABIAL_SIDE_DEFS:
        pts = _nasolabial_side_curve(
            landmarks, width, height, mouth_idx, nostril_ids, fold_key, samples=18
        )
        if pts.shape[0] >= 2:
            spacing = _mean_point_spacing(pts)
            thick = max(3, spacing * 0.3)
            union = np.maximum(union, _polyline_stroke_mask_tight(height, width, pts, thick))

    return np.clip(union, 0.0, 1.0)


_SWAP_MASK_KWARG_KEYS = frozenset(
    {
        "include_eyebrows",
        "include_eyes",
        "include_nose",
        "include_mouth",
        "include_face_triangle",
        "mask_edge_blur",
        "brow_thickness",
        "nose_expand",
        "mouth_expand",
        "clip_to_face",
        "face_padding_percent",
        "forehead_trim",
        "face_inset",
        "temple_trim",
        "exclude_hair",
    }
)


def _swap_mask_kwargs(mask_kwargs: dict) -> dict:
    """Only pass kwargs accepted by build_swap_features_mask."""
    return {k: v for k, v in mask_kwargs.items() if k in _SWAP_MASK_KWARG_KEYS}


def build_swap_features_mask(
    landmarks,
    height: int,
    width: int,
    *,
    include_eyebrows: bool = True,
    include_eyes: bool = True,
    include_nose: bool = True,
    include_mouth: bool = True,
    include_face_triangle: bool = False,
    mask_edge_blur: int = 5,
    brow_thickness: float = 1.0,
    nose_expand: float = 1.12,
    mouth_expand: float = 1.08,
    clip_to_face: bool = True,
    face_padding_percent: float = 0.0,
    forehead_trim: float = 35.0,
    face_inset: float = 8.0,
    temple_trim: float = 38.0,
    exclude_hair: bool = True,
    frame_rgb: np.ndarray | None = None,
    face_region: np.ndarray | None = None,
) -> np.ndarray:
    """Union mask for eyebrows / eyes / nose / mouth (MediaPipe landmarks)."""
    idx = _REGION_INDEX_SETS
    union = np.zeros((height, width), dtype=np.float32)
    blur = int(mask_edge_blur)

    if include_eyes:
        union = np.maximum(
            union,
            build_full_eye_union_mask(landmarks, height, width, blur),
        )
    if include_eyebrows:
        for key in ("left_eyebrow", "right_eyebrow"):
            union = np.maximum(
                union,
                _brow_region_mask(
                    landmarks, width, height, idx[key], blur, brow_thickness
                ),
            )
    if include_nose:
        union = np.maximum(
            union,
            _nose_region_mask(landmarks, width, height, idx["nose"], blur, nose_expand),
        )
    if include_mouth:
        union = np.maximum(
            union,
            _mouth_region_mask(landmarks, width, height, idx["lips"], blur, mouth_expand),
        )
    if include_face_triangle:
        union = np.maximum(
            union,
            build_face_triangle_mask(landmarks, height, width, blur),
        )
    union = np.clip(union, 0.0, 1.0)
    if clip_to_face:
        region = face_region
        if region is None:
            region = build_face_region_mask(
                landmarks,
                height,
                width,
                blur,
                face_padding_percent,
                forehead_trim=forehead_trim,
                face_inset=face_inset,
                temple_trim=temple_trim,
                frame_rgb=frame_rgb,
                exclude_hair=exclude_hair,
            )
        union = _clip_mask_to_region(union, region)
    return union


def detect_features_mask_from_bgr(
    bgr: np.ndarray,
    *,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    **mask_kwargs,
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Detect landmarks on a BGR crop; return (feature_mask, face_region_mask, ok)."""
    min_detection_confidence = sanitize_landmark_confidence(min_detection_confidence)
    min_presence_confidence = sanitize_landmark_confidence(min_presence_confidence)
    mask_kwargs = dict(mask_kwargs)
    if "mask_edge_blur" in mask_kwargs:
        mask_kwargs["mask_edge_blur"] = sanitize_mask_edge_blur(mask_kwargs["mask_edge_blur"])
    if "brow_thickness" in mask_kwargs:
        mask_kwargs["brow_thickness"] = sanitize_brow_thickness(mask_kwargs["brow_thickness"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    landmarks = detect_face_landmarks(
        rgb, min_detection_confidence, min_presence_confidence
    )
    if landmarks is None:
        return None, None, False
    blur = int(mask_kwargs.get("mask_edge_blur", 5))
    padding = float(mask_kwargs.get("face_padding_percent", 0.0))
    forehead_trim = float(mask_kwargs.get("forehead_trim", 35.0))
    face_inset = float(mask_kwargs.get("face_inset", 8.0))
    temple_trim = float(mask_kwargs.get("temple_trim", 38.0))
    exclude_hair = bool(mask_kwargs.get("exclude_hair", True))
    face_region = build_face_region_mask(
        landmarks,
        h,
        w,
        blur,
        padding,
        forehead_trim=forehead_trim,
        face_inset=face_inset,
        temple_trim=temple_trim,
        frame_rgb=rgb,
        exclude_hair=exclude_hair,
    )
    mask = build_swap_features_mask(
        landmarks,
        h,
        w,
        face_region=face_region,
        frame_rgb=rgb,
        **_swap_mask_kwargs(mask_kwargs),
    )
    ok = face_region_is_valid(face_region)
    return mask, face_region, ok


def detect_face_region_mask_from_bgr(
    bgr: np.ndarray,
    *,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    mask_edge_blur: int = 5,
    face_padding_percent: float = 0.0,
    forehead_trim: float = 35.0,
    face_inset: float = 8.0,
    temple_trim: float = 38.0,
    exclude_hair: bool = True,
) -> tuple[np.ndarray | None, bool]:
    """Detect face oval on a full target image (for paste-back clipping)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    landmarks = detect_face_landmarks(
        rgb, min_detection_confidence, min_presence_confidence
    )
    if landmarks is None:
        return None, False
    region = build_face_region_mask(
        landmarks,
        h,
        w,
        int(mask_edge_blur),
        float(face_padding_percent),
        forehead_trim=forehead_trim,
        face_inset=face_inset,
        temple_trim=temple_trim,
        frame_rgb=rgb,
        exclude_hair=exclude_hair,
    )
    return region, face_region_is_valid(region)


def detect_face_landmarks(
    frame_rgb: np.ndarray,
    min_detection_confidence: float,
    min_presence_confidence: float,
):
    import mediapipe as mp

    landmarker = _get_landmarker(min_detection_confidence, min_presence_confidence)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(frame_rgb[:, :, :3]),
    )
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    best = None
    best_area = 0.0
    for face_lms in result.face_landmarks:
        xs = [lm.x for lm in face_lms]
        ys = [lm.y for lm in face_lms]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best = face_lms
    return best


def build_face_masks_from_landmarks(
    landmarks,
    height: int,
    width: int,
    padding_percent: float,
    mask_edge_blur: int,
    skin_mask_mode: str = "semantic",
    frame_rgb: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    blur = int(mask_edge_blur)
    idx = _REGION_INDEX_SETS
    mode = skin_mask_mode if skin_mask_mode in SKIN_MASK_MODES else "semantic"

    left_eye = _fill_polygon_mask(
        height, width, _points_for_indices(landmarks, width, height, idx["left_eye"]), blur
    )
    right_eye = _fill_polygon_mask(
        height, width, _points_for_indices(landmarks, width, height, idx["right_eye"]), blur
    )
    eyes = np.clip(left_eye + right_eye, 0.0, 1.0)
    lips = _fill_polygon_mask(
        height, width, _points_for_indices(landmarks, width, height, idx["lips"]), blur
    )

    face_skin: np.ndarray | None = None
    if mode == "semantic" and frame_rgb is not None:
        try:
            semantic = _semantic_skin_mask(frame_rgb, landmarks, padding_percent)
            if semantic is not None:
                bbox = _landmark_bbox(landmarks, width, height, padding_percent)
                bgr = image_to_bgr_uint8(frame_rgb)
                refined = _grabcut_refine_skin(bgr, semantic, bbox)
                face_skin = _joint_bilateral_smooth(refined, bgr, blur)
                face_skin = _subtract_clipped(face_skin, eyes, lips)
                face_skin = _soften_mask_edges(np.clip(face_skin, 0, 1), blur)
        except Exception as exc:
            import logging

            logging.getLogger("ComfyUI-Simple-Face-Mask").warning(
                "semantic skin mask failed, using landmarks fallback: %s", exc
            )
            face_skin = None

    if face_skin is None:
        face_skin = _landmark_oval_skin_mask(landmarks, height, width, padding_percent, blur)

    under_eye = eyes.copy()
    if blur > 0:
        k = max(3, min(blur | 1, 31))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        under_eye = cv2.dilate((under_eye * 255).astype(np.uint8), kernel, iterations=1).astype(np.float32) / 255.0
    under_eye = np.clip(under_eye * face_skin, 0.0, 1.0)

    left_cheek = build_cheek_plump_mask(landmarks, height, width, face_skin, blur)
    cheek = left_cheek

    nasolabial = build_nasolabial_fold_mask(
        landmarks,
        height,
        width,
        blur,
        face_skin,
        frame_rgb,
    )

    return {
        "face": face_skin,
        "under_eye": under_eye,
        "nasolabial": nasolabial,
        "cheek": cheek,
    }


def _fallback_center_masks(height: int, width: int, mask_edge_blur: int) -> dict[str, np.ndarray]:
    blur = int(mask_edge_blur)
    cx, cy = width // 2, int(height * 0.46)
    rx, ry = int(width * 0.2), int(height * 0.32)

    def ellipse(cx_, cy_, rx_, ry_):
        m = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(m, (cx_, cy_), (max(1, rx_), max(1, ry_)), 0, 0, 360, 255, -1)
        if blur > 0:
            k = blur if blur % 2 == 1 else blur + 1
            m = cv2.GaussianBlur(m, (max(3, k), max(3, k)), 0)
        return m.astype(np.float32) / 255.0

    face = ellipse(cx, cy, rx, ry)
    under_eye = ellipse(cx, cy - ry // 3, int(rx * 0.75), int(ry * 0.22))

    def narrow_fold(offset_x: int) -> np.ndarray:
        m = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            m,
            (cx + offset_x, cy + ry // 4),
            (max(1, int(rx * 0.22)), max(1, int(ry * 0.14))),
            -25 if offset_x < 0 else 25,
            0,
            360,
            255,
            -1,
        )
        if blur > 0:
            k = max(3, min(blur | 1, 7))
            m = cv2.GaussianBlur(m, (k, k), 0)
        return m.astype(np.float32) / 255.0

    nasolabial = np.clip(narrow_fold(-int(rx * 0.55)) + narrow_fold(int(rx * 0.55)), 0.0, 1.0)
    cheek = ellipse(cx, cy, int(rx * 0.65), int(ry * 0.4))
    under_eye = np.clip(under_eye * face, 0.0, 1.0)
    nasolabial = np.clip(nasolabial * face, 0.0, 1.0)
    cheek = np.clip(cheek * face, 0.0, 1.0)
    return {"face": face, "under_eye": under_eye, "nasolabial": nasolabial, "cheek": cheek}


def masks_from_landmarks_or_fallback(
    landmarks,
    frame_rgb_uint8: np.ndarray,
    padding_percent: float,
    mask_edge_blur: int,
    fallback_center: bool,
    skin_mask_mode: str = "semantic",
) -> dict[str, np.ndarray]:
    h, w = frame_rgb_uint8.shape[:2]
    if landmarks is None:
        if fallback_center:
            return _fallback_center_masks(h, w, mask_edge_blur)
        return {
            "face": np.zeros((h, w), dtype=np.float32),
            "under_eye": np.zeros((h, w), dtype=np.float32),
            "nasolabial": np.zeros((h, w), dtype=np.float32),
            "cheek": np.zeros((h, w), dtype=np.float32),
        }
    return build_face_masks_from_landmarks(
        landmarks,
        h,
        w,
        padding_percent,
        mask_edge_blur,
        skin_mask_mode=skin_mask_mode,
        frame_rgb=frame_rgb_uint8,
    )


def detect_masks_and_landmarks_from_rgb_frame(
    frame_rgb_uint8: np.ndarray,
    padding_percent: float,
    mask_edge_blur: int,
    min_detection_confidence: float,
    min_presence_confidence: float,
    fallback_center: bool,
    skin_mask_mode: str = "semantic",
) -> tuple[dict[str, np.ndarray], object | None]:
    landmarks = detect_face_landmarks(
        frame_rgb_uint8,
        min_detection_confidence,
        min_presence_confidence,
    )
    masks = masks_from_landmarks_or_fallback(
        landmarks,
        frame_rgb_uint8,
        padding_percent,
        mask_edge_blur,
        fallback_center,
        skin_mask_mode=skin_mask_mode,
    )
    return masks, landmarks


def detect_masks_from_rgb_frame(
    frame_rgb_uint8: np.ndarray,
    padding_percent: float,
    mask_edge_blur: int,
    min_detection_confidence: float,
    min_presence_confidence: float,
    fallback_center: bool,
    skin_mask_mode: str = "semantic",
) -> dict[str, np.ndarray]:
    masks, _ = detect_masks_and_landmarks_from_rgb_frame(
        frame_rgb_uint8,
        padding_percent,
        mask_edge_blur,
        min_detection_confidence,
        min_presence_confidence,
        fallback_center,
        skin_mask_mode=skin_mask_mode,
    )
    return masks
