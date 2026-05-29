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

_LANDMARKER = None
_SEGMENTER = None

SKIN_MASK_MODES = ("semantic", "landmarks")


def _indices_from_connections(connections) -> list[int]:
    idx: set[int] = set()
    for a, b in connections:
        idx.add(int(a))
        idx.add(int(b))
    return sorted(idx)


def _load_region_index_sets() -> dict[str, list[int]]:
    try:
        from mediapipe.python.solutions import face_mesh_connections as fmc

        return {
            "face_oval": _indices_from_connections(fmc.FACEMESH_FACE_OVAL),
            "left_eye": _indices_from_connections(fmc.FACEMESH_LEFT_EYE),
            "right_eye": _indices_from_connections(fmc.FACEMESH_RIGHT_EYE),
            "lips": _indices_from_connections(fmc.FACEMESH_LIPS),
            "left_cheek": [50, 101, 36, 205, 187, 123, 116, 147, 213, 192, 214, 204, 203, 142, 126],
            "right_cheek": [280, 330, 371, 266, 411, 352, 345, 376, 433, 416, 434, 432, 427],
            "nasolabial_left": [266, 426, 436, 416, 352, 347, 330, 423, 391, 322, 410],
            "nasolabial_right": [36, 206, 216, 192, 147, 123, 117, 118, 101, 205, 187],
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
            "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
            "left_cheek": [50, 101, 36, 205, 187, 123, 116, 147, 213, 192, 214, 204, 203, 142, 126],
            "right_cheek": [280, 330, 371, 266, 411, 352, 345, 376, 433, 416, 434, 432, 427],
            "nasolabial_left": [266, 426, 436, 416, 352, 347, 330, 423, 391, 322, 410],
            "nasolabial_right": [36, 206, 216, 192, 147, 123, 117, 118, 101, 205, 187],
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
    global _LANDMARKER
    if _LANDMARKER is not None:
        return _LANDMARKER

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_landmarker_model())),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=3,
        min_face_detection_confidence=float(min_detection_confidence),
        min_face_presence_confidence=float(min_presence_confidence),
        min_tracking_confidence=float(min_presence_confidence),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    _LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    return _LANDMARKER


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

    left_cheek = _fill_polygon_mask(
        height, width, _points_for_indices(landmarks, width, height, idx["left_cheek"]), blur
    )
    right_cheek = _fill_polygon_mask(
        height, width, _points_for_indices(landmarks, width, height, idx["right_cheek"]), blur
    )
    cheek = np.clip(left_cheek + right_cheek, 0.0, 1.0)
    cheek = np.clip(cheek * face_skin, 0.0, 1.0)

    naso_l = _fill_polygon_mask(
        height, width, _points_for_indices(landmarks, width, height, idx["nasolabial_left"]), blur
    )
    naso_r = _fill_polygon_mask(
        height, width, _points_for_indices(landmarks, width, height, idx["nasolabial_right"]), blur
    )
    nasolabial = np.clip(naso_l + naso_r, 0.0, 1.0)
    nasolabial = np.clip(nasolabial * face_skin, 0.0, 1.0)

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
    nasolabial = ellipse(cx, cy + ry // 2, int(rx * 0.55), int(ry * 0.22))
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
