"""Face slimming: landmark-driven displacement warp (true liquify, no compositing)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .face_utils import (
    _REGION_INDEX_SETS,
    _fill_polygon_mask,
    _landmark_bbox,
    _landmark_xy,
    _points_for_indices,
    detect_face_landmarks,
    image_to_bgr_uint8,
)

_LOG = logging.getLogger("ComfyUI-Simple-Face-Mask")

# Cheeks / jaw — move toward midline
_LEFT_CHEEK_IDX = [234, 227, 116, 123, 147, 187, 205, 50, 101, 36, 142, 126]
_RIGHT_CHEEK_IDX = [454, 447, 345, 352, 376, 411, 266, 330, 280, 371, 427, 356]
_LEFT_JAW_IDX = [172, 136, 150, 176, 148, 149, 152]
_RIGHT_JAW_IDX = [397, 365, 379, 400, 377, 378]
_JAW_BBOX_IDX = _LEFT_JAW_IDX + _RIGHT_JAW_IDX + [152, 377, 175, 199, 18, 200]

# Nose / midline — fixed anchors (displacement = 0)
_FIXED_IDX = [
    1, 2, 4, 5, 6, 19, 94, 168, 195, 197,  # nose
    133, 362,  # inner eye corners
    10, 151, 9, 8,  # forehead / glabella
]


def _slider_to_ratio(amount: float, naturalness: float, *, scale: float = 0.22) -> float:
    """Fraction of horizontal distance from midline to move (0–~0.18)."""
    t = float(np.clip(amount / 100.0, 0.0, 1.0))
    if t < 1e-6:
        return 0.0
    n = float(np.clip(naturalness / 100.0, 0.0, 1.0))
    subtle = 0.50 + 0.50 * (1.0 - n)
    return t * scale * subtle


def _face_center_x(landmarks, width: int) -> float:
    xs = []
    for idx in (1, 4, 5, 195, 197, 168):
        if idx < len(landmarks):
            xs.append(landmarks[idx].x * width)
    if not xs:
        xs = [lm.x * width for lm in landmarks]
    return float(np.mean(xs))


def _collect_control_points(
    landmarks,
    width: int,
    height: int,
    center_x: float,
    cheek_ratio: float,
    jaw_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Landmark positions (K,2) and displacement vectors (K,2) in pixels."""
    pts: list[tuple[float, float]] = []
    disp: list[tuple[float, float]] = []

    def add_moving(indices: list[int], ratio: float) -> None:
        if ratio < 1e-6:
            return
        n = len(landmarks)
        for idx in indices:
            if idx >= n:
                continue
            lx, ly = _landmark_xy(landmarks, idx, width, height)
            delta_x = -(lx - center_x) * ratio
            pts.append((float(lx), float(ly)))
            disp.append((delta_x, 0.0))

    def add_fixed(indices: list[int]) -> None:
        n = len(landmarks)
        for idx in indices:
            if idx >= n:
                continue
            lx, ly = _landmark_xy(landmarks, idx, width, height)
            pts.append((float(lx), float(ly)))
            disp.append((0.0, 0.0))

    add_moving(_LEFT_CHEEK_IDX + _RIGHT_CHEEK_IDX, cheek_ratio)
    add_moving(_LEFT_JAW_IDX + _RIGHT_JAW_IDX, jaw_ratio)
    add_fixed(_FIXED_IDX)

    if len(pts) < 4:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    return np.array(pts, dtype=np.float32), np.array(disp, dtype=np.float32)


def _bbox_from_indices(
    landmarks,
    width: int,
    height: int,
    indices: list[int],
    *,
    margin_ratio: float,
) -> tuple[int, int, int, int] | None:
    pts = _points_for_indices(landmarks, width, height, indices)
    if len(pts) < 2:
        return None
    left = int(pts[:, 0].min())
    right = int(pts[:, 0].max())
    top = int(pts[:, 1].min())
    bottom = int(pts[:, 1].max())
    fw = max(float(right - left), 24.0)
    fh = max(float(bottom - top), 24.0)
    margin = int(max(fw, fh) * margin_ratio)
    x0 = max(0, left - margin)
    y0 = max(0, top - margin)
    x1 = min(width, right + margin + 1)
    y1 = min(height, bottom + margin + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _slim_roi_bbox(
    landmarks,
    width: int,
    height: int,
    *,
    cheek_ratio: float,
    jaw_ratio: float,
) -> tuple[int, int, int, int]:
    jaw_only = cheek_ratio < 1e-5 and jaw_ratio > 1e-5
    if jaw_only:
        jaw_bbox = _bbox_from_indices(
            landmarks,
            width,
            height,
            _JAW_BBOX_IDX,
            margin_ratio=0.22,
        )
        if jaw_bbox is not None:
            return jaw_bbox

    left, top, right, bottom = _landmark_bbox(landmarks, width, height, 0.12)
    fw = max(float(right - left), 24.0)
    fh = max(float(bottom - top), 24.0)
    margin = int(max(fw, fh) * 0.25)
    x0 = max(0, left - margin)
    y0 = max(0, top - margin)
    x1 = min(width, right + margin + 1)
    y1 = min(height, bottom + margin + 1)
    return x0, y0, x1, y1


def _warp_region_mask(
    landmarks,
    height: int,
    width: int,
    padding_percent: float,
) -> np.ndarray:
    """Generous face region: warp is full-strength inside, fades only into background."""
    oval = _fill_polygon_mask(
        height,
        width,
        _points_for_indices(landmarks, width, height, _REGION_INDEX_SETS["face_oval"]),
        0,
    )
    left, top, right, bottom = _landmark_bbox(landmarks, width, height, padding_percent)
    fh = max(12.0, float(bottom - top))
    k = max(7, int(fh * 0.14) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m8 = (np.clip(oval, 0, 1) * 255).astype(np.uint8)
    m8 = cv2.dilate(m8, kernel, iterations=2)
    return (m8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _jaw_warp_region_mask(landmarks, height: int, width: int) -> np.ndarray:
    """Lower-face mask for jaw-only warp (smaller ROI than full face oval)."""
    pts = _points_for_indices(
        landmarks,
        width,
        height,
        _LEFT_JAW_IDX + _RIGHT_JAW_IDX + [152, 377, 175, 199],
    )
    m = _fill_polygon_mask(height, width, pts, 0)
    k = max(5, int(min(height, width) * 0.02) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    m8 = cv2.dilate(m8, kernel, iterations=2)
    return (m8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _background_fade(mask: np.ndarray, fade_px: float) -> np.ndarray:
    """
    1.0 inside mask, smooth ramp to 0 outside (background untouched).
    Crucially: mask edge is INSIDE the face, so cheek outline also warps.
    """
    m8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    dist_out = cv2.distanceTransform(255 - m8, cv2.DIST_L2, 5).astype(np.float32)
    fade = np.clip(1.0 - dist_out / max(fade_px, 1.0), 0.0, 1.0)
    return fade.astype(np.float32)


def _displacement_field_roi(
    lm_pts: np.ndarray,
    lm_disp: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    sigma: float,
    fade_roi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian landmark interpolation → per-pixel backward displacement inside ROI."""
    rh, rw = y1 - y0, x1 - x0
    if rh <= 0 or rw <= 0 or lm_pts.shape[0] == 0:
        return np.zeros((rh, rw), np.float32), np.zeros((rh, rw), np.float32)

    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
    global_x = xx + float(x0)
    global_y = yy + float(y0)
    lm_x = lm_pts[:, 0]
    lm_y = lm_pts[:, 1]
    lm_dx = lm_disp[:, 0]
    lm_dy = lm_disp[:, 1]

    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    diff_x = global_x[:, :, np.newaxis] - lm_x[np.newaxis, np.newaxis, :]
    diff_y = global_y[:, :, np.newaxis] - lm_y[np.newaxis, np.newaxis, :]
    w_k = np.exp(-(diff_x * diff_x + diff_y * diff_y) * inv_2s2)
    w_sum = np.maximum(w_k.sum(axis=2), 1e-8)
    disp_x_roi = (w_k * lm_dx).sum(axis=2) / w_sum
    disp_y_roi = (w_k * lm_dy).sum(axis=2) / w_sum

    disp_x_roi *= fade_roi
    disp_y_roi *= fade_roi
    return disp_x_roi.astype(np.float32), disp_y_roi.astype(np.float32)


def _remap_roi(
    bgr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    disp_x_roi: np.ndarray,
    disp_y_roi: np.ndarray,
) -> np.ndarray:
    """Remap only the face/jaw ROI and paste back — avoids full-image cv2.remap."""
    rh, rw = disp_x_roi.shape
    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
    map_x = (xx + float(x0) - disp_x_roi).astype(np.float32)
    map_y = (yy + float(y0) - disp_y_roi).astype(np.float32)
    out = bgr.copy()
    out[y0:y1, x0:x1] = cv2.remap(
        bgr,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return out


def build_face_slim_mask(
    landmarks,
    height: int,
    width: int,
    mask_edge_blur: int = 11,
) -> np.ndarray:
    left = _fill_polygon_mask(
        height,
        width,
        _points_for_indices(landmarks, width, height, _REGION_INDEX_SETS["left_cheek"]),
        mask_edge_blur,
    )
    right = _fill_polygon_mask(
        height,
        width,
        _points_for_indices(landmarks, width, height, _REGION_INDEX_SETS["right_cheek"]),
        mask_edge_blur,
    )
    union = np.clip(left + right, 0.0, 1.0)
    if mask_edge_blur > 0 and union.max() > 0:
        k = max(3, min(mask_edge_blur | 1, 31))
        union = cv2.GaussianBlur(union, (k, k), 0)
    return union


def build_jaw_slim_mask(
    landmarks,
    height: int,
    width: int,
    mask_edge_blur: int = 11,
) -> np.ndarray:
    left = _fill_polygon_mask(
        height,
        width,
        _points_for_indices(landmarks, width, height, _LEFT_JAW_IDX),
        mask_edge_blur,
    )
    right = _fill_polygon_mask(
        height,
        width,
        _points_for_indices(landmarks, width, height, _RIGHT_JAW_IDX),
        mask_edge_blur,
    )
    union = np.clip(left + right, 0.0, 1.0)
    if mask_edge_blur > 0 and union.max() > 0:
        k = max(3, min(mask_edge_blur | 1, 31))
        union = cv2.GaussianBlur(union, (k, k), 0)
    return union


def build_slim_preview_mask(
    landmarks,
    height: int,
    width: int,
    mask_edge_blur: int,
    *,
    cheek_ratio: float,
    jaw_ratio: float,
) -> np.ndarray:
    jaw_only = cheek_ratio < 1e-5 and jaw_ratio > 1e-5
    cheek_only = jaw_ratio < 1e-5 and cheek_ratio > 1e-5
    if jaw_only:
        return build_jaw_slim_mask(landmarks, height, width, mask_edge_blur)
    if cheek_only:
        return build_face_slim_mask(landmarks, height, width, mask_edge_blur)
    cheek = build_face_slim_mask(landmarks, height, width, mask_edge_blur)
    jaw = build_jaw_slim_mask(landmarks, height, width, mask_edge_blur)
    return np.clip(cheek + jaw, 0.0, 1.0)


def apply_face_slim(
    bgr: np.ndarray,
    landmarks,
    *,
    face_slim: float,
    jaw_slim: float = 0.0,
    naturalness: float = 80.0,
    mask_edge_blur: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    center_x = _face_center_x(landmarks, w)
    cheek_ratio = _slider_to_ratio(face_slim, naturalness, scale=0.22)
    jaw_ratio = _slider_to_ratio(jaw_slim, naturalness, scale=0.18)
    jaw_only = cheek_ratio < 1e-5 and jaw_ratio > 1e-5
    slim_mask = build_slim_preview_mask(
        landmarks,
        h,
        w,
        mask_edge_blur,
        cheek_ratio=cheek_ratio,
        jaw_ratio=jaw_ratio,
    )

    if cheek_ratio < 1e-5 and jaw_ratio < 1e-5:
        return bgr, slim_mask

    lm_pts, lm_disp = _collect_control_points(
        landmarks, w, h, center_x, cheek_ratio, jaw_ratio
    )
    if lm_pts.shape[0] < 4:
        _LOG.warning("Face slim: not enough control points.")
        return bgr, slim_mask

    x0, y0, x1, y1 = _slim_roi_bbox(
        landmarks, w, h, cheek_ratio=cheek_ratio, jaw_ratio=jaw_ratio
    )
    roi_w = max(float(x1 - x0), 24.0)
    roi_h = max(float(y1 - y0), 24.0)

    if jaw_only:
        warp_mask = _jaw_warp_region_mask(landmarks, h, w)
        sigma = max(roi_w * 0.14, 14.0)
        fade_px = max(10.0, roi_w * 0.10)
    else:
        warp_mask = _warp_region_mask(landmarks, h, w, 0.10)
        sigma = max(roi_w * 0.16, 18.0)
        fade_px = max(12.0, roi_w * 0.08)

    fade_roi = _background_fade(warp_mask[y0:y1, x0:x1], fade_px)
    disp_x_roi, disp_y_roi = _displacement_field_roi(
        lm_pts, lm_disp, x0, y0, x1, y1, sigma, fade_roi
    )

    if float(np.abs(disp_x_roi).max()) < 0.5:
        _LOG.warning("Face slim: displacement too small.")
        return bgr, slim_mask

    out = _remap_roi(bgr, x0, y0, x1, y1, disp_x_roi, disp_y_roi)

    _LOG.info(
        "Face slim v4 roi-warp: cheek=%.3f jaw=%.3f max_disp=%.1fpx roi=%dx%d jaw_only=%s",
        cheek_ratio,
        jaw_ratio,
        float(np.abs(disp_x_roi).max()),
        x1 - x0,
        y1 - y0,
        jaw_only,
    )
    return out, slim_mask


def apply_face_slim_from_rgb(
    frame_rgb_uint8: np.ndarray,
    face_slim: float,
    jaw_slim: float,
    min_detection_confidence: float,
    min_presence_confidence: float,
    mask_edge_blur: int = 11,
    naturalness: float = 80.0,
    detect_rgb_uint8: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    detect_rgb = detect_rgb_uint8 if detect_rgb_uint8 is not None else frame_rgb_uint8
    landmarks = detect_face_landmarks(
        detect_rgb,
        min_detection_confidence,
        min_presence_confidence,
    )
    bgr = image_to_bgr_uint8(frame_rgb_uint8)
    h, w = bgr.shape[:2]

    if landmarks is None:
        _LOG.warning(
            "Face slim: no face landmarks — set min_detection_confidence to 0.3–0.5 "
            "and connect detect_image to the original photo."
        )
        return bgr, np.zeros((h, w), dtype=np.float32), False

    cheek_ratio = _slider_to_ratio(face_slim, naturalness, scale=0.22)
    jaw_ratio = _slider_to_ratio(jaw_slim, naturalness, scale=0.18)
    if cheek_ratio < 1e-6 and jaw_ratio < 1e-6:
        mask = build_slim_preview_mask(
            landmarks,
            h,
            w,
            mask_edge_blur,
            cheek_ratio=cheek_ratio,
            jaw_ratio=jaw_ratio,
        )
        return bgr, mask, True

    out, slim_mask = apply_face_slim(
        bgr,
        landmarks,
        face_slim=face_slim,
        jaw_slim=jaw_slim,
        naturalness=naturalness,
        mask_edge_blur=mask_edge_blur,
    )
    return out, slim_mask, True
