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
            # Move toward vertical midline: new_x = center + (lx-center)*(1-ratio)
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


def _background_fade(mask: np.ndarray, fade_px: float) -> np.ndarray:
    """
    1.0 inside mask, smooth ramp to 0 outside (background untouched).
    Crucially: mask edge is INSIDE the face, so cheek outline also warps.
    """
    m8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    dist_out = cv2.distanceTransform(255 - m8, cv2.DIST_L2, 5).astype(np.float32)
    fade = np.clip(1.0 - dist_out / max(fade_px, 1.0), 0.0, 1.0)
    return fade.astype(np.float32)


def _displacement_field(
    height: int,
    width: int,
    lm_pts: np.ndarray,
    lm_disp: np.ndarray,
    bbox: tuple[int, int, int, int],
    sigma: float,
    fade: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian landmark interpolation → per-pixel backward displacement."""
    x0, y0, x1, y1 = bbox
    rh, rw = y1 - y0, x1 - x0
    if rh <= 0 or rw <= 0 or lm_pts.shape[0] == 0:
        z = np.zeros((height, width), np.float32)
        return z, z

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    lm_x = lm_pts[:, 0]
    lm_y = lm_pts[:, 1]
    lm_dx = lm_disp[:, 0]
    lm_dy = lm_disp[:, 1]

    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    diff_x = xx[:, :, np.newaxis] - lm_x[np.newaxis, np.newaxis, :]
    diff_y = yy[:, :, np.newaxis] - lm_y[np.newaxis, np.newaxis, :]
    w_k = np.exp(-(diff_x * diff_x + diff_y * diff_y) * inv_2s2)
    w_sum = np.maximum(w_k.sum(axis=2), 1e-8)
    disp_x_roi = (w_k * lm_dx).sum(axis=2) / w_sum
    disp_y_roi = (w_k * lm_dy).sum(axis=2) / w_sum

    fade_roi = fade[y0:y1, x0:x1]
    disp_x_roi *= fade_roi
    disp_y_roi *= fade_roi

    disp_x = np.zeros((height, width), np.float32)
    disp_y = np.zeros((height, width), np.float32)
    disp_x[y0:y1, x0:x1] = disp_x_roi
    disp_y[y0:y1, x0:x1] = disp_y_roi
    return disp_x, disp_y


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
    slim_mask = build_face_slim_mask(landmarks, h, w, mask_edge_blur)

    if cheek_ratio < 1e-5 and jaw_ratio < 1e-5:
        return bgr, slim_mask

    lm_pts, lm_disp = _collect_control_points(
        landmarks, w, h, center_x, cheek_ratio, jaw_ratio
    )
    if lm_pts.shape[0] < 4:
        _LOG.warning("Face slim: not enough control points.")
        return bgr, slim_mask

    left, top, right, bottom = _landmark_bbox(landmarks, w, h, 0.12)
    fw = max(float(right - left), 24.0)
    fh = max(float(bottom - top), 24.0)
    margin = int(max(fw, fh) * 0.25)
    x0 = max(0, left - margin)
    y0 = max(0, top - margin)
    x1 = min(w, right + margin + 1)
    y1 = min(h, bottom + margin + 1)

    warp_mask = _warp_region_mask(landmarks, h, w, 0.10)
    fade_px = max(12.0, fw * 0.08)
    fade = _background_fade(warp_mask, fade_px)

    sigma = max(fw * 0.16, 18.0)
    disp_x, disp_y = _displacement_field(
        h, w, lm_pts, lm_disp, (x0, y0, x1, y1), sigma, fade
    )

    if float(np.abs(disp_x).max()) < 0.5:
        _LOG.warning("Face slim: displacement too small.")
        return bgr, slim_mask

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (xx - disp_x).astype(np.float32)
    map_y = (yy - disp_y).astype(np.float32)

    out = cv2.remap(
        bgr,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    _LOG.info(
        "Face slim v3 landmark-warp: cheek=%.3f jaw=%.3f max_disp=%.1fpx center_x=%.0f",
        cheek_ratio,
        jaw_ratio,
        float(np.abs(disp_x).max()),
        center_x,
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

    if abs(face_slim) < 1e-6 and abs(jaw_slim) < 1e-6:
        mask = build_face_slim_mask(landmarks, h, w, mask_edge_blur)
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
