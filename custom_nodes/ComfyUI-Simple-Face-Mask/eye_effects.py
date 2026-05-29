"""Natural eye enlargement: iris bulge + optional soft scale; naturalness controls subtlety."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .face_utils import (
    _landmark_xy,
    _points_for_indices,
    build_full_eye_union_mask,
    detect_face_landmarks,
    get_eye_region_index_sets,
    image_to_bgr_uint8,
)

_LOG = logging.getLogger("ComfyUI-Simple-Face-Mask")

_IRIS_LEFT_CENTER = 468
_IRIS_RIGHT_CENTER = 473


def _bulge_strength(eye_size: float, naturalness: float) -> float:
    """
    eye_size -100..100, naturalness 0..100 (higher = more subtle).
    @ eye=100, naturalness=0  → ~0.42 bulge
    @ eye=100, naturalness=80 → ~0.14 bulge
    """
    t = float(np.clip(eye_size / 100.0, -1.0, 1.0))
    if abs(t) < 1e-6:
        return 0.0
    n = float(np.clip(naturalness / 100.0, 0.0, 1.0))
    subtle = 0.30 + 0.70 * (1.0 - n)
    return t * 0.42 * subtle


def _affine_scale(eye_size: float, naturalness: float) -> float:
    """Extra whole-eye scale when naturalness is low; invisible when naturalness high."""
    t = float(np.clip(eye_size / 100.0, -1.0, 1.0))
    if abs(t) < 1e-6:
        return 1.0
    n = float(np.clip(naturalness / 100.0, 0.0, 1.0))
    # fade out affine above naturalness 70
    boost = max(0.0, 1.0 - n / 0.70)
    return 1.0 + t * 0.14 * boost


def _radius_scale(naturalness: float) -> float:
    n = float(np.clip(naturalness / 100.0, 0.0, 1.0))
    return 0.88 + 0.22 * (1.0 - n)


def _eye_metrics(
    landmarks,
    width: int,
    height: int,
    contour_indices: list[int],
    iris_index: int | None,
    *,
    naturalness: float = 80.0,
) -> tuple[float, float, float] | None:
    n_lm = len(landmarks)
    pts = _points_for_indices(landmarks, width, height, contour_indices)
    if pts.shape[0] < 3:
        return None

    if iris_index is not None and 0 <= iris_index < n_lm:
        cx, cy = _landmark_xy(landmarks, iris_index, width, height)
        cx, cy = float(cx), float(cy)
    else:
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())

    rx = float(pts[:, 0].max() - pts[:, 0].min()) * 0.5
    ry = float(pts[:, 1].max() - pts[:, 1].min()) * 0.5
    radius = max(12.0, max(rx, ry) * _radius_scale(naturalness))
    return cx, cy, radius


def _blend_falloff(
    height: int,
    width: int,
    cx: float,
    cy: float,
    radius: float,
    feather: float,
) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rad = max(radius * 1.15, 1.0)
    t = np.clip(dist / rad, 0.0, 1.0)
    core = (1.0 - t) ** 1.4
    feather_px = max(6.0, feather * rad * 0.35)
    edge_t = np.clip((dist - rad) / feather_px, 0.0, 1.0)
    edge = 1.0 - edge_t * edge_t * (3.0 - 2.0 * edge_t)
    m = np.where(dist <= rad, core, core * edge)
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def _bulge_warp_roi(
    bgr: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    strength: float,
) -> np.ndarray:
    if abs(strength) < 1e-5:
        return bgr

    h, w = bgr.shape[:2]
    r = int(np.ceil(radius * 1.45))
    x0 = max(0, int(cx) - r)
    x1 = min(w, int(cx) + r + 1)
    y0 = max(0, int(cy) - r)
    y1 = min(h, int(cy) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return bgr

    roi = bgr[y0:y1, x0:x1].astype(np.float32)
    rh, rw = roi.shape[:2]
    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
    gx = xx + x0
    gy = yy + y0
    dx = gx - cx
    dy = gy - cy
    dist = np.sqrt(dx * dx + dy * dy)
    rad = max(radius, 1.0)
    t = np.clip(dist / rad, 0.0, 1.0)
    falloff = (1.0 - t * t) ** 1.5
    factor = 1.0 + strength * falloff
    factor = np.maximum(factor, 1e-3)
    map_x = (cx + dx / factor - x0).astype(np.float32)
    map_y = (cy + dy / factor - y0).astype(np.float32)
    warped = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    out = bgr.astype(np.float32)
    out[y0:y1, x0:x1] = warped
    return np.clip(out, 0, 255).astype(np.uint8)


def _affine_warp_around(bgr: np.ndarray, cx: float, cy: float, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-4:
        return bgr
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((cx, cy), 0.0, scale)
    return cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _affine_open_eye(
    bgr: np.ndarray,
    cx: float,
    cy: float,
    *,
    eye_open: float,
    base_scale: float = 1.0,
) -> np.ndarray:
    """Commercial apps elongate eyes vertically more than horizontally."""
    o = float(np.clip(eye_open / 100.0, 0.0, 1.0))
    if o <= 0 and abs(base_scale - 1.0) < 1e-4:
        return bgr
    sx = base_scale * (1.0 + o * 0.05)
    sy = base_scale * (1.0 + o * 0.14)
    h, w = bgr.shape[:2]
    m = np.float32([[sx, 0, cx * (1.0 - sx)], [0, sy, cy * (1.0 - sy)]])
    return cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _blend_with_falloff(
    base: np.ndarray,
    effect: np.ndarray,
    falloff: np.ndarray,
    mix: float,
) -> np.ndarray:
    m = np.clip(falloff * mix, 0.0, 1.0)[:, :, np.newaxis]
    out = base.astype(np.float32) * (1.0 - m) + effect.astype(np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)


def build_eyes_union_mask(
    landmarks,
    height: int,
    width: int,
    mask_edge_blur: int = 9,
    naturalness: float = 80.0,
) -> np.ndarray:
    """Periocular mask: opening + eyelid skin + lash bands (landmark topology)."""
    del naturalness
    return build_full_eye_union_mask(landmarks, height, width, mask_edge_blur)


def apply_eye_size(
    bgr: np.ndarray,
    landmarks,
    *,
    amount: float,
    naturalness: float = 80.0,
    mask_edge_blur: int = 9,
    eye_open: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    strength = _bulge_strength(amount, naturalness)
    aff_scale = _affine_scale(amount, naturalness)
    h, w = bgr.shape[:2]
    n_val = float(np.clip(naturalness, 0, 100))
    feather = 5.0 + (100.0 - n_val) * 0.08
    eye_union = build_eyes_union_mask(landmarks, h, w, mask_edge_blur, naturalness)

    if abs(strength) < 1e-5 and abs(aff_scale - 1.0) < 1e-4 and eye_open <= 0:
        return bgr, eye_union

    mix = 0.72 + 0.28 * (1.0 - n_val / 100.0)
    eye_sets = get_eye_region_index_sets()
    n_lm = len(landmarks)
    out = bgr.copy()

    for label, indices, iris_idx in (
        ("left", eye_sets["left_eye"], _IRIS_LEFT_CENTER if n_lm >= 474 else None),
        ("right", eye_sets["right_eye"], _IRIS_RIGHT_CENTER if n_lm >= 474 else None),
    ):
        m = _eye_metrics(landmarks, w, h, indices, iris_idx, naturalness=naturalness)
        if m is None:
            continue
        cx, cy, radius = m
        falloff = _blend_falloff(h, w, cx, cy, radius, feather)

        warped = _bulge_warp_roi(out, cx, cy, radius, strength)
        if eye_open > 0 or abs(aff_scale - 1.0) >= 1e-4:
            warped = _affine_open_eye(warped, cx, cy, eye_open=eye_open, base_scale=aff_scale)

        out = _blend_with_falloff(out, warped, falloff, mix)
        _LOG.debug(
            "Eye %s: (%.0f,%.0f) r=%.0f bulge=%.3f aff=%.3f mix=%.2f",
            label,
            cx,
            cy,
            radius,
            strength,
            aff_scale,
            mix,
        )

    return out, eye_union


def apply_eye_size_from_rgb(
    frame_rgb_uint8: np.ndarray,
    amount: float,
    min_detection_confidence: float,
    min_presence_confidence: float,
    mask_edge_blur: int = 9,
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
            "Eye size: no face landmarks — lower min_detection_confidence (e.g. 0.3) "
            "or connect detect_image to the original photo."
        )
        return bgr, np.zeros((h, w), dtype=np.float32), False

    if abs(amount) < 1e-6:
        return bgr, build_eyes_union_mask(landmarks, h, w, mask_edge_blur, naturalness), True

    out, eye_mask = apply_eye_size(
        bgr,
        landmarks,
        amount=amount,
        naturalness=naturalness,
        mask_edge_blur=mask_edge_blur,
    )
    return out, eye_mask, True
