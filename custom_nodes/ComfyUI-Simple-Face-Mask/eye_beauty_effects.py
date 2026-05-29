"""Commercial-style eye beauty: geometry + brighten + sclera + catchlight + lash + aegyo + sharpen."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .eye_effects import (
    _IRIS_LEFT_CENTER,
    _IRIS_RIGHT_CENTER,
    _eye_metrics,
    apply_eye_size,
    build_eyes_union_mask,
)
from .face_utils import (
    _fill_polygon_mask,
    _points_for_indices,
    detect_face_landmarks,
    get_eye_region_index_sets,
    image_to_bgr_uint8,
)
from .skin_effects import apply_dark_circle, blend_mask, strength_01

_LOG = logging.getLogger("ComfyUI-Simple-Face-Mask")

# Upper eyelid ridge (Face Landmarker topology)
_UPPER_LID = {
    "left_eye": [466, 388, 387, 386, 385, 384, 398],
    "right_eye": [246, 161, 160, 159, 158, 157, 173],
}


@dataclass
class CommercialEyeParams:
    intensity: float = 120.0
    eye_size: float = 68.0
    eye_open: float = 55.0
    naturalness: float = 50.0
    bright_eyes: float = 72.0
    whiten_sclera: float = 55.0
    catchlight: float = 55.0
    lash_line: float = 58.0
    aegyo_sal: float = 30.0
    under_eye: float = 60.0
    sharpen: float = 45.0
    mask_edge_blur: int = 9


def _scaled_amount(amount: float, intensity: float) -> float:
    return float(np.clip(amount * intensity / 100.0, 0.0, 100.0))


def _iris_mask(h: int, w: int, cx: float, cy: float, radius: float, frac: float = 0.52) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), max(3, int(radius * frac)), 255, -1)
    k = max(3, int(radius * 0.15) | 1)
    k = min(k, 15)
    m = cv2.GaussianBlur(m, (k, k), 0)
    return m.astype(np.float32) / 255.0


def _eye_polygon_mask(
    landmarks,
    h: int,
    w: int,
    indices: list[int],
    blur: int,
) -> np.ndarray:
    pts = _points_for_indices(landmarks, w, h, indices)
    if pts.shape[0] < 3:
        return np.zeros((h, w), dtype=np.float32)
    return _fill_polygon_mask(h, w, pts, blur)


def _under_eye_mask(h: int, w: int, cx: float, cy: float, radius: float) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        m,
        (int(round(cx)), int(round(cy + radius * 0.72))),
        (max(8, int(radius * 1.05)), max(5, int(radius * 0.48))),
        0,
        0,
        360,
        255,
        -1,
    )
    k = max(5, int(radius * 0.35) | 1)
    k = min(k, 21)
    m = cv2.GaussianBlur(m, (k, k), 0)
    return m.astype(np.float32) / 255.0


def _aegyo_mask(h: int, w: int, cx: float, cy: float, radius: float) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        m,
        (int(round(cx)), int(round(cy + radius * 0.52))),
        (max(6, int(radius * 0.72)), max(4, int(radius * 0.28))),
        0,
        0,
        360,
        255,
        -1,
    )
    k = max(3, int(radius * 0.22) | 1)
    m = cv2.GaussianBlur(m, (k, k), 0)
    return m.astype(np.float32) / 255.0


def _lash_mask(landmarks, h: int, w: int, side: str, radius: float) -> np.ndarray:
    idx = _UPPER_LID.get(side, [])
    pts = _points_for_indices(landmarks, w, h, idx)
    if pts.shape[0] < 2:
        return np.zeros((h, w), dtype=np.float32)
    m = np.zeros((h, w), dtype=np.uint8)
    thick = max(1, int(radius * 0.09))
    cv2.polylines(m, [pts.reshape(-1, 1, 2)], False, 255, thick, cv2.LINE_AA)
    k = max(3, thick | 1)
    m = cv2.GaussianBlur(m, (k, k), 0)
    return m.astype(np.float32) / 255.0


def _catchlight_mask(
    h: int,
    w: int,
    cx: float,
    cy: float,
    radius: float,
    *,
    image_side: str,
) -> np.ndarray:
    """Single soft highlight on iris (not upper/lower lid blocks)."""
    m = np.zeros((h, w), dtype=np.uint8)
    ox = -0.10 if image_side == "left" else 0.10
    oy = -0.08
    pt = (int(round(cx + radius * ox)), int(round(cy + radius * oy)))
    r = max(2, int(radius * 0.14))
    cv2.circle(m, pt, r, 255, -1)
    k = max(7, int(radius * 0.32) | 1)
    k = min(k, 21)
    m = cv2.GaussianBlur(m, (k, k), 0)
    return m.astype(np.float32) / 255.0


def _apply_catchlight(bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Soft specular on iris only — avoids hard white blocks on lids."""
    s = strength_01(amount) * 0.55
    if s <= 0 or mask.max() <= 0:
        return bgr
    base = bgr.astype(np.float32)
    m3 = np.clip(mask * s, 0.0, 1.0)[:, :, np.newaxis]
    # Soft-light style, not +115 additive
    lift = np.clip(base + 42.0 * s, 0, 255)
    out = base * (1.0 - m3) + lift * m3
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_iris_brighten(
    bgr: np.ndarray,
    iris_m: np.ndarray,
    amount: float,
) -> np.ndarray:
    s = strength_01(amount)
    if s <= 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + s * 38.0, 0, 255)
    lab[:, :, 1] = lab[:, :, 1] + (128.0 - lab[:, :, 1]) * (s * 0.08)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + s * 6.0, 0, 255)
    effect = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return blend_mask(bgr, effect, iris_m, 0.75 + s * 0.25)


def _apply_sclera_whiten(
    bgr: np.ndarray,
    eye_m: np.ndarray,
    iris_m: np.ndarray,
    amount: float,
) -> np.ndarray:
    s = strength_01(amount)
    if s <= 0:
        return bgr
    sclera = np.clip(eye_m - iris_m * 0.92, 0, 1)
    if sclera.max() <= 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + s * 30.0, 0, 255)
    lab[:, :, 1] = lab[:, :, 1] + (128.0 - lab[:, :, 1]) * (s * 0.35)
    lab[:, :, 2] = lab[:, :, 2] + (128.0 - lab[:, :, 2]) * (s * 0.22)
    effect = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return blend_mask(bgr, effect, sclera, 0.65 + s * 0.35)


def _apply_lash_darken(bgr: np.ndarray, lash_m: np.ndarray, amount: float) -> np.ndarray:
    s = strength_01(amount)
    if s <= 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] - s * 22.0, 0, 255)
    effect = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return blend_mask(bgr, effect, lash_m, 0.55 + s * 0.45)


def _apply_aegyo(bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    s = strength_01(amount)
    if s <= 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + s * 10.0, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + s * 3.0, 0, 255)
    effect = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return blend_mask(bgr, effect, mask, 0.35 + s * 0.45)


def _apply_eye_sharpen(bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    s = strength_01(amount)
    if s <= 0:
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), 0.85 + s * 0.6)
    detail = cv2.addWeighted(bgr, 1.0 + s * 0.85, blur, -(s * 0.85), 0)
    return blend_mask(bgr, detail, mask, 0.7 + s * 0.3)


def _enhance_one_eye(
    bgr: np.ndarray,
    landmarks,
    h: int,
    w: int,
    side: str,
    contour_key: str,
    iris_idx: int | None,
    params: CommercialEyeParams,
) -> np.ndarray:
    eye_sets = get_eye_region_index_sets()
    indices = eye_sets[contour_key]
    m = _eye_metrics(landmarks, w, h, indices, iris_idx, naturalness=params.naturalness)
    if m is None:
        return bgr

    cx, cy, radius = m
    blur = int(params.mask_edge_blur)
    eye_poly = _eye_polygon_mask(landmarks, h, w, indices, blur)
    iris_m = _iris_mask(h, w, cx, cy, radius)
    # 眯眼时虹膜很小，用整眼轮廓做提亮/锐化蒙版
    eye_blend = np.clip(np.maximum(iris_m, eye_poly * 0.82), 0.0, 1.0)
    out = bgr
    k = params.intensity

    out = _apply_iris_brighten(out, eye_blend, _scaled_amount(params.bright_eyes, k))
    out = _apply_sclera_whiten(out, eye_poly, iris_m, _scaled_amount(params.whiten_sclera, k))
    cl = _scaled_amount(params.catchlight, k)
    if cl > 0:
        out = _apply_catchlight(
            out,
            _catchlight_mask(h, w, cx, cy, radius, image_side=side),
            cl,
        )
    out = _apply_lash_darken(out, _lash_mask(landmarks, h, w, contour_key, radius), _scaled_amount(params.lash_line, k))
    out = _apply_aegyo(out, _aegyo_mask(h, w, cx, cy, radius), _scaled_amount(params.aegyo_sal, k))

    return out


def apply_commercial_eye_beauty(
    bgr: np.ndarray,
    landmarks,
    params: CommercialEyeParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full commercial eye pipeline on BGR image.
    Returns (bgr_out, eye_union_mask, under_eye_union_mask).
    """
    h, w = bgr.shape[:2]
    n = len(landmarks)

    k = params.intensity
    eye_amt = min(100.0, _scaled_amount(params.eye_size, k) * 1.15)
    nat = float(np.clip(params.naturalness - (k - 100.0) * 0.12, 0, 100))

    out, eye_union = apply_eye_size(
        bgr,
        landmarks,
        amount=eye_amt,
        naturalness=nat,
        mask_edge_blur=params.mask_edge_blur,
        eye_open=_scaled_amount(params.eye_open, k),
    )

    pairs = (
        ("left", "left_eye", _IRIS_LEFT_CENTER if n >= 474 else None),
        ("right", "right_eye", _IRIS_RIGHT_CENTER if n >= 474 else None),
    )
    for side, key, iris_idx in pairs:
        out = _enhance_one_eye(out, landmarks, h, w, side, key, iris_idx, params)

    under_union = np.zeros((h, w), dtype=np.float32)
    eye_sets = get_eye_region_index_sets()
    for _side, key, iris_idx in pairs:
        m = _eye_metrics(landmarks, w, h, eye_sets[key], iris_idx, naturalness=params.naturalness)
        if m is None:
            continue
        cx, cy, radius = m
        u = _under_eye_mask(h, w, cx, cy, radius)
        under_union = np.maximum(under_union, u)
        if params.under_eye > 0:
            out = apply_dark_circle(out, u, _scaled_amount(params.under_eye, k))

    if params.sharpen > 0:
        eye_union = build_eyes_union_mask(landmarks, h, w, params.mask_edge_blur, nat)
        sharp_m = np.clip(eye_union + under_union * 0.35, 0, 1)
        out = _apply_eye_sharpen(out, sharp_m, _scaled_amount(params.sharpen, k))

    if np.array_equal(out, bgr):
        _LOG.warning(
            "Commercial eye beauty: output identical to input — raise intensity/eye_size "
            "or check sliders are not 0."
        )

    return out, eye_union, under_union


def apply_commercial_eye_beauty_from_rgb(
    frame_rgb_uint8: np.ndarray,
    params: CommercialEyeParams,
    min_detection_confidence: float,
    min_presence_confidence: float,
    detect_rgb_uint8: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    detect_rgb = detect_rgb_uint8 if detect_rgb_uint8 is not None else frame_rgb_uint8
    landmarks = detect_face_landmarks(
        detect_rgb,
        min_detection_confidence,
        min_presence_confidence,
    )
    bgr = image_to_bgr_uint8(frame_rgb_uint8)
    h, w = bgr.shape[:2]

    if landmarks is None:
        _LOG.warning("Commercial eye beauty: no face landmarks detected.")
        z = np.zeros((h, w), dtype=np.float32)
        return bgr, z, z, False

    out, eye_m, under_m = apply_commercial_eye_beauty(bgr, landmarks, params)
    return out, eye_m, under_m, True


def build_commercial_debug_preview(
    frame_rgb_uint8: np.ndarray,
    out_bgr: np.ndarray,
    eye_m: np.ndarray,
    under_m: np.ndarray,
    ok: bool,
) -> np.ndarray:
    """Overlay masks + status text for ComfyUI preview."""
    base = image_to_bgr_uint8(frame_rgb_uint8)
    h, w = base.shape[:2]

    if not ok or eye_m.max() <= 0.001:
        banner = base.copy()
        cv2.rectangle(banner, (0, 0), (w, min(56, h)), (0, 0, 180), -1)
        msg = "landmarks_detected=False  请 detect_image 接原图，阈值降到 0.3"
        if ok and eye_m.max() <= 0.001:
            msg = "landmarks_detected=True 但 eye_mask 为空，请检查人脸/眼睛"
        cv2.putText(banner, msg, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            banner,
            "Connect debug_preview -> Preview Image",
            (8, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 255),
            1,
            cv2.LINE_AA,
        )
        return banner

    green = np.zeros_like(base)
    green[:, :, 1] = (np.clip(eye_m, 0, 1) * 210).astype(np.uint8)
    cyan = np.zeros_like(base)
    cyan[:, :, 0] = (np.clip(under_m, 0, 1) * 180).astype(np.uint8)
    cyan[:, :, 1] = (np.clip(under_m, 0, 1) * 180).astype(np.uint8)
    overlay = cv2.addWeighted(base, 0.55, green, 0.30, 0)
    overlay = cv2.addWeighted(overlay, 0.92, cyan, 0.08, 0)

    cov = float(eye_m.mean()) * 100.0
    cv2.rectangle(overlay, (0, h - 36), (w, h), (20, 20, 20), -1)
    cv2.putText(
        overlay,
        f"landmarks_detected=True  eye_mask={cov:.2f}%  green=eye cyan=under_eye",
        (8, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    # side-by-side: original | result
    result = out_bgr
    if result.shape[:2] != base.shape[:2]:
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
    gap = np.full((h, 4, 3), 128, dtype=np.uint8)
    compare = np.concatenate([base, gap, result], axis=1)
    return compare
