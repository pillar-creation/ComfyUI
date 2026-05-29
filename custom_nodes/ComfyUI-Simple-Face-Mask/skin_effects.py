"""OpenCV skin retouch primitives (shared by monolithic and per-step nodes)."""

from __future__ import annotations

import cv2
import numpy as np

from .face_utils import bgr_to_rgb_float01, image_to_bgr_uint8


def strength_01(value: float) -> float:
    return float(np.clip(value / 100.0, 0.0, 1.0))


def blend_mask(base_bgr: np.ndarray, effect_bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return base_bgr
    m = np.clip(mask * amount, 0.0, 1.0)
    m3 = m[:, :, np.newaxis]
    out = base_bgr.astype(np.float32) * (1.0 - m3) + effect_bgr.astype(np.float32) * m3
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_smooth(bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    s = strength_01(amount)
    if s <= 0:
        return bgr
    d = int(5 + s * 10)
    sigma = 20 + s * 60
    smooth = cv2.bilateralFilter(bgr, d, sigma, sigma)
    return blend_mask(bgr, smooth, mask, s)


def apply_whiten(bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    s = strength_01(amount)
    if s <= 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + s * 22.0, 0, 255)
    effect = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return blend_mask(bgr, effect, mask, s * 0.85)


def apply_even_tone(bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Even skin color: multi-pass chroma + luminance low-frequency blend."""
    s = strength_01(amount)
    if s <= 0:
        return bgr
    h, w = bgr.shape[:2]
    base_k = max(11, min(h, w) // 18)
    k = int(base_k + s * base_k * 3.0)
    if k % 2 == 0:
        k += 1
    k = min(k, 81)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    sigma = 2.0 + s * 6.0

    a_blur = cv2.GaussianBlur(a, (k, k), sigma)
    b_blur = cv2.GaussianBlur(b, (k, k), sigma)
    if s >= 0.45:
        k2 = max(5, k // 2) | 1
        a_blur = cv2.GaussianBlur(a_blur, (k2, k2), sigma * 0.65)
        b_blur = cv2.GaussianBlur(b_blur, (k2, k2), sigma * 0.65)

    # Chroma: at 100 use fully flattened color
    mix_ab = min(1.0, 0.55 + s * 0.45)
    a_mix = cv2.addWeighted(a, 1.0 - mix_ab, a_blur, mix_ab, 0)
    b_mix = cv2.addWeighted(b, 1.0 - mix_ab, b_blur, mix_ab, 0)

    # Pull chroma toward face mean (removes patchy redness/yellow at high strength)
    if s >= 0.25:
        m = mask > 0.08
        if m.any():
            pull = s * 0.42
            ma = float(a[m].mean())
            mb = float(b[m].mean())
            a_mix = np.where(m, a_mix.astype(np.float32) * (1.0 - pull) + ma * pull, a_mix).astype(np.uint8)
            b_mix = np.where(m, b_mix.astype(np.float32) * (1.0 - pull) + mb * pull, b_mix).astype(np.uint8)

    l_blur = cv2.GaussianBlur(l, (k, k), sigma)
    if s >= 0.5:
        l_blur = cv2.GaussianBlur(l_blur, (max(5, k // 2) | 1, max(5, k // 2) | 1), sigma * 0.5)
    mix_l = s * 0.38
    l_mix = cv2.addWeighted(l, 1.0 - mix_l, l_blur, mix_l, 0)

    effect = cv2.cvtColor(cv2.merge([l_mix, a_mix, b_mix]), cv2.COLOR_LAB2BGR)
    return blend_mask(bgr, effect, mask, 0.35 + s * 0.65)


def apply_plump(bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    s = strength_01(amount)
    if s <= 0:
        return bgr
    k = int(3 + s * 8)
    if k % 2 == 0:
        k += 1
    soft = cv2.GaussianBlur(bgr, (k, k), 0)
    lab = cv2.cvtColor(soft, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + s * 12.0, 0, 255)
    effect = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return blend_mask(bgr, effect, mask, s * 0.7)


def apply_spot_remove(bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    s = strength_01(amount)
    if s <= 0:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.5)
    diff = cv2.absdiff(gray, blur)
    _, spot = cv2.threshold(diff, int(8 + (1.0 - s) * 10), 255, cv2.THRESH_BINARY)
    spot = cv2.bitwise_and(spot, m8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spot = cv2.morphologyEx(spot, cv2.MORPH_OPEN, k, iterations=1)
    spot = cv2.dilate(spot, k, iterations=1)
    if spot.max() == 0:
        return bgr
    radius = max(1, int(1 + s * 2))
    effect = cv2.inpaint(bgr, spot, radius, cv2.INPAINT_TELEA)
    spot_f = spot.astype(np.float32) / 255.0
    return blend_mask(bgr, effect, spot_f, 0.55 + s * 0.45)


def apply_nasolabial(bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    s = strength_01(amount)
    if s <= 0:
        return bgr
    k = int(5 + s * 14)
    if k % 2 == 0:
        k += 1
    effect = cv2.GaussianBlur(bgr, (k, k), 0)
    effect = cv2.bilateralFilter(effect, 5, 40 + s * 40, 40 + s * 40)
    return blend_mask(bgr, effect, mask, s * 0.9)


def apply_dark_circle(bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    s = strength_01(amount)
    if s <= 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + s * 18.0, 0, 255)
    lab[:, :, 1] = lab[:, :, 1] + (128.0 - lab[:, :, 1]) * (s * 0.25)
    lab[:, :, 2] = lab[:, :, 2] + (128.0 - lab[:, :, 2]) * (s * 0.35)
    effect = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return blend_mask(bgr, effect, mask, s * 0.95)


def apply_clarity(bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    s = strength_01(amount)
    if s <= 0:
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0 + s * 1.5)
    detail = cv2.addWeighted(bgr, 1.0 + s * 0.65, blur, -(s * 0.65), 0)
    return blend_mask(bgr, detail, mask, s * 0.75)


def process_skin_bgr(
    bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    smooth: float,
    whiten: float,
    even_tone: float,
    plump: float,
    spot_remove: float,
    nasolabial: float,
    dark_circle: float,
    clarity: float,
) -> np.ndarray:
    out = bgr.copy()
    face = masks["face"]
    if spot_remove > 0:
        out = apply_spot_remove(out, face, spot_remove)
    if smooth > 0:
        out = apply_smooth(out, face, smooth)
    if even_tone > 0:
        out = apply_even_tone(out, face, even_tone)
    if nasolabial > 0:
        out = apply_nasolabial(out, masks["nasolabial"], nasolabial)
    if dark_circle > 0:
        out = apply_dark_circle(out, masks["under_eye"], dark_circle)
    if plump > 0:
        out = apply_plump(out, masks["cheek"], plump)
    if whiten > 0:
        out = apply_whiten(out, face, whiten)
    if clarity > 0:
        out = apply_clarity(out, face, clarity)
    return out
