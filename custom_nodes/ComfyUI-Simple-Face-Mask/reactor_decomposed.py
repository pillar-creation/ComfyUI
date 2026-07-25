"""ReActor face swap split into crop + tunable paste-back (requires ComfyUI-ReActor)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import cv2
import folder_paths
import numpy as np
import torch

_INTERP = {
    "Nearest": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicubic": cv2.INTER_CUBIC,
    "Lanczos": cv2.INTER_LANCZOS4,
}

FACE_REGION_INPUT = {
    "forehead_trim": (
        "FLOAT",
        {
            "default": 35.0,
            "min": 0.0,
            "max": 100.0,
            "step": 1.0,
            "tooltip": "上收额头轮廓，避免圈到头发。先试 30–50，仍含发可提到 60。",
        },
    ),
    "face_inset": (
        "FLOAT",
        {
            "default": 8.0,
            "min": 0.0,
            "max": 50.0,
            "step": 1.0,
            "tooltip": "脸轮廓整体向内收缩（百分比）。",
        },
    ),
    "temple_trim": (
        "FLOAT",
        {
            "default": 38.0,
            "min": 0.0,
            "max": 100.0,
            "step": 1.0,
            "tooltip": "太阳穴/鬓角内收，避免圈到碎发。箭头处仍含发可调到 45–55。",
        },
    ),
    "exclude_hair": (
        "BOOLEAN",
        {
            "default": True,
            "tooltip": "语义分割扣除头发像素（需 selfie_multiclass 模型）。",
        },
    ),
}


@dataclass
class SwapPasteData:
    """Intermediate swap result: 128px BGR crop + affine matrix M (2x3)."""

    bgr_fake: np.ndarray
    affine_m: np.ndarray


@dataclass
class SwapAlignData:
    """Step 1 output: aligned crop + M + faces for ONNX (128 or 256)."""

    aligned_bgr: np.ndarray
    affine_m: np.ndarray
    source_face: object
    swap_model: str
    target_bgr: np.ndarray
    target_face: object
    input_size: int


_HYPERSWAP_STD_256 = np.array(
    [
        [84.87, 105.94],
        [171.13, 105.94],
        [128.00, 146.66],
        [96.95, 188.64],
        [159.05, 188.64],
    ],
    dtype=np.float32,
)


def _is_hyperswap(model: str) -> bool:
    return "hyperswap" in model.lower()


def _ensure_reactor_path() -> str:
    for base in folder_paths.get_folder_paths("custom_nodes"):
        path = os.path.join(base, "ComfyUI-ReActor")
        if os.path.isdir(path):
            if path not in sys.path:
                sys.path.insert(0, path)
            return path
    raise ImportError(
        "ComfyUI-ReActor not found. Install to custom_nodes/ComfyUI-ReActor."
    )


def _reactor_swap_models() -> list[str]:
    _ensure_reactor_path()
    from scripts.reactor_faceswap import get_models

    names = sorted({os.path.basename(x) for x in get_models()})
    return names or ["inswapper_128.onnx"]


def _rgb_tensor_to_bgr(image: torch.Tensor) -> np.ndarray:
    frame = (image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0)


def _checkerboard(h: int, w: int, cell: int = 16) -> np.ndarray:
    yy, xx = np.indices((h, w))
    check = ((yy // cell) + (xx // cell)) % 2
    base = np.where(check[..., np.newaxis], 200, 140).astype(np.uint8)
    return np.repeat(base, 3, axis=2)


def _crop_preview_tensor(bgr_fake: np.ndarray, pad_size: int = 256) -> torch.Tensor:
    """128px crop at 1:1 on checkerboard — clearly NOT pasted onto the original."""
    h, w = bgr_fake.shape[:2]
    canvas = _checkerboard(pad_size, pad_size)
    y0 = max(0, (pad_size - h) // 2)
    x0 = max(0, (pad_size - w) // 2)
    y1, x1 = min(pad_size, y0 + h), min(pad_size, x0 + w)
    canvas[y0:y1, x0:x1] = bgr_fake[: y1 - y0, : x1 - x0]
    return _bgr_to_tensor(canvas)


def _mask_tensor_to_2d(
    mask_t: torch.Tensor,
    expected_h: int,
    expected_w: int,
    name: str,
) -> np.ndarray:
    """ComfyUI MASK [B,H,W] → float32 [H,W], resize if needed to match swap crop."""
    m = mask_t.detach().cpu().numpy().astype(np.float32)
    if m.ndim == 3:
        m = m[0]
    elif m.ndim > 3:
        m = np.squeeze(m)
    if m.shape != (expected_h, expected_w):
        m = cv2.resize(m, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
    if m.shape != (expected_h, expected_w):
        raise ValueError(
            f"{name}: mask size {m.shape[1]}x{m.shape[0]} != crop {expected_w}x{expected_h}"
        )
    return np.clip(m, 0.0, 1.0)


def _mask_gray_preview_tensor(mask: np.ndarray, pad_size: int = 256) -> torch.Tensor:
    """Grayscale mask as RGB IMAGE for PreviewImage (MASK must not go to PreviewImage)."""
    h, w = mask.shape[:2]
    gray = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    canvas = _checkerboard(pad_size, pad_size)
    y0 = max(0, (pad_size - h) // 2)
    x0 = max(0, (pad_size - w) // 2)
    y1, x1 = min(pad_size, y0 + h), min(pad_size, x0 + w)
    canvas[y0:y1, x0:x1] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[: y1 - y0, : x1 - x0]
    return _bgr_to_tensor(canvas)


def _resolve_model_path(model: str) -> str:
    _ensure_reactor_path()
    from scripts.reactor_swapper import hyperswap_path, insightface_path, reswapper_path

    if "hyperswap" in model.lower():
        return os.path.join(hyperswap_path, model)
    if "reswapper" in model.lower():
        return os.path.join(reswapper_path, model)
    return os.path.join(insightface_path, model)


def _parse_face_indices(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _detect_face_pair(
    target_bgr: np.ndarray,
    source_bgr: np.ndarray,
    input_faces_index: str,
    source_faces_index: str,
):
    _ensure_reactor_path()
    from scripts.reactor_swapper import analyze_faces, get_face_single

    target_faces = analyze_faces(target_bgr)
    source_faces = analyze_faces(source_bgr)
    tgt_idx = _parse_face_indices(input_faces_index) or [0]
    src_idx = _parse_face_indices(source_faces_index) or [0]

    target_face, _, _ = get_face_single(
        target_bgr, target_faces, face_index=tgt_idx[0], order="large-small"
    )
    source_face, _, _ = get_face_single(
        source_bgr, source_faces, face_index=src_idx[0], order="large-small"
    )
    if target_face is None:
        raise RuntimeError("ReActor: no face detected on target image.")
    if source_face is None:
        raise RuntimeError("ReActor: no face detected on source image.")
    return target_face, source_face


def _compute_affine_m(target_face, input_size: int, align_tighten: float = 1.0) -> np.ndarray:
    _ensure_reactor_path()
    from reactor_core.inswap import ARCFACE_STD_POINTS

    ratio = float(input_size) / 128.0
    diff_x = 8.0 * ratio
    src_pts = ARCFACE_STD_POINTS.copy() * ratio
    src_pts[:, 0] += diff_x

    kps = np.asarray(target_face.kps, dtype=np.float32).copy()
    tighten = float(np.clip(align_tighten, 0.75, 1.0))
    if tighten < 0.999:
        center = kps.mean(axis=0)
        kps = center + (kps - center) * tighten

    m, _ = cv2.estimateAffinePartial2D(kps, src_pts)
    return m


def _compute_hyperswap_affine(target_face) -> np.ndarray:
    kps = np.asarray(target_face.kps, dtype=np.float32)
    m, _ = cv2.estimateAffinePartial2D(kps, _HYPERSWAP_STD_256)
    return m


def _align_face_bgr(img_bgr: np.ndarray, affine_m: np.ndarray, size: int) -> np.ndarray:
    return cv2.warpAffine(img_bgr, affine_m, (size, size), borderValue=0.0)


def _run_onnx_swap(swapper, aligned_bgr: np.ndarray, source_face) -> np.ndarray:
    blob = cv2.dnn.blobFromImage(
        aligned_bgr,
        1.0 / swapper.input_std,
        swapper.input_size,
        (swapper.input_mean, swapper.input_mean, swapper.input_mean),
        swapRB=True,
    )
    latent = source_face.normed_embedding.reshape((1, -1))
    latent = np.dot(latent, swapper.emap)
    latent /= np.linalg.norm(latent)
    pred = swapper.session.run(
        swapper.output_names,
        {
            swapper.input_names[0]: blob,
            swapper.input_names[1]: latent.astype(np.float32),
        },
    )[0]
    img_fake = pred.transpose((0, 2, 3, 1))[0]
    return np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]


def run_swap_align(
    target_bgr: np.ndarray,
    source_bgr: np.ndarray,
    swap_model: str,
    input_faces_index: str = "0",
    source_faces_index: str = "0",
    align_tighten: float = 1.0,
) -> SwapAlignData:
    target_face, source_face = _detect_face_pair(
        target_bgr, source_bgr, input_faces_index, source_faces_index
    )
    _ensure_reactor_path()
    from scripts.reactor_swapper import getFaceSwapModel

    swapper = getFaceSwapModel(_resolve_model_path(swap_model))
    if _is_hyperswap(swap_model):
        input_size = 256
        affine_m = _compute_hyperswap_affine(target_face)
        aligned_bgr = cv2.warpAffine(
            target_bgr,
            affine_m,
            (input_size, input_size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
    else:
        input_size = int(swapper.input_size[0])
        affine_m = _compute_affine_m(target_face, input_size, align_tighten)
        aligned_bgr = _align_face_bgr(target_bgr, affine_m, input_size)
    return SwapAlignData(
        aligned_bgr=aligned_bgr,
        affine_m=affine_m,
        source_face=source_face,
        swap_model=swap_model,
        target_bgr=target_bgr,
        target_face=target_face,
        input_size=input_size,
    )


def run_swap_infer(align_data: SwapAlignData) -> SwapPasteData:
    _ensure_reactor_path()
    from scripts.reactor_swapper import getFaceSwapModel

    swapper = getFaceSwapModel(_resolve_model_path(align_data.swap_model))
    if _is_hyperswap(align_data.swap_model):
        face_out, affine_m = swapper.get(
            align_data.target_bgr,
            align_data.target_face,
            align_data.source_face,
            paste_back=False,
        )
        if face_out is None or affine_m is None:
            raise RuntimeError("ReActorSwapInfer: hyperswap returned empty crop.")
        bgr_fake = np.ascontiguousarray(face_out[:, :, ::-1])
        return SwapPasteData(bgr_fake=bgr_fake, affine_m=affine_m)

    bgr_fake = _run_onnx_swap(swapper, align_data.aligned_bgr, align_data.source_face)
    return SwapPasteData(bgr_fake=bgr_fake, affine_m=align_data.affine_m)


def run_swap_crop(
    target_bgr: np.ndarray,
    source_bgr: np.ndarray,
    swap_model: str,
    input_faces_index: str = "0",
    source_faces_index: str = "0",
) -> SwapPasteData:
    align = run_swap_align(
        target_bgr, source_bgr, swap_model, input_faces_index, source_faces_index
    )
    return run_swap_infer(align)


def _crop_core_mask(
    crop_h: int, crop_w: int, *, width_scale: float = 1.0, height_scale: float = 1.0
) -> np.ndarray:
    """Ellipse mask in 128px crop space; scales <1 shrink paste region toward face center."""
    width_scale = float(np.clip(width_scale, 0.5, 1.0))
    height_scale = float(np.clip(height_scale, 0.5, 1.0))
    mask = np.zeros((crop_h, crop_w), np.float32)
    cx = (crop_w - 1) * 0.5
    cy = (crop_h - 1) * 0.5
    rx = max(crop_w * 0.5 * width_scale, 1.0)
    ry = max(crop_h * 0.5 * height_scale, 1.0)
    yy, xx = np.indices((crop_h, crop_w))
    inside = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    mask[inside] = 255.0
    return mask


def paste_back_tuned(
    target_bgr: np.ndarray,
    data: SwapPasteData,
    *,
    mask_core_scale: float = 1.0,
    mask_width_scale: float = 1.0,
    mask_erode_div: int = 0,
    mask_dilate_px: int = 12,
    mask_blur_div: int = 15,
    mask_blur_min: int = 5,
    mask_threshold: int = 20,
    warp_interpolation: str = "Bicubic",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tunable paste-back. mask_core_scale / mask_width_scale shrink the paste ellipse
    before warp (useful to avoid cheek/hair overlap from the 128px square crop).
    """
    bgr_fake = data.bgr_fake
    m = data.affine_m
    interp = _INTERP.get(warp_interpolation, cv2.INTER_CUBIC)

    im = cv2.invertAffineTransform(m)
    h, w = target_bgr.shape[:2]
    bgr_fake_warped = cv2.warpAffine(bgr_fake, im, (w, h), borderValue=0.0, flags=interp)

    crop_h, crop_w = bgr_fake.shape[:2]
    core = float(np.clip(mask_core_scale, 0.5, 1.0))
    cheek = float(np.clip(mask_width_scale, 0.5, 1.0))
    img_white = _crop_core_mask(
        crop_h, crop_w, width_scale=core * cheek, height_scale=core
    )
    img_white_warped = cv2.warpAffine(img_white, im, (w, h), borderValue=0.0)
    img_white_warped[img_white_warped > mask_threshold] = 255.0

    img_mask = img_white_warped.copy()
    mask_h_inds, mask_w_inds = np.where(img_mask >= 255.0)

    if len(mask_h_inds) > 0 and len(mask_w_inds) > 0:
        mask_h = int(np.max(mask_h_inds) - np.min(mask_h_inds))
        mask_w = int(np.max(mask_w_inds) - np.min(mask_w_inds))
        mask_size = int(np.sqrt(mask_h * mask_w))

        if mask_erode_div > 0:
            k = max(mask_size // mask_erode_div, 3)
            kernel = np.ones((k, k), np.uint8)
            img_mask = cv2.erode(img_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)

        if mask_dilate_px > 0:
            k = mask_dilate_px | 1
            kernel = np.ones((k, k), np.uint8)
            img_mask = cv2.dilate(img_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)

        blur_k = max(mask_size // max(mask_blur_div, 1), mask_blur_min)
        blur_k = blur_k | 1
        img_mask = cv2.GaussianBlur(img_mask, (blur_k, blur_k), 0)

    img_mask = np.clip(img_mask / 255.0, 0.0, 1.0)
    mask_2d = img_mask.astype(np.float32)
    merged = mask_2d[..., np.newaxis] * bgr_fake_warped.astype(np.float32) + (
        1.0 - mask_2d[..., np.newaxis]
    ) * target_bgr.astype(np.float32)
    return merged.astype(np.uint8), mask_2d, bgr_fake_warped


def _refine_feature_mask(
    mask: np.ndarray,
    *,
    mask_dilate_px: int = 2,
    mask_blur: int = 5,
) -> np.ndarray:
    out = np.clip(mask, 0.0, 1.0).astype(np.float32)
    if mask_dilate_px > 0:
        k = mask_dilate_px | 1
        kernel = np.ones((k, k), np.uint8)
        out = cv2.dilate((out * 255).astype(np.uint8), kernel, iterations=1).astype(np.float32) / 255.0
    if mask_blur > 0:
        k = max(3, mask_blur | 1)
        out = cv2.GaussianBlur(out, (k, k), 0)
    return np.clip(out, 0.0, 1.0)


def _warp_feature_mask_to_target(
    crop_mask: np.ndarray,
    affine_m: np.ndarray,
    target_w: int,
    target_h: int,
    *,
    mask_dilate_px: int = 2,
    mask_blur: int = 5,
    mask_threshold: int = 20,
) -> np.ndarray:
    """Warp crop-space feature mask to full target image with paste-back style cleanup."""
    im = cv2.invertAffineTransform(affine_m)
    crop_u8 = (np.clip(crop_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    warped = cv2.warpAffine(
        crop_u8,
        im,
        (target_w, target_h),
        borderValue=0,
        flags=cv2.INTER_LINEAR,
    ).astype(np.float32)

    if warped.max() <= 0:
        return np.zeros((target_h, target_w), dtype=np.float32)

    warped[warped > mask_threshold] = 255.0
    mask_h_inds, mask_w_inds = np.where(warped >= 255.0)
    if len(mask_h_inds) > 0 and len(mask_w_inds) > 0:
        mask_h = int(np.max(mask_h_inds) - np.min(mask_h_inds))
        mask_w = int(np.max(mask_w_inds) - np.min(mask_w_inds))
        mask_size = max(int(np.sqrt(mask_h * mask_w)), 8)

        if mask_dilate_px > 0:
            k = mask_dilate_px | 1
            kernel = np.ones((k, k), np.uint8)
            warped = cv2.dilate(warped.astype(np.uint8), kernel, iterations=1).astype(np.float32)

        if mask_blur > 0:
            blur_k = max(mask_size // 10, mask_blur) | 1
            warped = cv2.GaussianBlur(warped, (blur_k, blur_k), 0)

    return np.clip(warped / 255.0, 0.0, 1.0)


def paste_back_features(
    target_bgr: np.ndarray,
    data: SwapPasteData,
    *,
    include_eyebrows: bool = True,
    include_eyes: bool = True,
    include_nose: bool = True,
    include_mouth: bool = True,
    include_face_triangle: bool = False,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    mask_edge_blur: int = 5,
    brow_thickness: float = 1.0,
    mask_dilate_px: int = 2,
    mask_blur: int = 5,
    warp_interpolation: str = "Bicubic",
    forehead_trim: float = 35.0,
    face_inset: float = 8.0,
    temple_trim: float = 38.0,
    exclude_hair: bool = True,
    crop_feature_mask: np.ndarray | None = None,
    crop_face_region_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Landmark-based paste-back: only eyebrows / eyes / nose / mouth from swapped crop.
    Falls back to a small center ellipse if landmarks are not detected.

    When crop_feature_mask is supplied (e.g. from ②b), mask-generation kwargs are
    ignored and only paste-back tuning (dilate/blur/warp) is applied.
    """
    from .face_utils import detect_face_region_mask_from_bgr, detect_features_mask_from_bgr

    bgr_fake = data.bgr_fake
    m = data.affine_m
    interp = _INTERP.get(warp_interpolation, cv2.INTER_CUBIC)
    im = cv2.invertAffineTransform(m)
    h, w = target_bgr.shape[:2]
    crop_h, crop_w = bgr_fake.shape[:2]
    use_external_mask = crop_feature_mask is not None

    if use_external_mask:
        crop_mask = np.clip(crop_feature_mask.astype(np.float32), 0.0, 1.0)
        if crop_mask.shape[:2] != (crop_h, crop_w):
            crop_mask = cv2.resize(crop_mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        ok = crop_mask.max() > 0
        face_region = None
        if crop_face_region_mask is not None:
            face_region = np.clip(crop_face_region_mask.astype(np.float32), 0.0, 1.0)
            if face_region.shape[:2] != (crop_h, crop_w):
                face_region = cv2.resize(
                    face_region, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR
                )
    else:
        crop_mask, face_region, ok = detect_features_mask_from_bgr(
            bgr_fake,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            include_eyebrows=include_eyebrows,
            include_eyes=include_eyes,
            include_nose=include_nose,
            include_mouth=include_mouth,
            include_face_triangle=include_face_triangle,
            mask_edge_blur=mask_edge_blur,
            brow_thickness=brow_thickness,
            forehead_trim=forehead_trim,
            face_inset=face_inset,
            temple_trim=temple_trim,
            exclude_hair=exclude_hair,
        )
        if crop_mask is None or crop_mask.max() <= 0:
            crop_mask = _crop_core_mask(crop_h, crop_w, width_scale=0.55, height_scale=0.55) / 255.0
            face_region = None
            ok = False

    crop_mask = _refine_feature_mask(crop_mask, mask_dilate_px=mask_dilate_px, mask_blur=mask_blur)
    if not use_external_mask and face_region is not None and face_region.max() > 0:
        crop_mask = np.clip(crop_mask * face_region, 0.0, 1.0)

    bgr_fake_warped = cv2.warpAffine(bgr_fake, im, (w, h), borderValue=0.0, flags=interp)
    mask_2d = _warp_feature_mask_to_target(
        crop_mask,
        m,
        w,
        h,
        mask_dilate_px=mask_dilate_px,
        mask_blur=mask_blur,
    )

    if (
        crop_face_region_mask is not None
        and face_region is not None
        and face_region.max() > 0
    ):
        target_region = _warp_feature_mask_to_target(
            face_region,
            m,
            w,
            h,
            mask_dilate_px=0,
            mask_blur=max(0, mask_edge_blur),
        )
    else:
        target_region, _ = detect_face_region_mask_from_bgr(
            target_bgr,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            mask_edge_blur=mask_edge_blur,
            forehead_trim=forehead_trim,
            face_inset=face_inset,
            temple_trim=temple_trim,
            exclude_hair=exclude_hair,
        )
    if target_region is not None and target_region.max() > 0:
        mask_2d = np.clip(mask_2d * target_region, 0.0, 1.0)

    merged = mask_2d[..., np.newaxis] * bgr_fake_warped.astype(np.float32) + (
        1.0 - mask_2d[..., np.newaxis]
    ) * target_bgr.astype(np.float32)
    return merged.astype(np.uint8), mask_2d, bgr_fake_warped, ok


class ReActorSwapAlign:
    """Step 1: RetinaFace detect + 5-point affine → 128px aligned target crop."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            models = _reactor_swap_models()
        except ImportError:
            models = ["inswapper_128.onnx"]
        return {
            "required": {
                "target_image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "swap_model": (models, {"default": "inswapper_128.onnx"}),
                "input_faces_index": ("STRING", {"default": "0"}),
                "source_faces_index": ("STRING", {"default": "0"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "SWAP_ALIGN_DATA")
    RETURN_NAMES = ("aligned_crop", "align_data")
    OUTPUT_TOOLTIPS = (
        "128px 对齐后的目标脸（棋盘格底，ONNX 推理前）",
        "对齐数据：脸块 + 仿射矩阵 + 源脸身份，接 ② ONNX 节点",
    )
    FUNCTION = "run"
    CATEGORY = "image/reactor"

    def run(self, target_image, source_image, swap_model, input_faces_index, source_faces_index):
        if target_image.shape[0] != 1 or source_image.shape[0] != 1:
            raise ValueError("ReActorSwapAlign: batch size must be 1.")
        data = run_swap_align(
            _rgb_tensor_to_bgr(target_image),
            _rgb_tensor_to_bgr(source_image),
            swap_model,
            input_faces_index,
            source_faces_index,
        )
        return (_crop_preview_tensor(data.aligned_bgr), data)


class ReActorSwapInfer:
    """Steps 2–3: source embedding × emap → inswapper ONNX → 128px swapped crop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"align_data": ("SWAP_ALIGN_DATA",)}}

    RETURN_TYPES = ("IMAGE", "SWAP_PASTE_DATA")
    RETURN_NAMES = ("swapped_crop", "paste_data")
    OUTPUT_TOOLTIPS = (
        "128px 换脸结果（棋盘格底，ONNX 输出，未贴回）",
        "贴回所需数据：脸块 + 仿射矩阵，接 ③ 贴回节点",
    )
    FUNCTION = "run"
    CATEGORY = "image/reactor"

    def run(self, align_data):
        data = run_swap_infer(align_data)
        return (_crop_preview_tensor(data.bgr_fake), data)


class ReActorSwapCrop:
    """Shortcut: align + ONNX infer in one node (no paste-back)."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            models = _reactor_swap_models()
        except ImportError:
            models = ["inswapper_128.onnx"]
        return {
            "required": {
                "target_image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "swap_model": (models, {"default": "inswapper_128.onnx"}),
                "input_faces_index": ("STRING", {"default": "0"}),
                "source_faces_index": ("STRING", {"default": "0"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "SWAP_PASTE_DATA")
    RETURN_NAMES = ("face_crop_only", "paste_data")
    OUTPUT_TOOLTIPS = (
        "128px 换脸结果（棋盘格底，未贴回原图）",
        "贴回所需数据：脸块 + 仿射矩阵，接 ② 贴回节点",
    )
    FUNCTION = "run"
    CATEGORY = "image/reactor"

    def run(self, target_image, source_image, swap_model, input_faces_index, source_faces_index):
        if target_image.shape[0] != 1 or source_image.shape[0] != 1:
            raise ValueError("ReActorSwapCrop: batch size must be 1.")
        data = run_swap_crop(
            _rgb_tensor_to_bgr(target_image),
            _rgb_tensor_to_bgr(source_image),
            swap_model,
            input_faces_index,
            source_faces_index,
        )
        return (_crop_preview_tensor(data.bgr_fake), data)


class ReActorSwapPasteBack:
    """Step 4: inverse affine paste-back + feathered mask blend."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "paste_data": ("SWAP_PASTE_DATA",),
                "warp_interpolation": (
                    ["Nearest", "Bilinear", "Bicubic", "Lanczos"],
                    {"default": "Bicubic"},
                ),
                "mask_core_scale": (
                    "FLOAT",
                    {
                        "default": 0.88,
                        "min": 0.55,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "贴回椭圆整体缩放。越小越只贴脸心（1.0=整块128px）。看 paste_mask 预览调节。",
                    },
                ),
                "mask_width_scale": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.55,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "额外收左右脸颊（在 core_scale 基础上再缩宽度）。减轻两侧重叠。",
                    },
                ),
                "mask_erode_div": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 30,
                        "step": 1,
                        "tooltip": "蒙版均匀腐蚀像素块大小（越大越向内收）。与 core_scale 叠加使用。",
                    },
                ),
                "mask_dilate_px": (
                    "INT",
                    {
                        "default": 12,
                        "min": 0,
                        "max": 51,
                        "step": 2,
                        "tooltip": "蒙版外扩像素，覆盖贴图边缘露出的原脸。",
                    },
                ),
                "mask_blur_div": (
                    "INT",
                    {"default": 15, "min": 5, "max": 40, "step": 1},
                ),
                "mask_blur_min": ("INT", {"default": 5, "min": 3, "max": 31, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("swapped_image", "paste_mask", "warped_layer")
    OUTPUT_TOOLTIPS = (
        "贴回并与原图混合后的整图",
        "贴回蒙版（白=换脸区域）",
        "仅仿射变换到原图坐标的脸层，未混合（背景为黑）",
    )
    FUNCTION = "run"
    CATEGORY = "image/reactor"

    def run(
        self,
        target_image,
        paste_data,
        warp_interpolation,
        mask_core_scale,
        mask_width_scale,
        mask_erode_div,
        mask_dilate_px,
        mask_blur_div,
        mask_blur_min,
    ):
        if target_image.shape[0] != 1:
            raise ValueError("ReActorSwapPasteBack: batch size must be 1.")
        merged, mask, warped = paste_back_tuned(
            _rgb_tensor_to_bgr(target_image),
            paste_data,
            mask_core_scale=mask_core_scale,
            mask_width_scale=mask_width_scale,
            mask_erode_div=mask_erode_div,
            mask_dilate_px=mask_dilate_px,
            mask_blur_div=mask_blur_div,
            mask_blur_min=mask_blur_min,
            warp_interpolation=warp_interpolation,
        )
        return (
            _bgr_to_tensor(merged),
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
            _bgr_to_tensor(warped),
        )


class ReActorSwapPasteBackFeatures:
    """Step 4: paste swapped features onto target. Tune masks on ②b, wire feature_mask here."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "paste_data": ("SWAP_PASTE_DATA",),
                "warp_interpolation": (
                    ["Nearest", "Bilinear", "Bicubic", "Lanczos"],
                    {"default": "Bicubic"},
                ),
                "mask_dilate_px": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 15,
                        "step": 1,
                        "tooltip": "贴回蒙版外扩像素，消除边缘缝隙。",
                    },
                ),
                "mask_blur": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 31,
                        "step": 2,
                        "tooltip": "贴回前整体蒙版羽化。",
                    },
                ),
            },
            "optional": {
                "feature_mask": (
                    "MASK",
                    {
                        "tooltip": "接 ②b feature_mask（推荐）。未接时用默认规则在内部生成蒙版。",
                    },
                ),
                "face_region_mask": (
                    "MASK",
                    {
                        "tooltip": "接 ②b face_region_mask（推荐）。全图贴回脸区裁剪与 ②b 一致。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("swapped_image", "paste_mask", "warped_layer", "landmarks_detected")
    OUTPUT_TOOLTIPS = (
        "仅五官贴回后的整图",
        "五官联合蒙版（白=替换区域）",
        "仿射到原图坐标的换脸层（未混合）",
        "是否在换脸脸块上成功检测到关键点",
    )
    FUNCTION = "run"
    CATEGORY = "image/reactor"

    def run(
        self,
        target_image,
        paste_data,
        warp_interpolation,
        mask_dilate_px,
        mask_blur,
        feature_mask=None,
        face_region_mask=None,
    ):
        if target_image.shape[0] != 1:
            raise ValueError("ReActorSwapPasteBackFeatures: batch size must be 1.")
        crop_h, crop_w = paste_data.bgr_fake.shape[:2]
        crop_feature = None
        crop_face_region = None
        if feature_mask is not None:
            crop_feature = _mask_tensor_to_2d(feature_mask, crop_h, crop_w, "feature_mask")
        if face_region_mask is not None:
            crop_face_region = _mask_tensor_to_2d(
                face_region_mask, crop_h, crop_w, "face_region_mask"
            )
        merged, mask, warped, ok = paste_back_features(
            _rgb_tensor_to_bgr(target_image),
            paste_data,
            mask_dilate_px=mask_dilate_px,
            mask_blur=mask_blur,
            warp_interpolation=warp_interpolation,
            crop_feature_mask=crop_feature,
            crop_face_region_mask=crop_face_region,
        )
        return (
            _bgr_to_tensor(merged),
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
            _bgr_to_tensor(warped),
            ok,
        )


class ReActorFeatureMaskPreview:
    """Preview feature mask on swapped crop (128px) before paste-back."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "paste_data": ("SWAP_PASTE_DATA",),
                "include_eyebrows": ("BOOLEAN", {"default": True}),
                "include_eyes": ("BOOLEAN", {"default": True}),
                "include_nose": ("BOOLEAN", {"default": True}),
                "include_mouth": ("BOOLEAN", {"default": True}),
                "include_face_triangle": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "脸部三角区（内眦→嘴角→下巴 + 法令纹），填充面中衔接皮肤。",
                    },
                ),
                "min_detection_confidence": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05},
                ),
                "min_presence_confidence": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05},
                ),
                "mask_edge_blur": ("INT", {"default": 5, "min": 0, "max": 21, "step": 2}),
                "brow_thickness": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                **FACE_REGION_INPUT,
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MASK", "IMAGE", "BOOLEAN")
    RETURN_NAMES = (
        "feature_mask",
        "overlay_preview",
        "face_region_mask",
        "feature_mask_gray",
        "landmarks_detected",
    )
    OUTPUT_TOOLTIPS = (
        "五官联合蒙版（128px 脸块，接 ③ feature_mask 实现一套参数）",
        "绿区=五官、橙线=脸轮廓（接 PreviewImage）",
        "脸部椭圆区域蒙版（128px，接 ③ face_region_mask）",
        "五官蒙版灰度图（IMAGE，专供 PreviewImage 预览）",
        "是否检测到有效脸部区域",
    )
    FUNCTION = "run"
    CATEGORY = "image/reactor"

    def run(
        self,
        paste_data,
        include_eyebrows,
        include_eyes,
        include_nose,
        include_mouth,
        include_face_triangle,
        min_detection_confidence,
        min_presence_confidence,
        mask_edge_blur,
        brow_thickness,
        forehead_trim,
        face_inset,
        temple_trim,
        exclude_hair,
    ):
        from .face_utils import detect_features_mask_from_bgr

        bgr = paste_data.bgr_fake
        h, w = bgr.shape[:2]
        mask, face_region, ok = detect_features_mask_from_bgr(
            bgr,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            include_eyebrows=include_eyebrows,
            include_eyes=include_eyes,
            include_nose=include_nose,
            include_mouth=include_mouth,
            include_face_triangle=include_face_triangle,
            mask_edge_blur=mask_edge_blur,
            brow_thickness=brow_thickness,
            forehead_trim=forehead_trim,
            face_inset=face_inset,
            temple_trim=temple_trim,
            exclude_hair=exclude_hair,
        )
        if mask is None:
            mask = np.zeros((h, w), dtype=np.float32)
            face_region = np.zeros((h, w), dtype=np.float32)
            ok = False
        elif face_region is None:
            face_region = np.zeros((h, w), dtype=np.float32)

        overlay = bgr.copy()
        if mask.max() > 0:
            tint = np.zeros_like(overlay)
            tint[:, :, 1] = (np.clip(mask, 0, 1) * 200).astype(np.uint8)
            overlay = cv2.addWeighted(overlay, 0.55, tint, 0.45, 0)
        if face_region.max() > 0:
            face_u8 = (np.clip(face_region, 0, 1) * 255).astype(np.uint8)
            contours, _ = cv2.findContours(face_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 160, 40), 1, cv2.LINE_AA)

        pad = 256
        canvas = _checkerboard(pad, pad)
        y0 = max(0, (pad - h) // 2)
        x0 = max(0, (pad - w) // 2)
        y1, x1 = min(pad, y0 + h), min(pad, x0 + w)
        canvas[y0:y1, x0:x1] = overlay[: y1 - y0, : x1 - x0]

        return (
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
            _bgr_to_tensor(canvas),
            torch.from_numpy(face_region.astype(np.float32)).unsqueeze(0),
            _mask_gray_preview_tensor(mask),
            ok,
        )


NODE_CLASS_MAPPINGS = {
    "ReActorSwapAlign": ReActorSwapAlign,
    "ReActorSwapInfer": ReActorSwapInfer,
    "ReActorSwapCrop": ReActorSwapCrop,
    "ReActorSwapPasteBack": ReActorSwapPasteBack,
    "ReActorSwapPasteBackFeatures": ReActorSwapPasteBackFeatures,
    "ReActorFeatureMaskPreview": ReActorFeatureMaskPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReActorSwapAlign": "ReActor ① 对齐裁切",
    "ReActorSwapInfer": "ReActor ② ONNX 换脸",
    "ReActorSwapCrop": "ReActor ①② 换脸裁切 (快捷)",
    "ReActorSwapPasteBack": "ReActor ③ 贴回蒙版 (core/cheek)",
    "ReActorSwapPasteBackFeatures": "ReActor ③ 五官贴回 (眉/眼/鼻/嘴)",
    "ReActorFeatureMaskPreview": "ReActor ②b 五官蒙版预览",
}
