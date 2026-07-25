"""Per-step ComfyUI nodes for skin retouch pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .eye_effects import apply_eye_size_from_rgb
from .face_slim_effects import apply_face_slim_from_rgb
from .eye_beauty_effects import (
    CommercialEyeParams,
    apply_commercial_eye_beauty_from_rgb,
    build_commercial_debug_preview,
)
from .face_utils import detect_masks_from_rgb_frame
from .skin_effects import (
    apply_clarity,
    apply_dark_circle,
    apply_even_tone,
    apply_nasolabial,
    apply_plump,
    apply_smooth,
    apply_spot_remove,
    apply_whiten,
    bgr_to_rgb_float01,
    image_to_bgr_uint8,
)


def amount_slider(default: float = 0.0, tooltip: str = "") -> tuple:
    opts = {"default": default, "min": 0.0, "max": 100.0, "step": 1.0}
    if tooltip:
        opts["tooltip"] = tooltip
    return ("FLOAT", opts)


DETECT_INPUTS = {
    "skin_mask_mode": (
        ["semantic", "landmarks"],
        {
            "default": "semantic",
            "tooltip": "semantic=AI skin seg + edge refine (recommended); landmarks=mesh polygon only.",
        },
    ),
    "padding_percent": (
        "FLOAT",
        {
            "default": 0.06,
            "min": 0.0,
            "max": 0.35,
            "step": 0.01,
            "tooltip": "Face ROI margin for semantic mask (not rigid stretch).",
        },
    ),
    "mask_edge_blur": ("INT", {"default": 15, "min": 0, "max": 51, "step": 2}),
    "min_detection_confidence": (
        "FLOAT",
        {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05, "tooltip": "MediaPipe face detection threshold."},
    ),
    "min_presence_confidence": (
        "FLOAT",
        {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05, "tooltip": "MediaPipe landmark presence threshold."},
    ),
    "fallback_center_if_no_face": ("BOOLEAN", {"default": False}),
}

EYE_SIZE_INPUT = {
    "eye_size": (
        "FLOAT",
        {
            "default": 0.0,
            "min": -100.0,
            "max": 100.0,
            "step": 1.0,
            "tooltip": "强度。明显效果：60–100 + naturalness 30–50；自然：25–40 + naturalness 80+。",
        },
    ),
    "naturalness": (
        "FLOAT",
        {
            "default": 80.0,
            "min": 0.0,
            "max": 100.0,
            "step": 1.0,
            "tooltip": "越高越自然、越柔和；越低越明显（配合 eye_size 100 时建议 30–50）。",
        },
    ),
}

FACE_SLIM_INPUT = {
    "face_slim": (
        "FLOAT",
        {
            "default": 0.0,
            "min": 0.0,
            "max": 100.0,
            "step": 1.0,
            "tooltip": "面颊瘦脸（向面部中线收缩）。先试 60–85；仍不明显可提到 90–100 并降低 naturalness。",
        },
    ),
    "jaw_slim": (
        "FLOAT",
        {
            "default": 0.0,
            "min": 0.0,
            "max": 100.0,
            "step": 1.0,
            "tooltip": "下颌收窄。仅瘦下颚时保持 face_slim=0，只调此项（更快）。",
        },
    ),
    "naturalness": (
        "FLOAT",
        {
            "default": 80.0,
            "min": 0.0,
            "max": 100.0,
            "step": 1.0,
            "tooltip": "越高越自然；越低变形越明显。",
        },
    ),
}


def eye_amount(default: float, tooltip: str = "") -> tuple:
    opts: dict = {"default": default, "min": 0.0, "max": 100.0, "step": 1.0}
    if tooltip:
        opts["tooltip"] = tooltip
    return ("FLOAT", opts)


COMMERCIAL_EYE_INPUTS = {
    "intensity": (
        "FLOAT",
        {
            "default": 120.0,
            "min": 0.0,
            "max": 150.0,
            "step": 5.0,
            "tooltip": "总强度倍率。不明显时提到 130–150；过假则降到 90–100。",
        },
    ),
    "eye_size": eye_amount(68.0, "大眼（径向）"),
    "eye_open": eye_amount(55.0, "开眼（纵向拉长，商业 App 常用）"),
    "naturalness": eye_amount(50.0, "自然度：越高越柔和"),
    "bright_eyes": eye_amount(72.0, "亮瞳"),
    "whiten_sclera": eye_amount(55.0, "眼白（眯眼照宜低）"),
    "catchlight": eye_amount(55.0, "眼神光（虹膜柔和高光）"),
    "lash_line": eye_amount(58.0, "眼线 / 睫毛"),
    "aegyo_sal": eye_amount(30.0, "卧蚕（过高易眼下白点）"),
    "under_eye": eye_amount(60.0, "祛黑眼圈"),
    "sharpen": eye_amount(45.0, "眼部锐化"),
}


def _require_batch1(image: torch.Tensor, node_name: str) -> None:
    if image.shape[0] != 1:
        raise ValueError(f"{node_name}: batch size must be 1.")


def _frame_uint8(image: torch.Tensor) -> np.ndarray:
    return (image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


def _mask_numpy(mask: torch.Tensor, shape_hw: tuple[int, int]) -> np.ndarray:
    m = mask[0].detach().cpu().numpy().astype(np.float32)
    if m.shape != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)


def _tensor_from_bgr(bgr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bgr_to_rgb_float01(bgr)).unsqueeze(0)


def _resolve_detect_frame(image, detect_image):
    frame = _frame_uint8(image)
    detect_frame = _frame_uint8(detect_image) if detect_image is not None else None
    if detect_frame is not None and detect_frame.shape[:2] != frame.shape[:2]:
        detect_frame = cv2.resize(
            detect_frame,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return frame, detect_frame


def _apply_effect_node(image, mask, amount, node_name, effect_fn):
    _require_batch1(image, node_name)
    if amount <= 0:
        return (image,)
    frame = _frame_uint8(image)
    bgr = image_to_bgr_uint8(frame)
    m = _mask_numpy(mask, (bgr.shape[0], bgr.shape[1]))
    out = effect_fn(bgr, m, amount)
    return (_tensor_from_bgr(out),)


class FaceSkinDetectMasks:
    """MediaPipe landmarks → face skin / under-eye / nasolabial / cheek masks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                **DETECT_INPUTS,
            },
            "optional": {"restrict_mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("image", "face_mask", "under_eye_mask", "nasolabial_mask", "cheek_mask")
    FUNCTION = "detect"
    CATEGORY = "image/beauty"

    def detect(
        self,
        image,
        skin_mask_mode,
        padding_percent,
        mask_edge_blur,
        min_detection_confidence,
        min_presence_confidence,
        fallback_center_if_no_face,
        restrict_mask=None,
    ):
        _require_batch1(image, "FaceSkinDetectMasks")
        frame = _frame_uint8(image)
        masks = detect_masks_from_rgb_frame(
            frame,
            padding_percent,
            mask_edge_blur,
            min_detection_confidence,
            min_presence_confidence,
            fallback_center_if_no_face,
            skin_mask_mode=skin_mask_mode,
        )
        if restrict_mask is not None:
            user_m = _mask_numpy(restrict_mask, (frame.shape[0], frame.shape[1]))
            for key in masks:
                masks[key] = np.clip(masks[key] * user_m, 0.0, 1.0)

        def to_t(key):
            return torch.from_numpy(masks[key]).unsqueeze(0)

        return (image, to_t("face"), to_t("under_eye"), to_t("nasolabial"), to_t("cheek"))


class FaceSkinEyeSize:
    """接在检测蒙版之后：MediaPipe 检测双眼位置 → 径向变形。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "要变形的图像（可接美颜链输出）"}),
                **EYE_SIZE_INPUT,
                "min_detection_confidence": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.9,
                        "step": 0.05,
                        "tooltip": "人脸/眼睛关键点检测阈值。过高(如0.9)易检测失败、无效果；建议 0.5。",
                    },
                ),
                "min_presence_confidence": DETECT_INPUTS["min_presence_confidence"],
            },
            "optional": {
                "detect_image": (
                    "IMAGE",
                    {
                        "tooltip": "用于检测眼睛位置的图（建议接原图/上传人像）。不连则用 image。",
                    },
                ),
                "mask_edge_blur": ("INT", {"default": 9, "min": 0, "max": 31, "step": 2}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN", "IMAGE")
    RETURN_NAMES = ("image", "eye_mask", "landmarks_detected", "debug_preview")
    FUNCTION = "apply"
    CATEGORY = "image/beauty"

    def apply(
        self,
        image,
        eye_size,
        naturalness,
        min_detection_confidence,
        min_presence_confidence,
        detect_image=None,
        mask_edge_blur=9,
    ):
        _require_batch1(image, "FaceSkinEyeSize")
        frame, detect_frame = _resolve_detect_frame(image, detect_image)

        out_bgr, eye_mask, ok = apply_eye_size_from_rgb(
            frame,
            eye_size,
            min_detection_confidence,
            min_presence_confidence,
            mask_edge_blur=int(mask_edge_blur),
            naturalness=naturalness,
            detect_rgb_uint8=detect_frame,
        )
        mask_t = torch.from_numpy(eye_mask.astype(np.float32)).unsqueeze(0)

        # Debug: green = detected eye region (if empty, detection/warp did not land on eyes)
        dbg_bgr = image_to_bgr_uint8(frame)
        if eye_mask.max() > 0:
            green = np.zeros_like(dbg_bgr)
            green[:, :, 1] = (np.clip(eye_mask, 0, 1) * 200).astype(np.uint8)
            dbg_bgr = cv2.addWeighted(dbg_bgr, 0.65, green, 0.35, 0)
        dbg_t = _tensor_from_bgr(dbg_bgr)

        if abs(eye_size) < 1e-6:
            return (image, mask_t, ok, dbg_t)
        return (_tensor_from_bgr(out_bgr), mask_t, ok, dbg_t)


class FaceSkinFaceSlim:
    """MediaPipe 检测下颌/面颊 → 仅在脸部 ROI 内水平收缩（瘦脸）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "要变形的图像（建议接检测节点输出或原图）"}),
                **FACE_SLIM_INPUT,
                "min_detection_confidence": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.9,
                        "step": 0.05,
                        "tooltip": "人脸关键点检测阈值。过高易检测失败；建议 0.3–0.5。",
                    },
                ),
                "min_presence_confidence": DETECT_INPUTS["min_presence_confidence"],
            },
            "optional": {
                "detect_image": (
                    "IMAGE",
                    {"tooltip": "用于检测脸型的图（建议接原图/上传人像）。不连则用 image。"},
                ),
                "mask_edge_blur": ("INT", {"default": 11, "min": 0, "max": 31, "step": 2}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN", "IMAGE")
    RETURN_NAMES = ("image", "slim_mask", "landmarks_detected", "debug_preview")
    FUNCTION = "apply"
    CATEGORY = "image/beauty"

    def apply(
        self,
        image,
        face_slim,
        jaw_slim,
        naturalness,
        min_detection_confidence,
        min_presence_confidence,
        detect_image=None,
        mask_edge_blur=11,
    ):
        _require_batch1(image, "FaceSkinFaceSlim")
        frame, detect_frame = _resolve_detect_frame(image, detect_image)

        out_bgr, slim_mask, ok = apply_face_slim_from_rgb(
            frame,
            face_slim,
            jaw_slim,
            min_detection_confidence,
            min_presence_confidence,
            mask_edge_blur=int(mask_edge_blur),
            naturalness=naturalness,
            detect_rgb_uint8=detect_frame,
        )
        mask_t = torch.from_numpy(slim_mask.astype(np.float32)).unsqueeze(0)

        # Debug: show warp result with cheek mask tint (not original+overlay paste)
        dbg_bgr = out_bgr if (ok and (abs(face_slim) > 1e-6 or abs(jaw_slim) > 1e-6)) else image_to_bgr_uint8(frame)
        if slim_mask.max() > 0:
            magenta = np.zeros_like(dbg_bgr)
            magenta[:, :, 2] = (np.clip(slim_mask, 0, 1) * 160).astype(np.uint8)
            m3 = (np.clip(slim_mask, 0, 1) * 0.35)[:, :, np.newaxis]
            dbg_bgr = np.clip(
                dbg_bgr.astype(np.float32) * (1.0 - m3) + magenta.astype(np.float32) * m3,
                0,
                255,
            ).astype(np.uint8)
        dbg_t = _tensor_from_bgr(dbg_bgr)

        if abs(face_slim) < 1e-6 and abs(jaw_slim) < 1e-6:
            return (image, mask_t, ok, dbg_t)
        return (_tensor_from_bgr(out_bgr), mask_t, ok, dbg_t)


class FaceSkinEyeBeautyCommercial:
    """商业眼美颜：大眼 + 亮瞳 + 眼白 + 眼神光 + 卧蚕 + 祛黑眼圈 + 锐化（一键）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                **COMMERCIAL_EYE_INPUTS,
                "min_detection_confidence": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.1, "max": 0.9, "step": 0.05, "tooltip": "建议 0.3–0.5"},
                ),
                "min_presence_confidence": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.1, "max": 0.9, "step": 0.05, "tooltip": "建议 0.3–0.5"},
                ),
            },
            "optional": {
                "detect_image": ("IMAGE", {"tooltip": "检测用原图（建议接上传人像）"}),
                "mask_edge_blur": ("INT", {"default": 9, "min": 0, "max": 31, "step": 2}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "BOOLEAN", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "image",
        "eye_mask",
        "under_eye_mask",
        "landmarks_detected",
        "debug_preview",
        "mask_overlay_preview",
    )
    FUNCTION = "apply"
    CATEGORY = "image/beauty"

    def apply(
        self,
        image,
        intensity,
        eye_size,
        eye_open,
        naturalness,
        bright_eyes,
        whiten_sclera,
        catchlight,
        lash_line,
        aegyo_sal,
        under_eye,
        sharpen,
        min_detection_confidence,
        min_presence_confidence,
        detect_image=None,
        mask_edge_blur=9,
    ):
        _require_batch1(image, "FaceSkinEyeBeautyCommercial")
        frame, detect_frame = _resolve_detect_frame(image, detect_image)
        params = CommercialEyeParams(
            intensity=intensity,
            eye_size=eye_size,
            eye_open=eye_open,
            naturalness=naturalness,
            bright_eyes=bright_eyes,
            whiten_sclera=whiten_sclera,
            catchlight=catchlight,
            lash_line=lash_line,
            aegyo_sal=aegyo_sal,
            under_eye=under_eye,
            sharpen=sharpen,
            mask_edge_blur=int(mask_edge_blur),
        )
        out_bgr, eye_m, under_m, ok = apply_commercial_eye_beauty_from_rgb(
            frame,
            params,
            min_detection_confidence,
            min_presence_confidence,
            detect_rgb_uint8=detect_frame,
        )
        dbg = build_commercial_debug_preview(frame, out_bgr, eye_m, under_m, ok)
        dbg_t = _tensor_from_bgr(dbg)

        base_bgr = image_to_bgr_uint8(frame)
        if ok and eye_m.max() > 0:
            green = np.zeros_like(base_bgr)
            green[:, :, 1] = (np.clip(eye_m, 0, 1) * 200).astype(np.uint8)
            mask_vis = cv2.addWeighted(base_bgr, 0.6, green, 0.4, 0)
        else:
            mask_vis = base_bgr.copy()
            cv2.putText(
                mask_vis,
                "NO eye_mask",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        mask_vis_t = _tensor_from_bgr(mask_vis)

        z = torch.zeros((1, frame.shape[0], frame.shape[1]), dtype=torch.float32)
        if not ok:
            return (image, z, z, False, dbg_t, mask_vis_t)
        return (
            _tensor_from_bgr(out_bgr),
            torch.from_numpy(eye_m.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(under_m.astype(np.float32)).unsqueeze(0),
            True,
            dbg_t,
            mask_vis_t,
        )


class FaceSkinEyeBeautyPro(FaceSkinEyeBeautyCommercial):
    """v2.2+ 商业眼美颜（新 class id，避免工作流缓存旧 4 输出口）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return FaceSkinEyeBeautyCommercial.INPUT_TYPES()

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "BOOLEAN", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "image",
        "eye_mask",
        "under_eye_mask",
        "landmarks_detected",
        "debug_preview",
        "mask_overlay_preview",
    )
    FUNCTION = "apply"
    CATEGORY = "image/beauty"


class FaceSkinEyeMaskPreview:
    """把 eye_mask / under_eye_mask 叠到图上，方便排查检测（接 1c 的 MASK 输出）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "eye_mask": ("MASK",),
                "under_eye_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay",)
    FUNCTION = "preview"
    CATEGORY = "image/beauty"

    def preview(self, image, eye_mask=None, under_eye_mask=None):
        _require_batch1(image, "FaceSkinEyeMaskPreview")
        frame = _frame_uint8(image)
        bgr = image_to_bgr_uint8(frame)
        h, w = bgr.shape[:2]
        out = bgr.astype(np.float32)

        if eye_mask is not None:
            em = _mask_numpy(eye_mask, (h, w))
            green = np.zeros_like(bgr, dtype=np.float32)
            green[:, :, 1] = em * 220.0
            m3 = em[:, :, np.newaxis]
            out = out * (1.0 - m3 * 0.45) + green * (m3 * 0.45)

        if under_eye_mask is not None:
            um = _mask_numpy(under_eye_mask, (h, w))
            cyan = np.zeros_like(bgr, dtype=np.float32)
            cyan[:, :, 0] = um * 200.0
            cyan[:, :, 1] = um * 200.0
            m3 = um[:, :, np.newaxis]
            out = out * (1.0 - m3 * 0.35) + cyan * (m3 * 0.35)

        if eye_mask is None and under_eye_mask is None:
            u8 = np.clip(out, 0, 255).astype(np.uint8)
            cv2.putText(
                u8,
                "Connect eye_mask from 1c",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return (_tensor_from_bgr(u8),)

        return (_tensor_from_bgr(np.clip(out, 0, 255).astype(np.uint8)),)


class FaceSkinRegionMask:
    """Single region mask preview (same geometry as detect)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "region": (["face", "under_eye", "nasolabial", "cheek"],),
                **DETECT_INPUTS,
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "preview"
    CATEGORY = "image/beauty"

    def preview(
        self,
        image,
        region,
        skin_mask_mode,
        padding_percent,
        mask_edge_blur,
        min_detection_confidence,
        min_presence_confidence,
        fallback_center_if_no_face,
    ):
        _require_batch1(image, "FaceSkinRegionMask")
        frame = _frame_uint8(image)
        masks = detect_masks_from_rgb_frame(
            frame,
            padding_percent,
            mask_edge_blur,
            min_detection_confidence,
            min_presence_confidence,
            fallback_center_if_no_face,
            skin_mask_mode=skin_mask_mode,
        )
        return (torch.from_numpy(masks[region]).unsqueeze(0),)


class FaceSkinRegionMaskPreview:
    """Region mask + colored overlay preview (nasolabial = yellow lines on face)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "region": (["face", "under_eye", "nasolabial", "cheek"],),
                **DETECT_INPUTS,
                "overlay_alpha": (
                    "FLOAT",
                    {"default": 0.85, "min": 0.3, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("overlay_preview", "mask")
    FUNCTION = "preview"
    CATEGORY = "image/beauty"

    def preview(
        self,
        image,
        region,
        skin_mask_mode,
        padding_percent,
        mask_edge_blur,
        min_detection_confidence,
        min_presence_confidence,
        fallback_center_if_no_face,
        overlay_alpha=0.85,
    ):
        _require_batch1(image, "FaceSkinRegionMaskPreview")
        frame = _frame_uint8(image)
        masks = detect_masks_from_rgb_frame(
            frame,
            padding_percent,
            mask_edge_blur,
            min_detection_confidence,
            min_presence_confidence,
            fallback_center_if_no_face,
            skin_mask_mode=skin_mask_mode,
        )
        m = masks[region]
        bgr = image_to_bgr_uint8(frame)
        color = _MASK_OVERLAY_COLORS[region][0]
        display_m = np.clip(m, 0.0, 1.0)
        if region == "nasolabial" and display_m.max() > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dm = cv2.dilate((display_m * 255).astype(np.uint8), k, iterations=1)
            display_m = dm.astype(np.float32) / 255.0
        alpha = float(np.clip(overlay_alpha, 0.3, 1.0))
        out = _blend_color_mask(bgr, display_m, color, alpha)
        cov = float(m.mean()) * 100.0
        out = _draw_mask_legend(out, [f"{region} {cov:.2f}%", "cyan/yellow = mask area"])
        preview = torch.from_numpy(bgr_to_rgb_float01(out)).unsqueeze(0)
        return preview, torch.from_numpy(m.astype(np.float32)).unsqueeze(0)


class FaceSkinMaskOverlayPreview:
    """Overlay an existing MASK on image (for nasolabial preview from DetectMasks)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "overlay_alpha": (
                    "FLOAT",
                    {"default": 0.88, "min": 0.3, "max": 1.0, "step": 0.05},
                ),
                "label": ("STRING", {"default": "nasolabial"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay_preview",)
    FUNCTION = "preview"
    CATEGORY = "image/beauty"

    def preview(self, image, mask, overlay_alpha=0.88, label="nasolabial"):
        _require_batch1(image, "FaceSkinMaskOverlayPreview")
        frame = _frame_uint8(image)
        h, w = frame.shape[:2]
        m = _mask_numpy(mask, (h, w))
        bgr = image_to_bgr_uint8(frame)
        color = (80, 220, 220)
        display_m = np.clip(m, 0.0, 1.0)
        if display_m.max() > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dm = cv2.dilate((display_m * 255).astype(np.uint8), k, iterations=1)
            display_m = dm.astype(np.float32) / 255.0
        alpha = float(np.clip(overlay_alpha, 0.3, 1.0))
        out = _blend_color_mask(bgr, display_m, color, alpha)
        cov = float(m.mean()) * 100.0
        out = _draw_mask_legend(out, [f"{label} {cov:.2f}%", "cyan = nasolabial mask"])
        return (torch.from_numpy(bgr_to_rgb_float01(out)).unsqueeze(0),)


# BGR overlay colors + legend labels (ASCII — cv2.putText cannot render CJK)
_MASK_OVERLAY_COLORS = {
    "face": ((80, 220, 80), "face"),
    "under_eye": ((220, 200, 80), "under_eye"),
    "cheek": ((200, 80, 220), "cheek/plump"),
    "nasolabial": ((80, 220, 220), "nasolabial"),
}
_MASK_OVERLAY_ORDER = ("face", "under_eye", "cheek", "nasolabial")


def _draw_mask_legend(bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    if not lines:
        return bgr
    h, w = bgr.shape[:2]
    bar_h = min(28 + 18 * len(lines), h // 3)
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    for i, text in enumerate(lines):
        y = h - bar_h + 20 + i * 18
        cv2.putText(
            overlay,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return overlay


def _blend_color_mask(bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple, alpha: float) -> np.ndarray:
    m = np.clip(mask, 0.0, 1.0)[:, :, np.newaxis]
    color = np.array(color_bgr, dtype=np.float32)
    base = bgr.astype(np.float32)
    return np.clip(base * (1.0 - m * alpha) + color * (m * alpha), 0, 255).astype(np.uint8)


class FaceSkinMaskCheckPreview:
    """Overlay face region masks on image for inspection (蒙版检查)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "overlay_alpha": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "min": 0.1,
                        "max": 0.9,
                        "step": 0.05,
                        "tooltip": "蒙版颜色叠加强度",
                    },
                ),
                "show_legend": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "face_mask": ("MASK",),
                "under_eye_mask": ("MASK",),
                "nasolabial_mask": ("MASK",),
                "cheek_mask": ("MASK",),
                **DETECT_INPUTS,
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = (
        "overlay_preview",
        "face_mask",
        "under_eye_mask",
        "nasolabial_mask",
        "cheek_mask",
        "union_mask",
    )
    FUNCTION = "check"
    CATEGORY = "image/beauty"

    def check(
        self,
        image,
        overlay_alpha,
        show_legend,
        face_mask=None,
        under_eye_mask=None,
        nasolabial_mask=None,
        cheek_mask=None,
        skin_mask_mode="semantic",
        padding_percent=0.06,
        mask_edge_blur=15,
        min_detection_confidence=0.5,
        min_presence_confidence=0.5,
        fallback_center_if_no_face=False,
    ):
        _require_batch1(image, "FaceSkinMaskCheckPreview")
        frame = _frame_uint8(image)
        h, w = frame.shape[:2]
        shape_hw = (h, w)

        if face_mask is None:
            masks = detect_masks_from_rgb_frame(
                frame,
                padding_percent,
                mask_edge_blur,
                min_detection_confidence,
                min_presence_confidence,
                fallback_center_if_no_face,
                skin_mask_mode=skin_mask_mode,
            )
            face_m = masks["face"]
            under_m = masks["under_eye"]
            naso_m = masks["nasolabial"]
            cheek_m = masks["cheek"]
        else:
            face_m = _mask_numpy(face_mask, shape_hw)
            under_m = _mask_numpy(under_eye_mask, shape_hw) if under_eye_mask is not None else np.zeros(shape_hw, np.float32)
            naso_m = _mask_numpy(nasolabial_mask, shape_hw) if nasolabial_mask is not None else np.zeros(shape_hw, np.float32)
            cheek_m = _mask_numpy(cheek_mask, shape_hw) if cheek_mask is not None else np.zeros(shape_hw, np.float32)

        bgr = image_to_bgr_uint8(frame)
        alpha = float(np.clip(overlay_alpha, 0.1, 0.9))
        out = bgr.copy()
        stats = []
        mask_map = {"face": face_m, "under_eye": under_m, "nasolabial": naso_m, "cheek": cheek_m}
        for key in _MASK_OVERLAY_ORDER:
            color, label = _MASK_OVERLAY_COLORS[key]
            m = mask_map[key]
            cov = float(m.mean()) * 100.0
            stats.append(f"{label} {cov:.1f}%")
            key_alpha = min(0.9, alpha * 1.3) if key == "nasolabial" else alpha
            out = _blend_color_mask(out, m, color, key_alpha)

        union = np.clip(np.maximum.reduce([face_m, under_m, naso_m, cheek_m]), 0.0, 1.0)
        if show_legend:
            stats.append(f"union {float(union.mean()) * 100:.1f}%")
            out = _draw_mask_legend(out, stats)

        preview = torch.from_numpy(bgr_to_rgb_float01(out)).unsqueeze(0)

        def mask_t(arr):
            return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)

        return (
            preview,
            mask_t(face_m),
            mask_t(under_m),
            mask_t(naso_m),
            mask_t(cheek_m),
            mask_t(union),
        )


class FaceSkinSpotRemove:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "amount": amount_slider(0, "祛斑祛痘 0–100"),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/beauty"

    def apply(self, image, mask, amount):
        return _apply_effect_node(image, mask, amount, "FaceSkinSpotRemove", apply_spot_remove)


class FaceSkinSmooth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "amount": amount_slider(0, "磨皮 0–100"),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/beauty"

    def apply(self, image, mask, amount):
        return _apply_effect_node(image, mask, amount, "FaceSkinSmooth", apply_smooth)


class FaceSkinEvenTone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "amount": amount_slider(0, "匀肤 0–100"),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/beauty"

    def apply(self, image, mask, amount):
        return _apply_effect_node(image, mask, amount, "FaceSkinEvenTone", apply_even_tone)


class FaceSkinNasolabial:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "amount": amount_slider(0, "祛法令纹 0–100"),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/beauty"

    def apply(self, image, mask, amount):
        return _apply_effect_node(image, mask, amount, "FaceSkinNasolabial", apply_nasolabial)


class FaceSkinDarkCircle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "amount": amount_slider(0, "祛黑眼圈 0–100"),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/beauty"

    def apply(self, image, mask, amount):
        return _apply_effect_node(image, mask, amount, "FaceSkinDarkCircle", apply_dark_circle)


class FaceSkinPlump:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "amount": amount_slider(0, "丰盈 0–100"),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/beauty"

    def apply(self, image, mask, amount):
        return _apply_effect_node(image, mask, amount, "FaceSkinPlump", apply_plump)


class FaceSkinWhiten:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "amount": amount_slider(0, "美白 0–100"),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/beauty"

    def apply(self, image, mask, amount):
        return _apply_effect_node(image, mask, amount, "FaceSkinWhiten", apply_whiten)


class FaceSkinClarity:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "amount": amount_slider(0, "清晰 0–100"),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/beauty"

    def apply(self, image, mask, amount):
        return _apply_effect_node(image, mask, amount, "FaceSkinClarity", apply_clarity)


class FaceSkinMaskSubtract:
    """从主蒙版减去子蒙版，用于磨皮时排除眼区：face_mask - eye_mask。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK", {"tooltip": "主蒙版，如 face_mask"}),
                "subtract_mask": ("MASK", {"tooltip": "要扣除的区域，如 1c 的 eye_mask"}),
                "feather": (
                    "INT",
                    {"default": 5, "min": 0, "max": 31, "step": 2, "tooltip": "扣除边缘羽化，避免硬边"},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "apply"
    CATEGORY = "image/beauty"

    def apply(self, base_mask, subtract_mask, feather=5):
        b = base_mask[0].detach().cpu().numpy().astype(np.float32)
        s = subtract_mask[0].detach().cpu().numpy().astype(np.float32)
        if b.shape != s.shape:
            s = cv2.resize(s, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_LINEAR)
        if feather > 0:
            k = int(feather) | 1
            s = cv2.GaussianBlur(s, (k, k), 0)
        out = np.clip(b - s, 0.0, 1.0)
        return (torch.from_numpy(out).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "FaceSkinDetectMasks": FaceSkinDetectMasks,
    "FaceSkinFaceSlim": FaceSkinFaceSlim,
    "FaceSkinEyeSize": FaceSkinEyeSize,
    "FaceSkinEyeBeautyPro": FaceSkinEyeBeautyPro,
    "FaceSkinEyeBeautyV2": FaceSkinEyeBeautyPro,
    "FaceSkinEyeBeautyCommercial": FaceSkinEyeBeautyPro,
    "FaceSkinEyeMaskPreview": FaceSkinEyeMaskPreview,
    "FaceSkinMaskCheckPreview": FaceSkinMaskCheckPreview,
    "FaceSkinRegionMask": FaceSkinRegionMask,
    "FaceSkinRegionMaskPreview": FaceSkinRegionMaskPreview,
    "FaceSkinMaskOverlayPreview": FaceSkinMaskOverlayPreview,
    "FaceSkinSpotRemove": FaceSkinSpotRemove,
    "FaceSkinSmooth": FaceSkinSmooth,
    "FaceSkinEvenTone": FaceSkinEvenTone,
    "FaceSkinNasolabial": FaceSkinNasolabial,
    "FaceSkinDarkCircle": FaceSkinDarkCircle,
    "FaceSkinPlump": FaceSkinPlump,
    "FaceSkinWhiten": FaceSkinWhiten,
    "FaceSkinClarity": FaceSkinClarity,
    "FaceSkinMaskSubtract": FaceSkinMaskSubtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceSkinDetectMasks": "1. 检测蒙版 Face Skin Detect",
    "FaceSkinFaceSlim": "1a. 瘦脸 Face Slim",
    "FaceSkinEyeSize": "1b. 眼睛大小 Eye Size",
    "FaceSkinEyeBeautyPro": "1c. 商业眼美颜 Eye Beauty (Commercial)",
    "FaceSkinEyeBeautyV2": "1c. 商业眼美颜 (alias)",
    "FaceSkinEyeBeautyCommercial": "1c. 商业眼美颜 (alias)",
    "FaceSkinEyeMaskPreview": "Eye Mask Preview 眼蒙版预览",
    "FaceSkinMaskCheckPreview": "Mask Check 蒙版检查",
    "FaceSkinRegionMask": "Face Skin Region Mask",
    "FaceSkinRegionMaskPreview": "Region Mask Preview 区域蒙版预览",
    "FaceSkinMaskOverlayPreview": "Mask Overlay Preview 蒙版叠加预览",
    "FaceSkinSpotRemove": "2. Spot Remove 祛斑祛痘",
    "FaceSkinSmooth": "3. Smooth 磨皮",
    "FaceSkinEvenTone": "4. Even Tone 匀肤",
    "FaceSkinNasolabial": "5. Nasolabial 祛法令纹",
    "FaceSkinDarkCircle": "6. Dark Circle 祛黑眼圈",
    "FaceSkinPlump": "7. Plump 丰盈",
    "FaceSkinWhiten": "8. Whiten 美白",
    "FaceSkinClarity": "9. Clarity 清晰",
    "FaceSkinMaskSubtract": "Mask Subtract 蒙版相减（护眼磨皮）",
}

# Legacy aliases
NODE_CLASS_MAPPINGS["FaceSkinBeautyPreviewMask"] = FaceSkinRegionMask
NODE_DISPLAY_NAME_MAPPINGS["FaceSkinBeautyPreviewMask"] = "Face Skin Region Mask (legacy)"
