"""All-in-one skin beauty node (optional shortcut)."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .face_utils import bgr_to_rgb_float01, detect_masks_from_rgb_frame, image_to_bgr_uint8
from .skin_effects import process_skin_bgr
from .skin_nodes import amount_slider, DETECT_INPUTS


class FaceSkinBeauty:
    """Single-node skin management (all sliders in one node)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "smooth": amount_slider(0, "磨皮 0–100"),
                "whiten": amount_slider(0, "美白 0–100"),
                "even_tone": amount_slider(0, "匀肤 0–100"),
                "plump": amount_slider(0, "丰盈 0–100"),
                "spot_remove": amount_slider(0, "祛斑祛痘 0–100"),
                "nasolabial": amount_slider(0, "祛法令纹 0–100"),
                "dark_circle": amount_slider(0, "祛黑眼圈 0–100"),
                "clarity": amount_slider(0, "清晰 0–100"),
                **DETECT_INPUTS,
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "face_mask")
    FUNCTION = "apply"
    CATEGORY = "image/beauty"

    def apply(
        self,
        image,
        smooth,
        whiten,
        even_tone,
        plump,
        spot_remove,
        nasolabial,
        dark_circle,
        clarity,
        skin_mask_mode,
        padding_percent,
        mask_edge_blur,
        min_detection_confidence,
        min_presence_confidence,
        fallback_center_if_no_face,
        mask=None,
    ):
        if image.shape[0] != 1:
            raise ValueError("FaceSkinBeauty: batch size must be 1.")

        frame = (image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        bgr = image_to_bgr_uint8(frame)
        region_masks = detect_masks_from_rgb_frame(
            frame,
            padding_percent,
            mask_edge_blur,
            min_detection_confidence,
            min_presence_confidence,
            fallback_center_if_no_face,
            skin_mask_mode=skin_mask_mode,
        )

        if mask is not None:
            user_m = mask[0].detach().cpu().numpy().astype(np.float32)
            if user_m.shape != region_masks["face"].shape:
                user_m = cv2.resize(
                    user_m,
                    (region_masks["face"].shape[1], region_masks["face"].shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            for key in region_masks:
                region_masks[key] = np.clip(region_masks[key] * user_m, 0.0, 1.0)

        out_bgr = process_skin_bgr(
            bgr,
            region_masks,
            smooth=smooth,
            whiten=whiten,
            even_tone=even_tone,
            plump=plump,
            spot_remove=spot_remove,
            nasolabial=nasolabial,
            dark_circle=dark_circle,
            clarity=clarity,
        )
        out_t = torch.from_numpy(bgr_to_rgb_float01(out_bgr)).unsqueeze(0)
        mask_t = torch.from_numpy(region_masks["face"]).unsqueeze(0)
        return (out_t, mask_t)


NODE_CLASS_MAPPINGS = {
    "FaceSkinBeauty": FaceSkinBeauty,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceSkinBeauty": "Face Skin Beauty (All-in-One)",
}
