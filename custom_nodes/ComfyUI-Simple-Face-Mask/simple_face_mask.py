"""Face skin mask (semantic segmentation + landmarks)."""

import torch

from .face_utils import detect_masks_from_rgb_frame
from .skin_nodes import DETECT_INPUTS


class SimpleFaceMaskFromImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), **DETECT_INPUTS}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "make_mask"
    CATEGORY = "image/mask"

    def make_mask(
        self,
        image,
        skin_mask_mode,
        padding_percent,
        mask_edge_blur,
        min_detection_confidence,
        min_presence_confidence,
        fallback_center_if_no_face,
    ):
        if image.shape[0] != 1:
            raise ValueError("SimpleFaceMaskFromImage: batch size must be 1.")

        frame = (image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        masks = detect_masks_from_rgb_frame(
            frame,
            padding_percent,
            mask_edge_blur,
            min_detection_confidence,
            min_presence_confidence,
            fallback_center_if_no_face,
            skin_mask_mode=skin_mask_mode,
        )
        return (torch.from_numpy(masks["face"]).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"SimpleFaceMaskFromImage": SimpleFaceMaskFromImage}
NODE_DISPLAY_NAME_MAPPINGS = {"SimpleFaceMaskFromImage": "Face Skin Mask (Semantic)"}
