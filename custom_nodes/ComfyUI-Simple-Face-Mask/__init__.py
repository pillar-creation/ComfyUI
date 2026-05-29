"""ComfyUI-Simple-Face-Mask: MediaPipe landmarks + skin beauty nodes."""

try:
    import cv2  # noqa: F401
    import mediapipe  # noqa: F401
except ImportError as e:
    raise ImportError(
        "ComfyUI-Simple-Face-Mask requires OpenCV and MediaPipe. Install in ComfyUI's Python:\n"
        "  pip install opencv-python-headless mediapipe\n"
        f"Original error: {e}"
    ) from e

from .simple_face_mask import (
    NODE_CLASS_MAPPINGS as MASK_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MASK_NODE_DISPLAY_NAME_MAPPINGS,
)
from .skin_beauty import (
    NODE_CLASS_MAPPINGS as BEAUTY_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as BEAUTY_NODE_DISPLAY_NAME_MAPPINGS,
)
from .skin_nodes import (
    NODE_CLASS_MAPPINGS as STEP_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as STEP_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {
    **MASK_NODE_CLASS_MAPPINGS,
    **BEAUTY_NODE_CLASS_MAPPINGS,
    **STEP_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **MASK_NODE_DISPLAY_NAME_MAPPINGS,
    **BEAUTY_NODE_DISPLAY_NAME_MAPPINGS,
    **STEP_NODE_DISPLAY_NAME_MAPPINGS,
}

print("[ComfyUI-Simple-Face-Mask] v2.3.4 — clip nose side + taper inner canthus")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
