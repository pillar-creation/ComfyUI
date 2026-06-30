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

try:
    from .reactor_decomposed import (
        NODE_CLASS_MAPPINGS as REACTOR_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as REACTOR_NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError as _reactor_import_err:
    REACTOR_NODE_CLASS_MAPPINGS = {}
    REACTOR_NODE_DISPLAY_NAME_MAPPINGS = {}
    print(f"[ComfyUI-Simple-Face-Mask] ReActor split nodes disabled: {_reactor_import_err}")

NODE_CLASS_MAPPINGS = {
    **MASK_NODE_CLASS_MAPPINGS,
    **BEAUTY_NODE_CLASS_MAPPINGS,
    **STEP_NODE_CLASS_MAPPINGS,
    **REACTOR_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **MASK_NODE_DISPLAY_NAME_MAPPINGS,
    **BEAUTY_NODE_DISPLAY_NAME_MAPPINGS,
    **STEP_NODE_DISPLAY_NAME_MAPPINGS,
    **REACTOR_NODE_DISPLAY_NAME_MAPPINGS,
}

print("[ComfyUI-Simple-Face-Mask] v2.5.6 — temple trim + hair dilate at hairline")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
