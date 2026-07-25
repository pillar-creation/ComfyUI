"""ComfyUI-Simple-Face-Mask: MediaPipe landmarks + skin beauty nodes."""

import os
import threading

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


def _start_mediapipe_preload() -> None:
    skip = os.environ.get("COMFYUI_FACE_MASK_SKIP_MEDIAPIPE_PRELOAD", "").lower()
    if skip in ("1", "true", "yes"):
        return

    def _run() -> None:
        try:
            from .face_utils import preload_mediapipe

            preload_mediapipe()
        except Exception as exc:
            print(
                "[ComfyUI-Simple-Face-Mask] MediaPipe preload failed "
                f"(will retry on first use): {exc}"
            )

    sync = os.environ.get("COMFYUI_FACE_MASK_SYNC_MEDIAPIPE_PRELOAD", "").lower()
    if sync in ("1", "true", "yes"):
        _run()
        return

    threading.Thread(
        target=_run,
        name="ComfyUI-Simple-Face-Mask-mediapipe-preload",
        daemon=True,
    ).start()
    print("[ComfyUI-Simple-Face-Mask] MediaPipe preload started (background)")


_start_mediapipe_preload()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
