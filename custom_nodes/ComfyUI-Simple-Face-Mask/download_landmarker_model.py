"""Download MediaPipe models into models/. Run with ComfyUI Python."""
from face_utils import _LANDMARKER_PATH, _SEGMENTER_PATH, ensure_landmarker_model, ensure_segmenter_model

if __name__ == "__main__":
    p1 = ensure_landmarker_model()
    print("landmarker OK:", p1, p1.stat().st_size)
    p2 = ensure_segmenter_model()
    print("segmenter OK:", p2, p2.stat().st_size)
