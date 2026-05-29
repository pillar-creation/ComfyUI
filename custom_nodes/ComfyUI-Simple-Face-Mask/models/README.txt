Required models (auto-download on first run):

1) face_landmarker.task (~3.8MB)
   https://hf-mirror.com/spacepxl/FLAME/resolve/main/SMIRK/face_landmarker.task

2) selfie_multiclass_256x256.tflite (~16MB) — semantic skin / hair / face-skin
   https://hf-mirror.com/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite

Env overrides:
  COMFYUI_FACE_LANDMARKER_MODEL
  COMFYUI_SELFIE_SEGMENTER_MODEL

Run: python download_landmarker_model.py
