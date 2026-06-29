# ComfyUI-Simple-Face-Mask

MediaPipe 人脸关键点 + OpenCV 蒙版与美肤节点，用于 ComfyUI。

## v2.3.6 眼周蒙版

- 眼裂多边形（按眼睑 landmark 顺序）
- 上/下眼皮皮肤带、上睑脊线、睫毛描边
- 内眼角安全外扩（远离鼻梁）、外眼角垫片
- 鼻侧裁剪 + 边缘柔化

## 依赖

```bash
pip install -r requirements.txt
```

首次运行会自动下载 MediaPipe `face_landmarker` 模型到 `models/`。

## 节点

- `FaceSkinFaceSlim` — 瘦脸（面颊 + 可选下颌，MediaPipe 水平收缩）
- `FaceSkinEyeSize` / `FaceSkinEyeMaskPreview` — 大眼 / 眼周蒙版
- 美肤、眼部美化等节点见 `skin_nodes.py`、`eye_effects.py`

## 工作流

见 [comfyui-workflows](https://github.com/pillar-creation/comfyui-workflows) 仓库中的 Photo Face Swap 系列。
