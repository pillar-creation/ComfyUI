# 照片换脸工作流 (ReActor)

对应 ComfyUI 工作流：**`Photo Face Swap (ReActor).json`**

## 流程

```
待换脸照片 ──IMAGE──► ReActor ◄──IMAGE── 提供的人脸
                         │
                         ├──► 预览
                         └──► 保存 (output/ReActor_PhotoFaceSwap/)
```

| 节点 | 作用 |
|------|------|
| 1. 待换脸照片 | 底图：要被替换人脸的照片 |
| 2. 提供的人脸 | 参考图：换上去的人脸来源 |
| 3. ReActor 换脸 | inswapper_128 融合 + GFPGAN 清晰化 |
| 4–5. 预览 / 保存 | 查看并导出 PNG |

## 使用步骤

1. 在 ComfyUI 中 **Load** → 选择 `Photo Face Swap (ReActor).json`
2. **1. 待换脸照片**：上传或选择 `input` 目录中的底图
3. **2. 提供的人脸**：上传清晰正脸参考图（光线均匀、无遮挡）
4. 点击 **Queue Prompt** 运行
5. 结果保存在 `comfyui/output/ReActor_PhotoFaceSwap/`

## 多人脸

底图或参考图有多张脸时，在 ReActor 节点调整：

- `input_faces_index`：底图中要换的脸（默认 `0` 为检测到的第一张）
- `source_faces_index`：参考图中取哪张脸（默认 `0`）
- 多张可填 `0,1` 等（与 ReActor 文档一致）

## 模型与依赖

与 **视频换脸** 共用同一套 ReActor 模型，无需重复安装。详见：

- `Video Face Swap (ReActor + RIFE).md`

简要检查：

| 文件 | 路径 |
|------|------|
| `inswapper_128.onnx` | `comfyui/models/insightface/` |
| `buffalo_l` 各 `.onnx` | `comfyui/models/insightface/models/buffalo_l/` |
| `GFPGANv1.3.pth` 等 | `comfyui/models/facerestore_models/` |

未安装时运行：`custom_nodes/ComfyUI-ReActor/install.bat` 或 `install_deps_only.py`（见视频换脸文档）。

## 调参建议

| 参数 | 建议 |
|------|------|
| `face_restore_visibility` | 0.3–0.6，过高易“塑料感” |
| `codeformer_weight` | 0.5 左右 |
| `face_restore_model` | `GFPGANv1.3.pth` 或 `codeformer.pth` |
| 换脸不明显 | 换更清晰的正脸参考图；检查两张图是否都检测到脸 |

## 与视频换脸的区别

本工作流仅处理**单张或批量图片**（通过 ReActor 对单帧 `IMAGE`），无 RIFE 插帧、无视频合成，更轻量、适合照片批处理。

## 仅替换眼睛区域？

ReActor **没有**「只换眼」模式，inswapper 始终按整脸对齐替换。

可用 **蒙版合成** 实现近似效果：

1. 先整脸换脸得到 `SWAPPED_IMAGE`
2. 用 MediaPipe 生成**眼区蒙版**（本仓库 `ComfyUI-Simple-Face-Mask` 的 `FaceSkinEyeSize`，`eye_size=0` 只出蒙版、不变形）
3. 用 `ImageCompositeMasked` 或 `ReActorMaskHelper` 的 `mask_optional`，只把换脸图的眼区叠回**原底图**

专用工作流：**`Photo Face Swap Eyes Only (ReActor).json`**

| 注意 | 说明 |
|------|------|
| 原理 | 鼻、嘴、轮廓仍是原图，主要变化在虹膜/眼周 |
| 边界 | 蒙版过大会带到脸颊；调小 `mask_edge_blur` 或后续用 `ReActorMaskHelper` 精修 |
| 修复 | 眼区版默认 `face_restore_model=none`，避免整脸 GFPGAN 与局部合成冲突 |
