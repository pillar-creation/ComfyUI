# 视频换脸工作流 (ReActor + RIFE)

对应 ComfyUI 工作流：`Video Face Swap (ReActor + RIFE).json`

## 子图版（分步可视化）

加载 **`Video Face Swap (ReActor + RIFE) - ReActor子图.json`**：

- 主画布上的 **「3. ReActor 换脸（子图·可展开）」** 双击或点击展开，可看到：
  1. **① 换脸**（`face_restore_model = none`，仅 inswapper）
  2. **预览①** 仅换脸结果
  3. **② Restore Face**（GFPGAN 修复）
  4. **预览②** 最终结果
- **检测 / NSFW** 仍在 ① 节点内部，ReActor 未提供独立节点，无法再拆细。

## 流程

```
Load Video ──IMAGE──► ReActor ◄──IMAGE── Load Image (目标脸)
                         │
                         ▼
                    RIFE VFI (×2)
                         │
Load Video ──audio───────┼──► Video Combine ◄── FPS×2
         └──video_info──► Video Info Loaded ──► JWFloatMul(×2)
```

## 已安装的自定义节点

| 节点包 | 目录 |
|--------|------|
| Video Helper Suite | `ComfyUI-VideoHelperSuite`（已有） |
| ReActor | `ComfyUI-ReActor`（已克隆） |
| Frame Interpolation | `ComfyUI-Frame-Interpolation`（已克隆） |
| Float Multiply | `comfyui-various`（已有，用于 FPS×2） |

## 首次安装步骤

### 1. ReActor 依赖与模型

**若 `install.py` 报 SSL / 连接 HuggingFace 失败**（国内网络常见），请分两步：

**A. 仅安装 Python 包**（模型已手动放好后再跑，否则会提示缺模型）：

```powershell
conda activate comfyui
cd d:\workspace\ai\comfyui\custom_nodes\ComfyUI-ReActor
python install_deps_only.py
```

或只装依赖、不检查模型：

```powershell
pip install -r requirements.txt onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**B. 手动下载换脸模型**（约 554MB），保存为：

`d:\workspace\ai\comfyui\models\insightface\inswapper_128.onnx`

| 来源 | 链接 |
|------|------|
| 镜像站 | https://hf-mirror.com/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx |
| 官方 | https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx |

浏览器、IDM 或先 `set HF_ENDPOINT=https://hf-mirror.com` 再 `huggingface-cli download` 均可。

正常网络下也可直接：`python install.py` 或 `.\install.bat`（提示 Y 时输入 Y）。

**面部修复模型**（放 `comfyui/models/facerestore_models/`）：

| 文件 | 说明 |
|------|------|
| `GFPGANv1.3.pth` | 工作流默认（你本地已有可选此项） |
| `codeformer-v0.1.0.pth` | 可选，效果常更好 |

镜像下载：https://hf-mirror.com/datasets/Gourieff/ReActor/tree/main/models/facerestore_models

### 2. RIFE 模型

创建目录并下载权重：

```powershell
New-Item -ItemType Directory -Force -Path "d:\workspace\ai\comfyui\custom_nodes\ComfyUI-Frame-Interpolation\ckpts\rife"
```

将 `rife49.pth` 放入上述 `ckpts/rife/`（工作流默认；也可用 `rife47.pth`）。

或运行 Frame-Interpolation 的安装脚本（若仓库提供 `install.bat`）。

### 3. 重启 ComfyUI

安装完成后**完全重启** ComfyUI，在菜单中加载工作流 JSON。

## 使用说明

1. **Load Video**：在 `comfyui/input/` 放入视频，或节点内上传；长视频建议设置 `frame_load_cap`（如 64）和 `skip_first_frames` 分段跑。
2. **Load Image**：上传一张清晰正脸作为换脸源。
3. **ReActor**：`input_faces_index` / `source_faces_index` 默认 `0`（第一张脸）；多人脸可改为 `0,1` 等。
4. **RIFE**：`multiplier=2` 使帧数翻倍；输出帧率已通过 `FPS × 2` 自动匹配，避免播放过快。
5. **Video Combine**：勾选 `save_output`；输出在 `comfyui/output/ReActor_FaceSwap_*.mp4`。

## 内存 / 显存建议

ReActor 会把 **整段视频的帧一次性** 转成 CPU 张量。帧数 × 分辨率过大时会报：

`DefaultCPUAllocator: not enough memory`（例如一次申请约 4.6GB）。

| 情况 | 建议设置（Load Video 节点） |
|------|---------------------------|
| 先试跑通 | `frame_load_cap` = **32～48**，`custom_width` = **960**（工作流已默认） |
| 仍 OOM | 改为 **24** 帧，或宽 **720** |
| 长视频全长 | 分段：`skip_first_frames` 0→48→96…，每段 `frame_load_cap` 48，最后拼接 |

**估算**：1080p 单帧约 25MB（float32 张量）；48 帧约 1.2GB，再加 RIFE/换脸开销，建议系统内存 **16GB+**。

## 故障排除

| 问题 | 处理 |
|------|------|
| 找不到 ReActor / RIFE 节点 | 重启 ComfyUI；确认 `custom_nodes` 下文件夹存在 |
| `inswapper_128.onnx` 缺失 | 运行 `ComfyUI-ReActor/install.bat` |
| **运行时报 `SSL: UNEXPECTED_EOF`（ReActorFaceSwap）** | **首次运行会联网下 `buffalo_l`，必须改离线：** 见下方 |
| RIFE 报错找不到 ckpt | 确认 `rife49.pth` 在 `ComfyUI-Frame-Interpolation/ckpts/rife/` |
| 音画不同步 | 确认 `Video Combine` 的 `frame_rate` 已连接 `FPS × 2` |
| OOM | 减小 `frame_load_cap` 或降低视频分辨率 |

### 运行换脸时 SSL 报错（最常见）

报错出现在 **执行队列** 而不是安装时：ReActor 发现缺少 **buffalo_l** 人脸模型，会用 `urllib` 从 HuggingFace 下载 `buffalo_l.zip`，网络不稳就会 SSL 失败。

**离线准备（推荐）：**

```powershell
cd D:\workspace\ai\comfyui\custom_nodes\ComfyUI-ReActor
powershell -ExecutionPolicy Bypass -File .\download_models_offline.ps1
```

或浏览器手动下载并解压：

| 文件 | 保存位置 |
|------|----------|
| `inswapper_128.onnx` | `comfyui\models\insightface\` |
| `buffalo_l.zip` 解压后所有 `.onnx` | `comfyui\models\insightface\models\buffalo_l\` |

镜像页：https://hf-mirror.com/datasets/Gourieff/ReActor/tree/main/models

解压后必须存在：`models\insightface\models\buffalo_l\det_10g.onnx`（以及 `w600k_r50.onnx`、`genderage.onnx` 等）。

完成后 **重启 ComfyUI** 再点运行。
