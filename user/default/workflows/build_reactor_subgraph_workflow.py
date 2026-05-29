"""Generate video face-swap workflow with expandable ReActor subgraph."""
import json
import uuid

SUBGRAPH_ID = "b2c3d4e5-f6a7-8901-bcde-reactor-steps01"
OUT = r"d:\workspace\ai\comfyui\user\default\workflows\Video Face Swap (ReActor + RIFE) - ReActor子图.json"
SRC = r"d:\workspace\ai\comfyui\user\default\workflows\Video Face Swap (ReActor + RIFE).json"

SWAP_WIDGETS = [True, "inswapper_128.onnx", "retinaface_resnet50", "none", 1.0, 0.5, "no", "no", "0", "0", 1]
RESTORE_WIDGETS = ["retinaface_resnet50", "GFPGANv1.3.pth", 0.5, 0.5]


def sg_node(nid, ntype, pos, title, inputs=None, outputs=None, widgets=None, size=None):
    n = {
        "id": nid,
        "type": ntype,
        "pos": pos,
        "size": size or [280, 120],
        "flags": {},
        "order": nid,
        "mode": 0,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "title": title,
        "properties": {"Node name for S&R": ntype},
    }
    if widgets is not None:
        n["widgets_values"] = widgets
    return n


def mk_inp(name, typ, link=None, widget=None):
    d = {"name": name, "type": typ, "link": link}
    if widget:
        d["widget"] = {"name": widget}
    return d


def mk_out(name, typ, links=None):
    return {"name": name, "type": typ, "links": links or []}


# --- Subgraph internal nodes ---
sg_nodes = [
    sg_node(
        101,
        "MarkdownNote",
        [-40, -280],
        "换脸内部步骤说明",
        size=[520, 200],
        widgets=[
            "## ReActor 内部分步（子图）\n\n"
            "| 步骤 | 节点 | 说明 |\n"
            "|------|------|------|\n"
            "| ① | 换脸(无修复) | RetinaFace 检测 + inswapper_128 融合 |\n"
            "| 预览① | Preview | 仅换脸、未 GFPGAN |\n"
            "| ② | Restore Face | GFPGAN 人脸清晰化 |\n"
            "| 预览② | Preview | 最终输出 |\n\n"
            "NSFW 检测、buffalo 嵌入仍在 ① 节点**内部**，ReActor 未提供独立节点。"
        ],
    ),
    sg_node(
        102,
        "ReActorFaceSwap",
        [-40, -20],
        "① 换脸 (inswapper, 无修复)",
        size=[320, 360],
        inputs=[
            mk_inp("input_image", "IMAGE", 205),
            mk_inp("source_image", "IMAGE", 206),
            mk_inp("face_model", "FACE_MODEL", None),
            mk_inp("face_boost", "FACE_BOOST", None),
            mk_inp("enabled", "BOOLEAN", None, "enabled"),
            mk_inp("swap_model", "COMBO", None, "swap_model"),
            mk_inp("facedetection", "COMBO", None, "facedetection"),
            mk_inp("face_restore_model", "COMBO", None, "face_restore_model"),
            mk_inp("face_restore_visibility", "FLOAT", None, "face_restore_visibility"),
            mk_inp("codeformer_weight", "FLOAT", None, "codeformer_weight"),
            mk_inp("detect_gender_input", "COMBO", None, "detect_gender_input"),
            mk_inp("detect_gender_source", "COMBO", None, "detect_gender_source"),
            mk_inp("input_faces_index", "STRING", None, "input_faces_index"),
            mk_inp("source_faces_index", "STRING", None, "source_faces_index"),
            mk_inp("console_log_level", "COMBO", None, "console_log_level"),
        ],
        outputs=[
            mk_out("SWAPPED_IMAGE", "IMAGE", [201]),
            mk_out("FACE_MODEL", "FACE_MODEL", None),
            mk_out("ORIGINAL_IMAGE", "IMAGE", [202]),
        ],
        widgets=SWAP_WIDGETS,
    ),
    sg_node(
        103,
        "PreviewImage",
        [360, -20],
        "预览① 仅换脸",
        size=[280, 280],
        inputs=[mk_inp("images", "IMAGE", 201)],
    ),
    sg_node(
        104,
        "ReActorRestoreFace",
        [-40, 400],
        "② 人脸修复 (GFPGAN)",
        size=[300, 180],
        inputs=[mk_inp("image", "IMAGE", 201)],
        outputs=[mk_out("IMAGE", "IMAGE", [203, 204])],
        widgets=RESTORE_WIDGETS,
    ),
    sg_node(
        105,
        "PreviewImage",
        [360, 400],
        "预览② 修复后",
        size=[280, 280],
        inputs=[mk_inp("images", "IMAGE", 203)],
    ),
    sg_node(
        106,
        "PreviewImage",
        [360, 120],
        "预览 原图(对照)",
        size=[280, 200],
        inputs=[mk_inp("images", "IMAGE", 202)],
    ),
]

sg_links = [
    {"id": 201, "origin_id": 102, "origin_slot": 0, "target_id": 103, "target_slot": 0, "type": "IMAGE"},
    {"id": 202, "origin_id": 102, "origin_slot": 2, "target_id": 106, "target_slot": 0, "type": "IMAGE"},
    {"id": 203, "origin_id": 104, "origin_slot": 0, "target_id": 105, "target_slot": 0, "type": "IMAGE"},
    {"id": 204, "origin_id": 104, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "IMAGE"},
    {"id": 205, "origin_id": -10, "origin_slot": 0, "target_id": 102, "target_slot": 0, "type": "IMAGE"},
    {"id": 206, "origin_id": -10, "origin_slot": 1, "target_id": 102, "target_slot": 1, "type": "IMAGE"},
]

subgraph = {
    "id": SUBGRAPH_ID,
    "version": 1,
    "state": {"lastNodeId": 106, "lastLinkId": 206, "lastGroupId": 1, "lastRerouteId": 0},
    "revision": 0,
    "config": {},
    "name": "ReActor 换脸（分步）",
    "inputNode": {
        "id": -10,
        "bounding": [-320, 80, 140, 120],
    },
    "outputNode": {
        "id": -20,
        "bounding": [720, 420, 120, 60],
    },
    "inputs": [
        {
            "id": str(uuid.uuid4()),
            "name": "input_image",
            "type": "IMAGE",
            "linkIds": [205],
            "localized_name": "input_image",
            "label": "视频帧",
            "pos": [-200, 100],
        },
        {
            "id": str(uuid.uuid4()),
            "name": "source_image",
            "type": "IMAGE",
            "linkIds": [206],
            "localized_name": "source_image",
            "label": "目标脸",
            "pos": [-200, 130],
        },
    ],
    "outputs": [
        {
            "id": str(uuid.uuid4()),
            "name": "IMAGE",
            "type": "IMAGE",
            "linkIds": [204],
            "localized_name": "IMAGE",
            "pos": [740, 440],
        }
    ],
    "widgets": [],
    "nodes": sg_nodes,
    "links": sg_links,
    "groups": [
        {
            "id": 1,
            "title": "① 检测+换脸",
            "bounding": [-60, -50, 340, 340],
            "color": "#8A8",
            "font_size": 20,
            "flags": {},
        },
        {
            "id": 2,
            "title": "② 修复",
            "bounding": [-60, 370, 340, 240],
            "color": "#88A",
            "font_size": 20,
            "flags": {},
        },
        {
            "id": 3,
            "title": "预览",
            "bounding": [340, -50, 320, 720],
            "color": "#444",
            "font_size": 20,
            "flags": {},
        },
    ],
    "extra": {"workflowRendererVersion": "LG"},
}

# --- Main workflow from source ---
with open(SRC, encoding="utf-8") as f:
    wf = json.load(f)

# Replace ReActor node with subgraph node
for n in wf["nodes"]:
    if n.get("type") == "ReActorFaceSwap":
        n["type"] = SUBGRAPH_ID
        n["title"] = "3. ReActor 换脸（子图·可展开）"
        n["size"] = [400, 200]
        n["properties"] = {
            "Node name for S&R": "ReActor 换脸（分步）",
            "proxyWidgets": [],
        }
        n["widgets_values"] = []
        # Keep same inputs/outputs interface
        n["inputs"] = [
            mk_inp("input_image", "IMAGE", 1),
            mk_inp("source_image", "IMAGE", 2),
        ]
        n["outputs"] = [mk_out("IMAGE", "IMAGE", [3])]
        break

wf.setdefault("definitions", {})["subgraphs"] = [subgraph]
wf["id"] = str(uuid.uuid4())
wf["extra"] = wf.get("extra", {})
wf["extra"]["workflowRendererVersion"] = "LG"

# Fix main link: subgraph output slot 0 -> RIFE (was node 3 slot 0)
for L in wf["links"]:
    if isinstance(L, list) and len(L) >= 5 and L[1] == 3:
        pass  # still node id 3, slot 0

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(wf, f, ensure_ascii=False, separators=(",", ":"))

print("Wrote", OUT)
