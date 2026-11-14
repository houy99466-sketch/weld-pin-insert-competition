# 视觉方案（工件识别：形态 + 角度 + 位置）

> ⚠️ 本文是**早期选型讨论**（OBB 六类 + 平面单应的轻量路线），作为备选保留。
> 比赛**实际使用**的方案见 [视觉方案.md](视觉方案.md)：D435i RGB-D + YOLOv8s-seg 掩膜 + CAD + FoundationPose 6D 位姿。

## 一、选型结论

| 需求 | 用什么 | 理由 |
|---|---|---|
| **识别工件 + 输出角度** | **YOLO11-OBB**（Ultralytics，`yolo11n-obb`） | OBB=有向框，直接给出旋转角 θ；COCO/DOTA 预训练，迁移学习几百张图即可；n 模型在 4060 上几分钟一轮；可导 ONNX，CPU 也能实时 |
| 标注 | **X-AnyLabeling** | 支持 OBB 旋转框标注、内置 SAM 辅助点选，Windows 直接装 |
| （可选）自动预标注 | **SAM 3 / SAM 2 + Grounding DINO** | 先自动分割、再转 OBB，人工只修错，标注量能砍一半 |
| （后续，要深度相机再上） | **FoundationPose** | 用 STEP 模型做 6D 位姿；需要 RGB-D（RealSense 之类），单目精度不可信 |

**为什么不直接上 6D 位姿**：我们现在只有单目 USB 相机，FoundationPose 这类方法要深度图 + CAD mesh 才能稳定收敛；单目版本在弱纹理金属件上误差大。先用 OBB + 平面单应（见第四节）足够把工件抓到 ±1 mm / ±1°。

## 二、类别定义（6 类）

```
A_up     工件A 正姿
A_side   工件A 侧姿
A_down   工件A 倒姿
B_up     工件B 正姿
B_side   工件B 侧姿
B_down   工件B 倒姿
```

这样 **形态直接由类别给出**，角度由 OBB 的 `xywhr` 里那个 `r` 给出，位置由框中心给出——正好对应机械臂需要的 `{part, form, angle, pos}`。

## 三、本地训练流程（4 步）

### 1. 采数据（相机页面）
打开 <http://127.0.0.1:8765/cam> → 录像 → 每种形态拍 **3~5 段、每段 20~60 秒**，覆盖：
- 不同摆放位置（工作区四角 + 中心）
- 不同朝向（0°/30°/60°/90°…）
- 不同光照（开灯/关灯/手遮挡一半）
- 场景里同时有 0 件、1 件、2 件

目标：**每类 ≥ 300 张有效帧**（6 类共约 2000 张）。

### 2. 抽帧
```powershell
vision\.venv\Scripts\python.exe vision\extract_frames.py --src real_robot\web\cam_videos --dst vision\dataset\raw --every 5
```
（`--every 5` = 每 5 帧取一张，30fps 视频即 6 张/秒，够密且不重复）

### 3. 标注
1. 装 X-AnyLabeling：`pip install x-anylabeling-cvhub`
2. 打开 `vision\dataset\raw`，格式选 **YOLO OBB**
3. 每个工件画一个**最小外接旋转框**，类别按上表 6 类之一
4. 导出，得到 `labels/*.txt`，格式：`class x1 y1 x2 y2 x3 y3 x4 y4`（归一化）

### 4. 训练
```powershell
vision\.venv\Scripts\python.exe vision\train.py --data vision\dataset\data.yaml --model yolo11n-obb.pt --epochs 150
```
4060 上 640 分辨率、batch 8，约 **3~6 分钟/轮**，150 轮 1 小时内出结果。
训练完权重在 `vision\runs\obb\train\weights\best.pt`。

### 5. 验证
```powershell
vision\.venv\Scripts\python.exe vision\predict.py --model vision\runs\obb\train\weights\best.pt --src real_robot\web\cam_videos\cam_xxx.mp4
```
打印每帧的 `{cls, conf, center_px, angle_deg}`，并输出带框视频。

## 四、像素 → 机械臂坐标（平面单应）

相机固定俯视工作区，工作时**所有工件都在同一平面**，所以用单应矩阵就够，不用手眼标定矩阵：

1. 在工作区放 4 个标记点（比如销、角块），记录它们在**机械臂坐标系**下的 (x, y)
2. 让臂带着夹爪依次点这 4 个点，记下对应的**像素坐标**（从相机画面读）
3. 解 `H`：`[u,v,1]^T ~ H · [x,y,1]^T`
4. 以后任意像素点 `(u,v)` → `(x,y)` 直接用 `H` 反变换

`vision\homography.py` 就是干这个的（4 点求解 + 校验残差）。

**角度**：OBB 的 θ 是图像角；要转到机械臂 yaw，只要按 H 的旋转分量换算。实操上更简单：让工件在 θ=0 时录一张，记下模型输出的 θ₀，之后 `Δθ = θ_measured − θ₀`。

## 五、和机械臂对接

最终视觉进程对外只吐这一条 JSON（放到 `vision_server.py`，臂那边轮询）：

```json
{"ts": 1763000000.0, "frame": "base", "part": "A", "form": "side",
 "angle_deg": 37.5, "pos_mm": [312.4, -18.2, 0.0], "conf": 0.93}
```

臂的 `POST /api/task/run` 直接吃这三个字段（`part` / `form` / `angle`），不用改后端。

## 六、常见坑

| 坑 | 处理 |
|---|---|
| 金属件反光，框会抖 | 相机固定 + 加柔光（一张白纸挡闪光）；训练集里专门加反光样本 |
| 正/倒 外观太像 | 用**孔的位置**区分：正姿孔朝上可见，倒姿孔被遮 → 加"是否看到孔"的特征样本 |
| 类别不平衡 | 每类都录够；训练时 `--augment` 默认开，小幅旋转/翻转够用 |
| 显卡 8 GB 溢出 | `batch=8 imgsz=640`；不够就 `batch=4` |
| 训练后现场变差 | 现场光照和数据集不一致 → 补录 30 张现场真实光照图再微调 10 轮 |

---

## 七、目录与提交约定（团队）

视觉成果统一放 `vision/`，只往这些位置写：

| 路径 | 放什么 | 是否入库 |
|---|---|---|
| `extract_frames.py` / `train.py` / `predict.py` | 脚本 | ✅ 入库 |
| `dataset/data.yaml` | 类别定义 | ✅ 入库 |
| `dataset/images/` `dataset/labels/` | 图片与 OBB 标注 | ❌ 不入库 |
| `weights/` | `best.pt` 等权重 | ❌ 不入库（留 `.gitkeep`） |
| `runs/` | 训练输出 | ❌ 不入库（留 `.gitkeep`） |
| `samples/`（仓库根） | 少量带框示意截图 | ✅ 入库 |
| `report/` | 报告素材（19 图 + 1 视频 + 证据快照） | ✅ 入库（只读归档，见 [report/README.md](report/README.md)） |

- 权重/数据集共享：传网盘后在本文档「权重下载」处补链接，不要塞进 Git。
- 与机械臂的接口 JSON 定稿后，字段表补在第五节，`real_robot/` 按此实现。
