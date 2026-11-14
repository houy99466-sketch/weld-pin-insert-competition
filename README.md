# 焊前自动组对 · 比赛工程

本次比赛实际交付的两部分：**视觉感知** 和 **真机调试台**。赛前没有可复用的既有工程，
这里的每一行都是这段时间从零做出来的。不含仿真内容。

| 模块 | 目录 | 内容 |
|---|---|---|
| 视觉 | [`vision/`](vision/) | 实际方案 [`视觉方案.md`](vision/视觉方案.md)：D435i RGB-D + YOLOv8s-seg 掩膜 + CAD + FoundationPose 6D 位姿；报告素材在 [`vision/report/`](vision/report/) |
| 真机 | [`real_robot/`](real_robot/) | reBot Arm B601-DM 网页调试台：动作链、夹爪标定、示教录制、主从跟随 |
| 工具 | [`tools/`](tools/) | 真机排查脚本（串口嗅探、舵机枚举、主臂速率基准、server.py 修复） |
| 文档 | [`docs/`](docs/) | 协作约定 |

---

## 目录结构

```
.
├─ vision/              视觉
│  ├─ 视觉方案.md        实际使用方案：四级流水线 · 对接契约 · 证据索引 · 引用口径
│  ├─ report/           报告素材：19 图 + 1 演示视频 + 20 份证据快照
│  ├─ dataset/          YOLO 数据集配置（data.yaml，早期 OBB 六类，备选）
│  ├─ weights/          权重（不入库）
│  └─ runs/             训练输出（不入库）
├─ real_robot/
│  ├─ rebot_dm_move_chain.py
│  └─ web/              调试台：server.py + chain_core.py + index.html + leader_scs.py
├─ tools/               真机排查与修复脚本（14 个）
└─ docs/00_协作约定.md   目录归属 · 分支 · 大文件红线 · 上线前检查
```

## 视觉

方案、参数、指标口径全部在 [vision/视觉方案.md](vision/视觉方案.md)。核心链路：

```
RGB ──► YOLOv8s-seg ──► 实例掩膜 ──► 二维方向（模 90° / 模 180°）
                                       │
Depth + 内参 + CAD ──► FoundationPose 初始化 ──► 连续跟踪 ──► 4×4 位姿 + 摆放状态
```

已跑通并留有证据：分割掩膜 mAP50 0.515→0.695（对自动标签）、150 帧跟踪 14.5 Hz / 单帧 27.7 ms、
初始化轮廓一致性 IoU 0.918、415 帧视频方向 281 帧有效、混合三类 3,209 帧（训练 2,174 / 验证 1,035）。

**未闭环**：手眼标定、机器人坐标变换、抓取闭环、物理位姿真值。报告口径见方案第 14 节，逐条对照再写。

## 真机

硬件：reBot Arm B601-DM，机械臂 → USB-CAN 驱动板 → USB-CDC（默认 `COM21` @ 921600）
→ 达妙 DM4340P×3 + DM4310×4；主臂 6 个 Feetech STS3215 挂 `COM22` @ 1 Mbps（只读）。

```bat
cd real_robot\web
python server.py          REM 独占串口，网页只发 HTTP
REM 打开 http://127.0.0.1:8765/
```

实测（2026-11-13/14）：动作链 43 s 跑通，末端位移（臂系）前 +34.3 / 侧 0.0 / 上 +104.3 mm，
IK 残差 ~1 mm；主臂读取 250–300 Hz、主从录制 50 Hz 名义；停止示教后保持使能零漂移。
细节见 [real_robot/web/README.md](real_robot/web/README.md)。

## 环境

| 项 | 版本/说明 |
|---|---|
| 视觉 | Python + OpenCV 5.0.0 / NumPy 2.5.4 / Ultralytics(YOLOv8s-seg) / FoundationPose |
| 相机 | Intel RealSense D435i（USB 2.1；彩色 640×480@30，RGB-D 640×480@15） |
| 真机 | reBot Arm B601-DM；DM4340P + DM4310；motorbridge 0.6.0 |

## 协作

分支、提交、视觉目录与上线前检查见 [docs/00_协作约定.md](docs/00_协作约定.md)。

## 说明

本仓库为团队私有协作空间，未开放转载、分发或商业使用授权。
