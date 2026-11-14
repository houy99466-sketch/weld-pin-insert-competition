# -*- coding: utf-8 -*-
"""把 server_rebuilt.py 里剩下的 33 处残句按原文补齐（行号 → 新行）。"""

import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

DOC = '''"""
reBot B601-DM 调试前端后端。

为什么需要它：浏览器不能直接开串口，DM 的 USB-CAN 板是独占 CDC 设备。
所以由这个进程独占 COM 口，网页通过 HTTP 下指令。

启动：
    python server.py                 # 默认 COM21 + http://127.0.0.1:8765
    python server.py --com COM21 --http-port 8765

API：
    GET  /api/info        硬件/动作链元数据
    GET  /api/state       读一次关节状态（deg/vel/torq/status/fault）
    POST /api/open        占用串口
    POST /api/release     释放串口（让 MotorBridge Studio / DM_Tools 用）
    POST /api/enable      使能
    POST /api/disable     失能（卸力矩）
    POST /api/run         {"scale":1.0,"stay":false} 跑动作链
    POST /api/abort       急停（中止 + 失能）
    GET  /api/job         当前任务进度与日志
"""'''.split("\n")

PATCH = {
    2: DOC,                                                     # 2..21 整块文档串
    22: [],
    41: ['import leader_scs as leadmod  # noqa: E402  主臂(另一台臂)读取层'],
    55: ['    """返回 (版本号, 已注入版本号的 html)。文件变了版本号就变，前端据此自动重载。"""'],
    91: ['        # ---- 主臂（另一台 6 轴，Feetech STS3215 总线舵机） ----'],
    100: ['    # ---------- 夹爪（独立，随时可用） ----------'],
    102: ['        """不需要示教也能用：只驱动 J7，其他关节不动。'],
    103: ['        示教进行中则交给示教循环（避免两条指令抢同一个电机）。"""'],
    107: ['            raise RuntimeError("任务运行中：夹爪由任务控制，先中止任务")'],
    122: ['            raise RuntimeError("有任务在跑，先中止")'],
    139: ['            raise RuntimeError("有任务在跑，先中止")'],
    152: ['    # ---------- 零校准 ----------'],
    156: ['            raise RuntimeError("有任务在跑，先中止")'],
    176: ['            raise RuntimeError("槽位名不能为空")'],
    179: ['            raise RuntimeError("读不到全部关节反馈（机械臂没上电 / CAN 没通），先点『读状态』确认")'],
    196: ['        """找一个名为 push2_A*.json 的录制，用于"整条轨迹回放"式推件。"""'],
    230: ['    # ---------- 主臂（另一台 6 轴臂，Feetech STS3215 总线舵机） ----------'],
    233: ['            raise RuntimeError("示教进行中：先『停止并保存』")'],
    242: ['                raise RuntimeError(f"{p} @{b} 上没有扫到 STS 舵机（查波特率/接线/供电）")'],
    294: ['        """改映射：signs/gains/use_abs/joints[i].lmin..lmax，写回 leader_map.json。"""'],
    313: ['            raise RuntimeError("示教进行中：先点『停止并保存』")'],
    318: ['            raise RuntimeError("上一轮示教线程还没退出，等 2 秒再试")'],
    322: ['            raise RuntimeError("有任务在跑，先中止")'],
    397: ['            raise RuntimeError("当前没有在示教")'],
    403: ['        # free 模式下夹爪上过电的，收尾要关掉'],
    438: ['            raise RuntimeError("先开始示教")'],
    449: ['                "note": ("向闭合方向爬，力矩到 1.5Nm 自动停" if action == "close"'],
    450: ['                         else "向张开方向爬，遇阻或走满 110° 停")}'],
    455: ['            raise RuntimeError("先开始示教")'],
    472: ['        """只允许 recordings 目录下的 .json，防止路径穿越。"""'],
    475: ['            raise RuntimeError("缺少文件名")'],
    478: ['            raise RuntimeError("文件名不合法（必须在本目录下的 .json）")'],
    494: ['            raise RuntimeError(f"{target.name} 已存在")'],
    508: ['        """扫描 recordings 目录下所有 .json（不限定文件名前缀，改名/手动拷入都能识别）。"""'],
    549: ['        return ("串口被别的程序占用。先关掉 MotorBridge Studio / ws_gateway / DM_Tools，"'],
    550: ['                "或点『释放串口』再关掉它们。查占用："'],
    554: ['        """系统当前存在的串口。优先 pyserial；没装就退回 WMI（PowerShell 查 PnP）。"""'],
    571: ['        # 退路：只要当前配置的口还在，就至少把它列出来'],
    618: ['                if "找不到" in msg or "cannot find" in low or "no such file" in low:'],
    619: ['                    have = ", ".join(p["device"] for p in self.ports()) or "（一个都没有）"'],
    621: ['                        f"找不到串口 {self.port}：USB 线没插 / 驱动没起来 / 端口号变了。"'],
    622: ['                        f"当前可用串口：{have}。插好后点『读状态』重试，或在下拉框里换端口。")'],
    642: ['    # ---------- 状态 ----------'],
    645: ['            # 示教中：真实读一次（free 模式下电机是失能的，status 会如实报 0）'],
    657: ['            # 任务进行中也要真实读一次，否则前端显示的是旧缓存（会让人以为臂没动）'],
    688: ['    # ---------- 起始位 ----------'],
    701: ['            raise RuntimeError("还没记录起始位：先点『记录起始位』")'],
    705: ['    # ---------- 动作链 ----------'],
    752: ['                    emit("回到起始位", 0.0, None)'],
    755: ['                               on_tick=_pct_tick(emit, "回到起始位", float(params.get("duration", 4.0))))'],
    756: ['                    emit("回到起始位", 100.0, arm.read())'],
    760: ['                "停在目标位" if params.get("stay") else "完成")'],
    792: ['        """中断当前动作但【保持使能、原地锁住】（区别于 abort：不卸力）。"""'],
    822: ['    """给 hold/stream 用的百分比回调（按时间估算）。"""'],
    840: ['        """持续输出 multipart MJPEG（浏览器 <img> 直接显示）。"""'],
    877: ['    def log_message(self, fmt, *args):  # 安静点，只在控制台打短日志'],
    886: ['        # 首页：注入当前版本号 + 禁缓存（否则改了前端、浏览器还在跑旧 JS）'],
    936: ['                    "note": "J7 负=张开(自由到-140°)，正=夹紧(+10°即有负载，+18°顶死)；"'],
    937: ['                            "主运动段末端约 前+36mm / 侧+1mm / 上+106mm（scale=1）",'],
    967: ['                    raise RuntimeError("示教进行中，先停止示教")'],
    974: ['                    raise RuntimeError("示教进行中，先停止示教")'],
    1071: ['    print(f"串口: {args.com} @{core.BAUD}（首次读状态时占用；点『释放串口』可交给 MotorBridge）")'],
    1075: ['        print("\\n退出，释放串口…")'],
}

text = src.read_text(encoding="utf-8").replace("\r\n", "\n")
lines = text.split("\n")

# 2..21 合并成 DOC
out = []
i = 0
while i < len(lines):
    n = i + 1
    if n == 2:
        out.extend(DOC)               # DOC 最后一行就是收尾的 """
        i = 22                        # 跳过到 22（原 22 是 '"""'）
        continue
    if n in PATCH and n != 2:
        out.extend(PATCH[n])
        i += 1
        continue
    out.append(lines[i])
    i += 1

dst.write_bytes(("\n".join(out)).replace("\n", "\r\n").encode("utf-8"))
print("写出", dst, len(out), "行")
