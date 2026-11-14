# -*- coding: utf-8 -*-
"""
主臂(leader) 读取层：飞特 Feetech SCS/STS 总线舵机 (STS3215, model=777)
USB 转串口 CH343 -> COM22 @ 1,000,000 8N1

本模块只做「读」：不解锁力矩、不发运动指令，主臂永远保持可徒手拖动。
对外给出 6 个关节角(deg)，并按 leader_map.json 映射到从臂(reBot DM 6 轴+夹爪)的关节角。

用法（独立自检）:
    python leader_scs.py COM22                       # 读 2 秒速率测试
    python leader_scs.py COM22 --range 6             # 记录 6 秒行程范围
    python leader_scs.py COM22 --monitor             # 连续打印 + 映射后的从臂目标
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import serial

HERE = Path(__file__).resolve().parent
MAP_FILE = HERE / "leader_map.json"

# 飞特 STS 系列寄存器（与 SCS 系列同构，地址不同）
R_MODEL = (3, 2)
R_ID = (5, 1)
R_BAUD = (6, 1)
R_MIN = (9, 2)
R_MAX = (11, 2)
R_OFS = (31, 2)
R_TORQUE_EN = (40, 1)
R_GOAL = (42, 2)
R_POS = (56, 2)
R_SPEED = (58, 2)
R_LOAD = (60, 2)
R_VOLT = (62, 1)
R_TEMP = (63, 1)
R_MOVING = (66, 1)
R_CURRENT = (69, 2)

CNT_PER_DEG = 4096.0 / 360.0
MODEL_777 = "STS3215"

# 从臂(reBot B601-DM)关节语义限位，单位 deg —— 与 chain_core.URDF_LIMITS 一致
SLAVE_LIMITS_DEG = (
    (-160.0, 160.0),
    (-180.0, 0.0),
    (-180.0, 0.0),
    (-107.0, 109.0),
    (-90.0, 90.0),
    (-180.0, 180.0),
)
SLAVE_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6", "J7")


def _chk(body) -> bytes:
    return bytes([(~(sum(body))) & 0xFF])


class LeaderError(RuntimeError):
    pass


class LeaderBus:
    """主臂总线：只读。"""

    def __init__(self, port: str = "COM22", baud: int = 1_000_000,
                 ids=(1, 2, 3, 4, 5, 6), timeout: float = 0.02):
        self.port = port
        self.baud = baud
        self.ids = list(ids)
        self.timeout = timeout
        self._lock = threading.RLock()
        # timeout=0：非阻塞。Windows 上 Serial.read(n) 会等满 n 字节或超时，
        # 用它读整帧会平白多花一整个 timeout（实测 60ms），所以改成轮询 in_waiting。
        self.s = serial.Serial(port, baud, timeout=0, write_timeout=0.2)
        # 丢掉上电残留
        try:
            self.s.reset_input_buffer()
        except Exception:
            pass
        self.last_ok = time.time()
        self.tx = 0
        self.rx = 0

    # ---------------- 底层 ----------------
    def close(self):
        try:
            self.s.close()
        except Exception:
            pass

    @staticmethod
    def _frame(sid, instr, params=()):
        body = [sid, len(params) + 2, instr] + list(params)
        return bytes([0xFF, 0xFF] + body) + _chk(body)

    def _talk(self, sid, instr, params=(), expect_len=0, tries=2):
        f = self._frame(sid, instr, params)
        for _ in range(tries):
            with self._lock:
                self.s.reset_input_buffer()
                self.s.write(f)
                self.tx += 1
                buf = bytearray()
                deadline = time.perf_counter() + 0.02
                while time.perf_counter() < deadline:
                    n = self.s.in_waiting
                    if n:
                        buf += self.s.read(n)
                        self.rx += n
                        i = buf.find(b"\xff\xff")
                        if i >= 0 and len(buf) >= i + 4:
                            ln = buf[i + 3]
                            if len(buf) >= i + 4 + ln:
                                # 必须校验 ID：多线程/多帧交错时不能把别人的应答当成自己的
                                if buf[i + 2] != sid:
                                    i2 = buf.find(b"\xff\xff", i + 1)
                                    if i2 >= 0:
                                        buf = buf[i2:]
                                        continue
                                    buf = bytearray()
                                    continue
                                return bytes(buf[i:i + 4 + ln])
                    else:
                        time.sleep(0.0002)
        return None

    def ping(self, sid, tries=2) -> bool:
        return self._talk(sid, 0x01, (), tries=tries) is not None

    def read_reg(self, sid, reg, tries=2):
        addr, ln = reg
        r = self._talk(sid, 0x02, [addr, ln], tries=tries)
        if not r or len(r) < 6 or r[4] != 0:
            return None
        d = r[5:-1]
        if len(d) < ln:
            return None
        return d[0] if ln == 1 else (d[0] | (d[1] << 8))

    # ---------------- 高层 ----------------
    def scan(self, lo=1, hi=16):
        return [i for i in range(lo, hi + 1) if self.ping(i, tries=1)]

    def info(self):
        out = []
        for sid in self.ids:
            out.append(dict(
                id=sid,
                model=self.read_reg(sid, R_MODEL),
                pos_cnt=self.read_reg(sid, R_POS),
                min_cnt=self.read_reg(sid, R_MIN),
                max_cnt=self.read_reg(sid, R_MAX),
                ofs_cnt=self.read_reg(sid, R_OFS),
                torque_en=self.read_reg(sid, R_TORQUE_EN),
                volt=self.read_reg(sid, R_VOLT),
                temp=self.read_reg(sid, R_TEMP),
            ))
        return out

    def read_counts(self):
        """一次性读 6 个位置（counts），缺失返回 None。整轮持锁，避免与他线程交错。"""
        with self._lock:
            return [self.read_reg(sid, R_POS, tries=1) for sid in self.ids]

    def read_motion(self):
        """位置 + 速度 + 负载 + 电压 + 温度（用于示教时的健康检查）。"""
        rows = []
        for sid in self.ids:
            rows.append(dict(
                pos=self.read_reg(sid, R_POS, tries=1),
                speed=self.read_reg(sid, R_SPEED, tries=1),
                load=self.read_reg(sid, R_LOAD, tries=1),
                volt=self.read_reg(sid, R_VOLT, tries=1),
                temp=self.read_reg(sid, R_TEMP, tries=1),
            ))
        return rows


# ---------------------------------------------------------------- 映射
DEFAULT_MAP = {
    "note": "leader(主臂 STS3215) -> slave(reBot DM) 关节映射；默认走相对增量映射",
    "use_abs": False,
    "max_travel_deg": 130.0,
    "vlim_scale": 0.55,
    "j7_hold": False,
    "leader": {"port": "COM22", "baud": 1000000, "ids": [1, 2, 3, 4, 5, 6]},
    "joints": [
        {"src": 1, "sign": +1, "gain": 1.0, "lmin": 60.3, "lmax": 296.3, "smin": -160.0, "smax": 160.0},
        {"src": 2, "sign": +1, "gain": 1.0, "lmin": 71.1, "lmax": 279.9, "smin": -180.0, "smax": 0.0},
        {"src": 3, "sign": +1, "gain": 1.0, "lmin": 84.5, "lmax": 280.7, "smin": -180.0, "smax": 0.0},
        {"src": 4, "sign": +1, "gain": 1.0, "lmin": 79.9, "lmax": 284.6, "smin": -107.0, "smax": 109.0},
        {"src": 5, "sign": +1, "gain": 1.0, "lmin": 10.4, "lmax": 348.3, "smin": -90.0, "smax": 90.0},
        {"src": 6, "sign": +1, "gain": 1.0, "lmin": 179.9, "lmax": 312.1, "smin": -180.0, "smax": 180.0},
    ],
}


def load_map(path: Path = MAP_FILE) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    d = json.loads(json.dumps(DEFAULT_MAP))
    save_map(d, path)
    return d


def save_map(d: dict, path: Path = MAP_FILE) -> None:
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def counts_to_deg(cnt):
    return None if cnt is None else cnt * 360.0 / 4096.0


def unwrap(prev, cur):
    """处理 0/360 跨界：让角度连续。"""
    if prev is None or cur is None:
        return cur
    d = cur - prev
    while d > 180.0:
        cur -= 360.0
        d = cur - prev
    while d < -180.0:
        cur += 360.0
        d = cur - prev
    return cur


class LeaderMapper:
    """主臂角(deg) -> 从臂 7 关节目标角(deg)。"""

    def __init__(self, mp: dict | None = None):
        self.mp = mp or load_map()
        self.cont = [None] * 6

    def reload(self):
        self.mp = load_map()

    def lead_deg(self, counts):
        """counts -> 连续角度(deg)，自动处理 0/360 跨界。读不到时沿用上一次的值。"""
        out = []
        for i, c in enumerate(counts):
            if i >= len(self.cont):
                break
            deg = counts_to_deg(c)
            if deg is None:
                out.append(self.cont[i])
                continue
            if self.cont[i] is not None:
                deg = unwrap(self.cont[i], deg)
            self.cont[i] = deg
            out.append(deg)
        return out

    def to_slave(self, lead_deg):
        tgt = []
        jl = self.mp["joints"]
        for i, j in enumerate(jl):
            L = lead_deg[i] if i < len(lead_deg) else None
            lmin, lmax = float(j["lmin"]), float(j["lmax"])
            smin, smax = float(j["smin"]), float(j["smax"])
            if L is None or lmax - lmin < 1e-6:
                tgt.append(None)
                continue
            u = (L - lmin) / (lmax - lmin)
            u = 0.0 if u < 0 else (1.0 if u > 1 else u)
            if int(j.get("sign", 1)) < 0:
                u = 1.0 - u
            tgt.append(smin + u * (smax - smin))
        # J7 夹爪：主臂没有夹爪舵机，保持当前值（None=不接管）
        tgt.append(None)
        return tgt


# ---------------------------------------------------------------- 自检
def _rate_test(bus: LeaderBus, sec=2.0):
    n = 0
    t0 = time.time()
    bad = 0
    while time.time() - t0 < sec:
        c = bus.read_counts()
        n += 1
        if any(x is None for x in c):
            bad += 1
    dt = time.time() - t0
    print(f"读取 {n} 轮 / {dt:.2f}s = {n/dt:.1f} Hz，缺帧轮次 {bad}")
    return n / dt


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM22"
    baud = 1_000_000
    # 不连硬件也能用的命令：翻转某关节方向 / 打印当前映射
    if "--flip" in sys.argv:
        j = int(sys.argv[sys.argv.index("--flip") + 1]) - 1
        mp = load_map()
        mp["joints"][j]["sign"] = -int(mp["joints"][j].get("sign", 1))
        save_map(mp)
        print(f"J{j+1} sign -> {mp['joints'][j]['sign']:+d}（已写入 {MAP_FILE}）")
        return
    if "--map" in sys.argv:
        mp = load_map()
        print("J1..J6 sign:" , [int(j.get("sign", 1)) for j in mp["joints"]])
        print("gain        :", [float(j.get("gain", 1.0)) for j in mp["joints"]])
        print("use_abs     :", mp.get("use_abs"), " max_travel:", mp.get("max_travel_deg"))
        return
    bus = LeaderBus(port, baud)
    try:
        print(f"[leader] {port} @{baud}")
        print("扫描:", bus.scan())
        for r in bus.info():
            print("  ", r)
        mp = LeaderMapper()
        print("\n-- 速率测试 --")
        hz = _rate_test(bus)
        if "--range" in sys.argv:
            sec = float(sys.argv[sys.argv.index("--range") + 1])
            print(f"\n-- 行程采集 {sec}s：请把主臂各关节从一端拖到另一端 --")
            lo = [1e9] * 6
            hi = [-1e9] * 6
            t0 = time.time()
            while time.time() - t0 < sec:
                degs = mp.lead_deg(bus.read_counts())
                for i, d in enumerate(degs):
                    if d is not None:
                        lo[i] = min(lo[i], d)
                        hi[i] = max(hi[i], d)
                time.sleep(0.02)
            for i in range(6):
                print(f"  J{i+1}: {lo[i]:7.2f} .. {hi[i]:7.2f}  span={hi[i]-lo[i]:6.2f}")
            if "--save" in sys.argv:
                for i, j in enumerate(mp.mp["joints"]):
                    if hi[i] - lo[i] > 10.0:
                        j["lmin"], j["lmax"] = round(lo[i], 2), round(hi[i], 2)
                save_map(mp.mp)
                print("已写入", MAP_FILE)
        if "--monitor" in sys.argv:
            print("\n-- 连续监视 (Ctrl+C 退出)：主臂角 -> 从臂目标 --")
            while True:
                degs = mp.lead_deg(bus.read_counts())
                tgt = mp.to_slave(degs)
                ls = " ".join("  --  " if d is None else f"{d:7.2f}" for d in degs)
                ts = " ".join("  --  " if d is None else f"{d:7.2f}" for d in tgt)
                print(f"lead[{ls}] -> slave[{ts}]")
                time.sleep(0.1)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
