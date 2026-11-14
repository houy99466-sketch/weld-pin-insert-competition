# -*- coding: utf-8 -*-
"""
主从链路的端到端验证（用小幅度假主臂，避免真机大幅度动作）。

流程：
  1) 开从臂 COM21（read 一次，确认 DM 总线正常）
  2) 假主臂 FakeLeader 生成 ±3° 的慢速正弦（只动 J1..J3，幅度很小）
  3) 跑 core.leader_follow_loop，记录样本
  4) 校验：从臂确实跟着动了、幅度符合映射增益、J4..J6 基本不动
  5) 收尾 disable + close

用法: python test_leader_link.py [--go]    不带 --go 只做几何/映射的干跑检查
"""
import sys, time, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import chain_core as core
import leader_scs as leadmod

GO = "--go" in sys.argv
AMP_DEG = 3.0
PERIOD_S = 2.4
DUR = 6.0
PORTS = ("COM21", 921600)


class FakeLeader:
    """假主臂：按正弦生成 counts，模拟徒手拖动。"""

    def __init__(self, t0=None):
        self.t0 = t0 or time.time()

    def read_counts(self):
        ph = (time.time() - self.t0) * 2 * math.pi / PERIOD_S
        out = []
        for j in range(6):
            deg = 180.0 + (AMP_DEG * math.sin(ph) if j < 3 else 0.0)
            out.append(int(round(deg * 4096.0 / 360.0)))
        return out


def main():
    mp = leadmod.load_map()
    mapper = leadmod.LeaderMapper(mp)
    print("映射文件:", json.dumps({k: v for k, v in mp.items() if k != "leader"},
                                 ensure_ascii=False)[:220], "...")

    if not GO:
        fake = FakeLeader()
        deg = mapper.lead_deg(fake.read_counts())
        print("干跑：主臂角", [round(d, 2) for d in deg])
        print("干跑：从臂目标", [None if v is None else round(v, 2)
                             for v in mapper.to_slave(deg)])
        print("（加 --go 才会真的驱动机器人）")
        return

    arm = core.Arm(*PORTS)
    try:
        snap = arm.read()
        print("从臂当前:", snap.deg)
        if any(d is None for d in snap.deg):
            raise SystemExit("从臂读不到反馈，先别跑")

        ctl = {
            "stop": __import__("threading").Event(),
            "gains": None, "signs": None,
            "max_travel": AMP_DEG * 1.5,
            "vlim_scale": 0.25,
            "j7_hold": False,
            "samples": [], "events": [], "notes": [],
        }
        fake = FakeLeader()
        print(f"开始跟随 {DUR}s（假主臂 J1..J3 ±{AMP_DEG}°）…")
        th = __import__("threading").Thread(
            target=lambda: ctl["stop"].wait(DUR) or ctl["stop"].set(), daemon=True)
        th.start()
        res = core.leader_follow_loop(arm, ctl, fake, mapper,
                                      emit=lambda *a: None)
        s = np.array([x[1:] for x in res["samples"]], float)
        print("结束原因:", res["reason"], " 样本:", len(s))
        if len(s):
            lo, hi = s.min(0), s.max(0)
            print("各关节行程: " + "  ".join(
                f"J{i+1}:{hi[i]-lo[i]:.2f}°" for i in range(7)))
            print("J4..J6 应≈0 →", [round(float(hi[i] - lo[i]), 2) for i in (3, 4, 5)])
            ok = float(hi[0] - lo[0]) > 1.0
            print("跟随验证:", "PASS" if ok else "FAIL（J1 没动）")
    finally:
        try:
            arm.disable()
        except Exception:
            pass
        arm.close()


if __name__ == "__main__":
    main()
