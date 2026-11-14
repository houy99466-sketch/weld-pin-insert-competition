# -*- coding: utf-8 -*-
"""不动机器人的单元验证：主从循环里「方向/增益改了要立刻生效，并且重新锚定不跳」。

用假的从臂（StubArm，只记录下发指令）和假的主臂（FakeLeader，J2 匀速走 +10°），
跑到一半把 map 里 J2 的 sign 从 +1 改成 -1，检查：
  1) 改之前 J2 指令 ≈ S0 + delta
  2) 改之后 J2 指令 ≈ S0（重新锚定，不跳）
  3) 之后继续走，方向相反
"""
import sys, time, threading, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import chain_core as core

Q0 = [0.0, -90.0, -90.0, 0.0, 0.0, 0.0, -160.0]


class StubMotor:
    def __init__(self, i, arm):
        self.i, self.arm = i, arm

    def send_pos_vel(self, pos, vlim):
        self.arm.cmd[self.i] = math.degrees(pos)

    def enable(self): pass
    def disable(self): pass


class StubArm:
    def __init__(self):
        self.vlims = np.array([0.22, 0.22, 0.22, 0.28, 0.28, 0.28, 1.2])
        self.cmd = [None] * 7
        self.motors = [StubMotor(i, self) for i in range(7)]

    def enable(self): pass
    def disable(self): pass

    def _snap(self):
        return core.Snapshot(t=0.0, deg=list(Q0), vel=[0.0] * 7,
                             torq=[0.0] * 7, status=[1] * 7, fault=[0] * 7)

    def read(self):
        return self._snap()

    def read_fast(self, spins=1):
        return self._snap()


class FakeLeader:
    """J2 从 180° 匀速走到 190°（其它关节不动）。"""

    def __init__(self, rate=10.0, t_end=3.0):
        self.t0 = time.time()
        self.rate, self.t_end = rate, t_end

    def read_counts(self):
        dt = min(time.time() - self.t0, self.t_end)
        out = []
        for j in range(6):
            deg = 180.0 + (self.rate * dt if j == 1 else 0.0)
            out.append(int(round(deg * 4096.0 / 360.0)))
        return out


class Mapper:
    """直接用 to_slave 不参与，这里只要 lead_deg。"""

    def __init__(self):
        self.cont = [None] * 6

    def lead_deg(self, counts):
        out = []
        for i, c in enumerate(counts):
            d = c * 360.0 / 4096.0
            prev = self.cont[i]
            if prev is not None:
                while d - prev > 180:
                    d -= 360
                while d - prev < -180:
                    d += 360
            self.cont[i] = d
            out.append(d)
        return out

    def to_slave(self, lead):
        return [None] * 7


def main():
    arm, leader = StubArm(), FakeLeader()
    mp = {"joints": [{"sign": 1, "gain": 1.0} for _ in range(6)]}
    ctl = {"stop": threading.Event(), "max_travel": 60.0, "vlim_scale": 1.0,
           "samples": [], "events": [], "notes": [], "map": mp}

    rec = []          # (t, cmd_J2, lead_J2)

    def flip_later():
        time.sleep(1.6)
        mp["joints"][1]["sign"] = -1
        rec.append((time.time(), "FLIP", None))

    threading.Thread(target=flip_later, daemon=True).start()
    th = threading.Thread(target=lambda: (ctl["stop"].wait(3.2), ctl["stop"].set()),
                          daemon=True)
    th.start()

    orig_read_fast = arm.read_fast

    def read_fast(spins=1):
        snap = orig_read_fast(spins)
        rec.append((time.time(), arm.cmd[1], None))
        return snap

    arm.read_fast = read_fast

    res = core.leader_follow_loop(arm, ctl, leader, Mapper(), emit=None)
    print("结束原因:", res["reason"], "样本:", len(res["samples"]))
    print("事件:", [(e["action"], e.get("signs")) for e in res["events"]])

    flip_t = next((t for t, a, _ in rec if a == "FLIP"), None)
    before = [c for t, c, _ in rec if isinstance(c, float) and t < flip_t - 0.1]
    right = [c for t, c, _ in rec if isinstance(c, float) and flip_t <= t < flip_t + 0.12]
    after = [c for t, c, _ in rec if isinstance(c, float) and t > flip_t + 0.5]
    if not before or not after or not right:
        print("采样不足，无法判定")
        return 1
    b0, b1 = before[0], before[-1]
    a0, a1 = after[0], after[-1]
    print(f"翻转前 J2 指令: {b0:.2f} -> {b1:.2f}  (delta {b1-b0:+.2f})")
    print(f"翻转瞬间 J2 指令: {right[0]:.2f}（应≈起点 {Q0[1]:.1f}，即重新锚定不跳）")
    print(f"翻转后 J2 指令: {a0:.2f} -> {a1:.2f}  (delta {a1-a0:+.2f})")
    ok_dir = (b1 - b0) > 0.5 and (a1 - a0) < -0.5
    ok_reanchor = abs(right[0] - Q0[1]) < 2.0
    reanchored = any(e["action"] == "reanchor" for e in res["events"])
    print("方向生效:", "PASS" if ok_dir else "FAIL")
    print("重新锚定(无跳变):", "PASS" if ok_reanchor and reanchored else "FAIL")
    return 0 if (ok_dir and ok_reanchor and reanchored) else 1


if __name__ == "__main__":
    raise SystemExit(main())
