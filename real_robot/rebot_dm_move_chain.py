# -*- coding: utf-8 -*-
"""
reBot Arm B601-DM 最简单的「全身」动作链：向前 + 向上，不乱左右晃。

硬件路径：机械臂 -> USB-CAN 驱动板 -> CDC 串口 COM21 @921600 -> 达妙 DM4340P/DM4310
通信库：motorbridge (pip install motorbridge)  transport = dm-serial

动作链（每段都有斜坡，末段回到起始姿态，最后失能，跟插上电时状态一致）：
  S0 自检        读 7 个电机状态，有任何故障码就退出
  S1 夹爪张/合   J7 张开 8° 再回位                       —— 用到 J7
  S2 微摆       J1 正 0.8°（垂直轴，转到就是左右，故意只给极小量）—— 用到 J1，净侧移 <3mm
  S3 主运动     J2 +10°, J3 -9°, J4 -9°, J5 -6°, J6 +8°  —— 5s 斜坡，末端向前 ~60mm、向上 ~128mm
  S4 保持       3s 停在目标位并记录
  S5 回程       5s 逆向回起始姿态
  S6 收尾       保持 1.5s 后失能

方向依据：官方 URDF (urdf/DM/urdf/ReBot_Arm_DM.urdf) 手工建链求 FK/Jacobian。
在「J2 与 URDF 符号相反」这条假设下（唯一能同时满足 URDF 限位的一组）:
  J2 +10 / J3 -9 / J4 -9 / J5 -6  =>  dx +61.9mm  dy -4.3mm  dz +127.6mm
另一条假设（读数即 URDF 角）下同一组增量 => dx +27.7mm  dz +138.9mm，依旧前+上。
用法:
  python rebot_dm_move_chain.py --dry-run     # 只算不动
  python rebot_dm_move_chain.py               # 真动
  python rebot_dm_move_chain.py --port COM21 --scale 0.6   # 幅度打折
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from datetime import datetime

import numpy as np

try:
    from motorbridge.core import Controller
    from motorbridge.models import Mode
except Exception as e:  # pragma: no cover
    sys.exit(f"需要 motorbridge: pip install motorbridge  ({e})")

PORT = "COM21"
BAUD = 921600
RATE_HZ = 100.0

# (关节名, motor_id, feedback_id, 型号, 位置限幅 deg)  —— 限幅取官方 URDF/机械限位再留余量
JOINTS = [
    ("J1", 0x01, 0x11, "4340P", 2.8 * 180 / math.pi, 0.22),
    ("J2", 0x02, 0x12, "4340P", 180.0, 0.22),
    ("J3", 0x03, 0x13, "4340P", 180.0, 0.22),
    ("J4", 0x04, 0x14, "4310", 107.0, 0.28),
    ("J5", 0x05, 0x15, "4310", 90.0, 0.28),
    ("J6", 0x06, 0x16, "4310", 180.0, 0.28),
    ("J7", 0x07, 0x17, "4310", 90.0, 0.12),
]

# 动作链主增量（度）：向前 + 向上。
# 由当前姿态 [J1..J6]=[1.93,-0.01,-10.28,43.48,0.62,-0.49]deg 用官方 URDF 求 FK/Jacobian 反解：
# 该组增量 => 末端 前 +36.3mm / 侧 +1.1mm / 上 +105.5mm（净侧移 1mm，等于不左右动）
DELTA_DEG = [1.5, -3.0, -4.0, -20.0, 2.0, 8.0]
GRIPPER_OPEN_DEG = 8.0

T_GRIP = 0.8
T_MAIN = 5.0
T_HOLD = 3.0
T_BACK = 5.0
T_END = 1.5


def s_curve(s: float) -> float:
    """0..1 的 raised-cosine，起停都平顺。"""
    s = min(max(s, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * s)


class Arm:
    def __init__(self, port: str):
        self.ctrl = Controller.from_dm_serial(port, BAUD)
        self.motors = []
        for name, mid, fid, model, lim, vlim in JOINTS:
            m = self.ctrl.add_damiao_motor(mid, fid, model)
            self.motors.append(m)
        self.names = [j[0] for j in JOINTS]
        self.lims = np.array([j[4] for j in JOINTS])
        self.vlims = np.array([j[5] for j in JOINTS])
        self.log = []

    # ---- 底层 ----
    def read(self, spins: int = 3) -> np.ndarray:
        q = np.full(len(self.motors), np.nan)
        st = [None] * len(self.motors)
        for _ in range(spins):
            for m in self.motors:
                m.request_feedback()
            for _ in range(6):
                try:
                    self.ctrl.poll_feedback_once()
                except Exception:
                    pass
                time.sleep(0.004)
            for i, m in enumerate(self.motors):
                s = m.get_state()
                if s is not None:
                    st[i] = s
                    q[i] = s.pos
            if not np.isnan(q).any():
                break
        self.last_state = st
        return q

    def snap(self, tag: str, q: np.ndarray, extra: dict | None = None) -> None:
        rec = {
            "t": round(time.time() - self.t0, 3),
            "tag": tag,
            "deg": [None if np.isnan(v) else round(math.degrees(v), 2) for v in q],
        }
        if extra:
            rec.update(extra)
        self.log.append(rec)
        print(f"[{rec['t']:6.2f}s] {tag:22s} " +
              " ".join(f"{n}={d:+7.2f}" if d is not None else f"{n}=  ?  "
                       for n, d in zip(self.names, rec["deg"])))

    def stream(self, target: np.ndarray, duration: float, tag: str,
               start: np.ndarray | None = None, check_limit: bool = True) -> np.ndarray:
        """从 start（默认当前实测）以 s 曲线走到 target，返回走完后的实测角。"""
        if start is None:
            start = self.read()
        if np.isnan(start).any():
            raise RuntimeError("读不到某些关节角度，停止")
        if check_limit and np.any(np.abs(target) > np.deg2rad(self.lims)):
            raise RuntimeError(f"目标超过限幅: {np.rad2deg(target).round(1)}")
        n = max(int(duration * RATE_HZ), 1)
        dt = 1.0 / RATE_HZ
        worst = 0.0
        stuck = 0
        for k in range(1, n + 1):
            a = s_curve(k / n)
            cmd = start + (target - start) * a
            for m, c, v in zip(self.motors, cmd, self.vlims):
                m.send_pos_vel(float(c), float(v))
            for m in self.motors:
                m.request_feedback()
            for _ in range(2):
                try:
                    self.ctrl.poll_feedback_once()
                except Exception:
                    pass
                time.sleep(dt / 2.0)
            q = np.array([(m.get_state().pos if m.get_state() else np.nan) for m in self.motors])
            if not np.isnan(q).any():
                err = float(np.max(np.abs(q - cmd)))
                worst = max(worst, err)
                stuck = stuck + 1 if err > 0.5 else 0
                if stuck > 40:  # 0.4s 跟不上 0.5rad(28°)
                    raise RuntimeError(f"关节跟踪失败，最大误差 {math.degrees(err):.1f}°，中止")
        q_end = self.read()
        self.snap(tag, q_end, {"worst_track_err_deg": round(math.degrees(worst), 2),
                               "dur_s": duration})
        return q_end


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--scale", type=float, default=1.0, help="增量缩放，0.5=半幅")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-disable", action="store_true", help="收尾保持使能不释放")
    ap.add_argument("--stay", action="store_true", help="停在目标位不回程（隐含 --no-disable）")
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    delta = np.deg2rad(np.array(DELTA_DEG) * args.scale)

    if args.dry_run:
        print("DRY-RUN 计划（度）:")
        for n, d in zip([j[0] for j in JOINTS[:6]], DELTA_DEG):
            print(f"  {n}: {d*args.scale:+6.2f}")
        print(f"  J7: 先 +{GRIPPER_OPEN_DEG*args.scale:.1f} 再回原位")
        print(f"  时长: 夹爪{T_GRIP}s / 主运动{T_MAIN}s / 保持{T_HOLD}s / 回程{T_BACK}s / 收尾{T_END}s")
        return 0

    arm = Arm(args.port)
    arm.t0 = time.time()
    ok = False
    try:
        print(f"=== 连接 {args.port} @{BAUD} ===")
        q0 = arm.read()
        if np.isnan(q0).any():
            print("有电机读不到反馈，先查线束/CAN 再跑")
            return 2
        arm.snap("S0 自检", q0)
        print("  使能状态 status_code: 0=失能 1=使能 | " +
              " ".join(f"{n}:{s.status_code}" for n, s in zip(arm.names, arm.last_state) if s))

        bad = [(arm.names[i], s.fault_code) for i, s in enumerate(arm.last_state)
               if s is not None and s.fault_code != 0]
        if bad:
            print(f"[abort] 有故障码: {bad}")
            return 3

        print("\n=== 使能 ===")
        for m in arm.motors:
            m.enable()
        time.sleep(0.3)
        for m in arm.motors:
            try:
                m.ensure_mode(Mode.POS_VEL, 1000)
            except Exception as e:
                print(f"[warn] ensure_mode: {e}")

        q = arm.read()
        arm.snap("S1 使能保持", q)

        # S1 夹爪张/合
        grip0 = q[6]
        tg = q.copy(); tg[6] = grip0 + math.radians(GRIPPER_OPEN_DEG * args.scale)
        q = arm.stream(tg, T_GRIP, "S2 夹爪张开")
        tg2 = q.copy(); tg2[6] = grip0
        q = arm.stream(tg2, T_GRIP, "S3 夹爪回位")

        # S2/S3 主链：J1..J6 一起走到 前+上
        target = q.copy()
        target[:6] = q[:6] + delta
        q = arm.stream(target, T_MAIN, "S4 向前+向上(5s斜坡)")

        # S4 保持
        t_end = time.time() + T_HOLD
        while time.time() < t_end:
            for m, c, v in zip(arm.motors, target, arm.vlims):
                m.send_pos_vel(float(c), float(v))
            for m in arm.motors:
                m.request_feedback()
            try:
                arm.ctrl.poll_feedback_once()
            except Exception:
                pass
            time.sleep(0.01)
        q = arm.read()
        arm.snap("S5 目标位保持", q)
        with open(args.log or f"chain_{datetime.now():%Y%m%d_%H%M%S}.json", "w", encoding="utf-8") as f:
            json.dump({"port": args.port, "scale": args.scale, "delta_deg": DELTA_DEG,
                       "q0_deg": [round(math.degrees(v), 2) for v in q0],
                       "target_deg": [round(math.degrees(v), 2) for v in target],
                       "log": arm.log}, f, ensure_ascii=False, indent=1)

        # S5 回程（--stay 时停在目标位持续保持）
        if args.stay:
            print("\n=== --stay：停在目标位，Ctrl+C 结束（结束后自动失能）===")
            while True:
                for m, c, v in zip(arm.motors, target, arm.vlims):
                    m.send_pos_vel(float(c), float(v))
                time.sleep(0.02)
        q = arm.stream(q0, T_BACK, "S6 回起始姿态")
        t_end = time.time() + T_END
        while time.time() < t_end:
            for m, c, v in zip(arm.motors, q0, arm.vlims):
                m.send_pos_vel(float(c), float(v))
            time.sleep(0.01)
        ok = True
        print("\n=== 动作链完成 ===")
        return 0
    except KeyboardInterrupt:
        print("\n[中断] 立即失能")
        return 4
    except Exception as e:
        print(f"\n[异常] {e}")
        return 5
    finally:
        try:
            if not args.no_disable:
                for m in arm.motors:
                    try:
                        m.disable()
                    except Exception:
                        pass
                q = arm.read()
                arm.snap("S7 失能(收尾)", q)
        finally:
            try:
                arm.ctrl.close_bus()
            finally:
                arm.ctrl.close()


if __name__ == "__main__":
    raise SystemExit(main())
