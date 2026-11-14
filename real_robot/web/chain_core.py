# -*- coding: utf-8 -*-
"""
reBot Arm B601-DM 共用执行层：串口/电机访问 + 动作链。
被 web/server.py（调试前端）与命令行脚本共同使用；这里没有任何 UI 依赖。

硬件：机械臂 -> USB-CAN 驱动板 -> CDC 串口(默认 COM21) @921600 -> 达妙 DM4340P/DM4310
通信：motorbridge  transport=dm-serial

动作链定义集中在 PLAN 里（单一数据源）：网页从 /api/info 读同一份计划来渲染。
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from motorbridge.core import Controller
from motorbridge.models import Mode

DEFAULT_PORT = "COM21"
BAUD = 921600
RATE_HZ = 100.0

# (关节名, motor_id, feedback_id, 型号, 角度限幅 deg, 速度上限 rad/s)
JOINTS = (
    ("J1", 0x01, 0x11, "4340P", 400.0, 0.22),
    ("J2", 0x02, 0x12, "4340P", 400.0, 0.22),
    ("J3", 0x03, 0x13, "4340P", 400.0, 0.22),
    ("J4", 0x04, 0x14, "4310", 400.0, 0.28),
    ("J5", 0x05, 0x15, "4310", 400.0, 0.28),
    ("J6", 0x06, 0x16, "4310", 400.0, 0.28),
    ("J7", 0x07, 0x17, "4310", 720.0, 1.20),   # 夹爪：行程很大（实测 -100~+260°），只做兜底限幅
)

NAMES = tuple(j[0] for j in JOINTS)

# 官方 URDF 语义限位（rad），兜底保护
URDF_LIMITS = (
    (-2.8, 2.8), (-3.14, 0.0), (-3.14, 0.0),
    (-1.87, 1.90), (-1.57, 1.57), (-3.14, 3.14),
)

# ---------------------------------------------------------------- 动作链 v2
# 实机标定后的设计约束：
#   * J7 负方向 = 张开（-90° 内力矩仅 ~0.4Nm）；正方向 +10° 起夹紧、+18° 顶死 5.2Nm
#   * J2 限位在 0 一侧，只能往负走；J3/J4 往负 = 抬臂/前伸
#   * J1(垂直轴) 与 J5(腕偏航) 只影响侧向，+1.5/+2.0 互相抵消，净侧移 ~1mm
# 夹爪：grip 标记在运行时由 gripper_calibrate() 现场标定（零点/换件后自动适配），
# 不再写死绝对角 —— 曾经写死 -120/+2/-80，重新校零后直接超程被保护拦下。
GRIP_SWEEP_MAX_DEG = 75.0     # 找闭合端最多往正方向扫多少度
GRIP_TORQUE_TH = 1.0          # Nm，超过即认为顶到位
GRIP_STALL_TORQUE = 6.5       # Nm，运行中超过即判定夹爪卡滞（实测有摩擦尖峰 7.5Nm）
GRIP_OPEN_SPAN_DEG = 120.0    # 张开：从闭合端往负方向 120°
GRIP_RELEASE_SPAN_DEG = 60.0  # 释放：从闭合端往负方向 60°

PLAN: list[dict] = [
    dict(id="S2", name="夹爪张开", grip="open"),
    # 主运动改成笛卡尔目标：臂自身坐标系里 前 +35mm / 侧 0 / 上 +105mm，用 IK 反解
    dict(id="S3", name="前伸 + 上抬", dur=6.0, ik={"forward": 35.0, "lateral": 0.0, "up": 105.0}),
    dict(id="S4", name="夹爪闭合（轻夹）", grip="close"),
    dict(id="S5", name="夹持保持", dur=1.2, hold=True, deltas={}),
    dict(id="S6", name="夹爪张开（释放）", grip="release"),
    dict(id="S7", name="回起始姿态", dur=5.0, home=True, deltas={}),
]

GRIP_VLIM = 0.60
T_GRIP, T_MAIN, T_HOLD, T_BACK, T_END = 2.4, 5.0, 3.0, 5.0, 1.5   # 兼容旧脚本

# ---- 官方 ReBot_Arm_DM.urdf 链（base_link -> link6 -> end_link），仅用于 FK 预估 ----
_URDF = (
    ((-8.416e-05, 0, 0.08465), (0, 0, 0), (0, 0, 1)),
    ((0.020084, 0.031625, 0.05555), (-1.5708, 0, 0), (0, 0, -1)),
    ((-0.264, 0, 0), (0, 0, 0), (0, 0, 1)),
    ((0.2426, -0.054, -0.001625), (0, 0, 0), (0, 0, 1)),
    ((0.078308, -0.0375, -0.03), (-1.5708, 0, 0), (0, 0, 1)),
    ((0.028008, 0, 0.04), (0, 1.5708, 0), (0, 0, 1)),
)
_END = ((0, 0, 0.15539), (0, -1.5708, 3.1415))


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def fk_full(deg7) -> tuple[np.ndarray, np.ndarray]:
    """末端完整位姿：返回 (位置 m, 旋转矩阵 3x3)。"""
    q = [math.radians(v) for v in list(deg7)[:6]]
    M = np.eye(4)
    for (xyz, r, axis), ang in zip(_URDF, q):
        A = np.eye(4)
        A[:3, :3] = _rpy(*r)
        A[:3, 3] = xyz
        M = M @ A
        ax = np.array(axis, float)
        ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R4 = np.eye(4)
        R4[:3, :3] = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
        M = M @ R4
    A = np.eye(4)
    A[:3, :3] = _rpy(*_END[1])
    A[:3, 3] = _END[0]
    M = M @ A
    return M[:3, 3].copy(), M[:3, :3].copy()


def rotvec(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 旋转向量（axis*angle）。"""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-8:
        return np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    return ax * ang


def _jac6d(q_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """6D 数值雅可比：位置(mm 归一为 m) + 旋转向量，对 6 个关节(rad) 求导。"""
    eps = 1e-5
    p0, R0 = fk_full(q_deg)
    J = np.zeros((6, 6))
    for i in range(6):
        q = np.array(q_deg, float)
        q[i] += math.degrees(eps)
        p1, R1 = fk_full(q)
        J[:3, i] = (p1 - p0) / eps
        J[3:, i] = rotvec(R1 @ R0.T) / eps
    return p0, R0, J


def ik_pose(q_start_deg, p_target, R_target,
            iters: int = 600, lam: float = 1e-3, max_step: float = 0.08,
            pull: float = 0.005) -> tuple[np.ndarray, float, float]:
    """6D IK（位置+姿态）。返回 (6 关节角deg, 位置残差m, 姿态残差rad)。"""
    q0 = np.deg2rad(np.array(list(q_start_deg)[:6], float))
    q = q0.copy()
    perr = rerr = 1e9
    for _ in range(iters):
        p, R, J = _jac6d(np.rad2deg(q))
        ep = np.asarray(p_target, float) - p
        er = rotvec(np.asarray(R_target, float) @ R.T)
        perr, rerr = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
        if perr < 3e-4 and rerr < 3e-3:
            break
        e = np.concatenate([ep, er])
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
        n = float(np.linalg.norm(dq))
        if n > max_step:
            dq *= max_step / n
        q = q + dq - pull * (q - q0)
        for i, (lo, hi) in enumerate(URDF_LIMITS):
            q[i] = min(max(q[i], lo), hi)
    return np.rad2deg(q), perr, rerr


def pose_shift(q_deg7, dz_mm: float = 0.0, dyaw_deg: float = 0.0,
               dxyz_mm=(0.0, 0.0)) -> tuple[np.ndarray, float, float]:
    """在当前位姿上：竖直平移 dz、绕竖直轴自转 dyaw（TCP 位置不变）。
    返回 (新的 7 关节角deg, 位置残差m, 姿态残差rad)。"""
    p, R = fk_full(q_deg7)
    c, s = math.cos(math.radians(dyaw_deg)), math.sin(math.radians(dyaw_deg))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    p2 = p + np.array([dxyz_mm[0] / 1000.0, dxyz_mm[1] / 1000.0, dz_mm / 1000.0])
    R2 = Rz @ R
    q6, perr, rerr = ik_pose(q_deg7, p2, R2)
    return np.array(list(q6) + [list(q_deg7)[6]]), perr, rerr


def fk_tcp(deg7) -> list[float]:
    """正解：6 个关节角(deg) -> 末端(x,y,z) 米。用于预估位移，不做实时控制。"""
    q = [math.radians(v) for v in deg7[:6]]
    M = np.eye(4)
    for (xyz, r, axis), ang in zip(_URDF, q):
        A = np.eye(4)
        A[:3, :3] = _rpy(*r)
        A[:3, 3] = xyz
        M = M @ A
        ax = np.array(axis, float)
        ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
        R4 = np.eye(4)
        R4[:3, :3] = R
        M = M @ R4
    A = np.eye(4)
    A[:3, :3] = _rpy(*_END[1])
    A[:3, 3] = _END[0]
    M = M @ A
    return [float(v) for v in M[:3, 3]]


def plan_summary(scale: float = 1.0) -> list[dict]:
    """给前端用的计划摘要（含缩放后的增量）。"""
    out = []
    for st in PLAN:
        out.append({
            "id": st["id"],
            "name": st["name"],
            "dur": st.get("dur"),
            "hold": bool(st.get("hold")),
            "home": bool(st.get("home")),
            "grip": st.get("grip"),
            "ik": ({k: round(float(v) * scale, 2) for k, v in (st.get("ik") or {}).items()}
                   or None),
            "deltas": {k: round(v * scale, 3) for k, v in st.get("deltas", {}).items()},
            "abs": {k: round(v, 3) for k, v in (st.get("abs") or {}).items()},
        })
    return out


class ChainAborted(RuntimeError):
    pass


class SegmentStalled(RuntimeError):
    """某个关节卡滞（力矩顶住但位置跟不上）。夹爪会用它做"跳过夹爪段"。"""


@dataclass
class Snapshot:
    t: float
    deg: list[float | None]
    vel: list[float | None] = field(default_factory=list)
    torq: list[float | None] = field(default_factory=list)
    status: list[int] = field(default_factory=list)
    fault: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "t": round(self.t, 3),
            "names": list(NAMES),
            "deg": self.deg,
            "vel": self.vel,
            "torq": self.torq,
            "status": self.status,
            "fault": self.fault,
        }


def s_curve(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * s)


def _pct_tick(emit, phase: str, duration: float):
    """按时间估算百分比进度，供 stream/hold 的 on_tick 用。"""
    t0 = time.time()

    def tick():
        emit(phase, min(100.0, (time.time() - t0) / max(duration, 0.1) * 100.0), None)

    return tick


class Arm:
    """DM 串口机械臂封装：读状态 / 使能失能 / 平滑走到目标角。"""

    def __init__(self, port: str = DEFAULT_PORT, baud: int = BAUD):
        self.port = port
        self.baud = baud
        self.ctrl = Controller.from_dm_serial(port, baud)
        self.motors = [self.ctrl.add_damiao_motor(mid, fid, model)
                       for _n, mid, fid, model, _lim, _v in JOINTS]
        self.lims = np.deg2rad(np.array([j[4] for j in JOINTS]))
        self.vlims = np.array([j[5] for j in JOINTS])
        self.t0 = time.time()

    # ---------- 生命周期 ----------
    def close(self) -> None:
        try:
            self.ctrl.close_bus()
        except Exception:
            pass
        try:
            self.ctrl.close()
        except Exception:
            pass

    # ---------- 状态 ----------
    def read_fast(self, spins: int = 2) -> Snapshot:
        """轻量读：只发一轮查询，够示教循环用（~50Hz）。"""
        st = [None] * len(self.motors)
        for _ in range(max(1, spins)):
            for m in self.motors:
                m.request_feedback()
            for _ in range(2):
                try:
                    self.ctrl.poll_feedback_once()
                except Exception:
                    pass
            time.sleep(0.002)
            for i, m in enumerate(self.motors):
                s = m.get_state()
                if s is not None:
                    st[i] = s
        return Snapshot(
            t=time.time() - self.t0,
            deg=[None if s is None else round(math.degrees(s.pos), 3) for s in st],
            vel=[None if s is None else round(s.vel, 3) for s in st],
            torq=[None if s is None else round(s.torq, 3) for s in st],
            status=[-1 if s is None else s.status_code for s in st],
            fault=[-1 if s is None else s.fault_code for s in st],
        )

    def read(self, spins: int = 3) -> Snapshot:
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
            if all(s is not None for s in st):
                break
        return Snapshot(
            t=time.time() - self.t0,
            deg=[None if s is None else round(math.degrees(s.pos), 2) for s in st],
            vel=[None if s is None else round(s.vel, 3) for s in st],
            torq=[None if s is None else round(s.torq, 3) for s in st],
            status=[-1 if s is None else s.status_code for s in st],
            fault=[-1 if s is None else s.fault_code for s in st],
        )

    # ---------- 使能 ----------
    def enable(self) -> None:
        for m in self.motors:
            m.enable()
        time.sleep(0.25)
        for m in self.motors:
            try:
                m.ensure_mode(Mode.POS_VEL, 1000)
            except Exception:
                pass

    def disable(self) -> None:
        for m in self.motors:
            try:
                m.disable()
            except Exception:
                pass

    def set_mode_mit(self) -> None:
        for m in self.motors:
            try:
                m.ensure_mode(Mode.MIT, 500)
            except Exception:
                pass

    # ---------- 运动 ----------
    def hold(self, target: np.ndarray, seconds: float,
             stop: threading.Event | None = None, on_tick=None) -> None:
        t_end = time.time() + seconds
        while time.time() < t_end:
            if stop is not None and stop.is_set():
                raise ChainAborted("收到中止指令")
            for m, c, v in zip(self.motors, target, self.vlims):
                m.send_pos_vel(float(c), float(v))
            for m in self.motors:
                m.request_feedback()
            try:
                self.ctrl.poll_feedback_once()
            except Exception:
                pass
            if on_tick:
                on_tick()
            time.sleep(0.02)

    def stream(self, target: np.ndarray, duration: float,
               stop: threading.Event | None = None, on_tick=None,
               stall_joint: int | None = None,
               stall_torque: float = GRIP_STALL_TORQUE,
               stall_lag_deg: float = 25.0,
               stall_grace: float = 0.40) -> Snapshot:
        # 电机若处于失能状态，指令不会被执行 → 先使能
        s0 = self.read_fast(spins=1)
        if s0.status and all(v == 0 for v in s0.status if v is not None):
            self.enable()
        start = self.read()
        if any(d is None for d in start.deg):
            raise RuntimeError("有电机读不到反馈")
        q0 = np.deg2rad([d for d in start.deg])
        if np.any(np.abs(target) > self.lims + 1e-9):
            raise RuntimeError(f"目标角度超出限幅: {np.rad2deg(target).round(1)}")
        # 行程太长而时长太短时，自动加长时长（否则会因跟随不上而误判"跟踪失败"）
        # s 曲线峰值速度 = 平均速度 × π/2 ≈ 1.57，时长必须按峰值留余量，否则指令会跑在电机前面
        need = float(np.max(np.abs(target - q0) / np.maximum(self.vlims, 1e-6))) * 1.57 * 1.25
        duration = max(float(duration), need)
        n = max(int(duration * RATE_HZ), 1)
        dt = 1.0 / RATE_HZ
        stuck = 0
        stall_cnt = 0
        q_last = q0.copy()
        max_lead = math.radians(MAX_LEAD_DEG)
        stall_need = max(1, int(stall_grace * RATE_HZ))
        for k in range(1, n + 1):
            if stop is not None and stop.is_set():
                raise ChainAborted("收到中止指令")
            cmd = q0 + (target - q0) * s_curve(k / n)
            # 指令不许跑到实际前面去：钳在实测 ±max_lead 内（误差不会累积）
            cmd = np.clip(cmd, q_last - max_lead, q_last + max_lead)
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
            q = np.array([(m.get_state().pos if m.get_state() else np.nan)
                          for m in self.motors])
            if not np.isnan(q).any():
                q_last = np.where(np.isnan(q), q_last, q)
                tq6 = [abs(m.get_state().torq) if m.get_state() else 0.0
                       for m in self.motors[:6]]
                if tq6 and max(tq6) > STREAM_TORQUE_ABORT:
                    raise RuntimeError(
                        f"关节力矩饱和 {max(tq6):.1f} Nm（>{STREAM_TORQUE_ABORT}），举不动或顶住了，已停止")
                errs = np.abs(q - cmd)
                # 夹爪段：用「力矩 或 该关节滞后」单独判卡滞，并把它从通用判据里剔除
                if stall_joint is not None:
                    st = self.motors[stall_joint].get_state()
                    tq = abs(st.torq) if st is not None else 0.0
                    lag = float(errs[stall_joint])
                    if tq > stall_torque or lag > math.radians(stall_lag_deg):
                        stall_cnt += 1
                        if stall_cnt >= stall_need:
                            raise SegmentStalled(
                                f"{NAMES[stall_joint]} 卡滞：力矩 {tq:.1f} Nm / "
                                f"滞后 {math.degrees(lag):.0f}°（阈值 {stall_torque} Nm / "
                                f"{stall_lag_deg}°），停在 "
                                f"{math.degrees(q[stall_joint]):+.1f}°")
                    else:
                        stall_cnt = 0
                    errs = np.delete(errs, stall_joint)
                err = float(np.max(errs)) if errs.size else 0.0
                stuck = stuck + 1 if err > 0.5 else 0
                if stuck > 40:
                    raise RuntimeError(
                        f"关节跟踪失败，误差 {math.degrees(err):.0f}°，自动中止")
            if on_tick:
                on_tick()
        # 收尾：指令就停在终点，等臂跟上来（最多 8 秒）
        t_end = time.time() + 8.0
        while time.time() < t_end:
            for m, c, v in zip(self.motors, target, self.vlims):
                m.send_pos_vel(float(c), float(v))
            for m in self.motors:
                m.request_feedback()
            try:
                self.ctrl.poll_feedback_once()
            except Exception:
                pass
            time.sleep(dt)
            q = np.array([(m.get_state().pos if m.get_state() else np.nan)
                          for m in self.motors])
            if not np.isnan(q).any() and float(np.max(np.abs(q - target))) < math.radians(2.0):
                break
        return self.read()


def gripper_calibrate(arm: "Arm", stop: threading.Event | None = None, emit=None) -> dict:
    """
    现场标定夹爪（零点/机构一致性会变，不能写死绝对角）：
      1) 以 0.6°/12ms 慢慢往正方向压，力矩超阈值且位置跟得上 → 记「闭合端」
      2) 张开端 = 闭合端 - GRIP_OPEN_SPAN_DEG（限幅内）
      3) 释放位 = 闭合端 - GRIP_RELEASE_SPAN_DEG
    """
    m7 = arm.motors[6]

    def fresh():
        m7.request_feedback()
        try:
            arm.ctrl.poll_feedback_once()
        except Exception:
            pass
        time.sleep(0.03)
        return m7.get_state()

    st = fresh()
    start = st.pos if st else 0.0
    cmd = start
    step = math.radians(0.6)
    closed = None
    t_end = time.time() + 9.0
    while math.degrees(cmd - start) < GRIP_SWEEP_MAX_DEG and time.time() < t_end:
        if stop is not None and stop.is_set():
            raise ChainAborted("收到中止指令")
        cmd += step
        m7.send_pos_vel(float(cmd), 0.9)
        m7.request_feedback()
        try:
            arm.ctrl.poll_feedback_once()
        except Exception:
            pass
        time.sleep(0.012)
        if emit:
            emit("S1.5 夹爪标定", min(100.0, abs(math.degrees(cmd - start)) / GRIP_SWEEP_MAX_DEG * 100.0), None)
        s = m7.get_state()
        if s is None:
            continue
        if abs(s.torq) > GRIP_STALL_TORQUE:
            raise SegmentStalled(
                f"J7 寻找闭合端时卡滞：力矩 {s.torq:+.1f} Nm，位置 {math.degrees(s.pos):+.1f}°")
        if abs(s.torq) > GRIP_TORQUE_TH and abs(s.pos - cmd) < math.radians(3.0):
            closed = s.pos
            break
    if closed is None:
        s = fresh()
        closed = s.pos if s else cmd
    lim = float(arm.lims[6])
    open_ = max(closed - math.radians(GRIP_OPEN_SPAN_DEG), -lim)
    release = max(closed - math.radians(GRIP_RELEASE_SPAN_DEG), -lim)
    return {
        "closed": closed, "open": open_, "release": release,
        "deg": {k: round(math.degrees(v), 2)
                for k, v in (("closed", closed), ("open", open_), ("release", release))},
    }


def _need_duration(q_now: np.ndarray, target: np.ndarray, cfg: float | None,
                   vlims: np.ndarray) -> float:
    """按行程和速度上限算一个不会触发跟随保护的最短时长（再乘 1.25 余量）。"""
    need = float(np.max(np.abs(target - q_now) / np.maximum(vlims, 1e-6))) * 1.57 * 1.25
    return max(float(cfg) if cfg else 1.0, need)


def _jac6(q_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """数值雅可比（对弧度求导）：末端位置对 6 个关节角。"""
    eps = 1e-5
    p0 = np.array(fk_tcp(q_deg))
    J = np.zeros((3, 6))
    for i in range(6):
        q = np.array(q_deg, float)
        q[i] += math.degrees(eps)
        J[:, i] = (np.array(fk_tcp(q)) - p0) / eps
    return p0, J


def ik_delta(q_start_deg: list[float], target_m: np.ndarray,
             iters: int = 500, lam: float = 1e-3,
             max_step: float = 0.05, pull: float = 0.01) -> tuple[np.ndarray, float]:
    """阻尼最小二乘 IK（内部全用弧度）：求末端到 target_m 的 6 关节角(deg)。
    只解位置，用「向起始位回拉 + 步长上限」抑制零空间漂移。返回 (deg 角, 残余误差 m)。"""
    q0 = np.deg2rad(np.array(q_start_deg, float))
    q = q0.copy()
    err = 1e9
    for _ in range(iters):
        p, J = _jac6(np.rad2deg(q))
        e = np.asarray(target_m, float) - p
        err = float(np.linalg.norm(e))
        if err < 5e-4:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), e)
        n = float(np.linalg.norm(dq))
        if n > max_step:
            dq *= max_step / n
        q = q + dq - pull * (q - q0)
        for i, (lo, hi) in enumerate(URDF_LIMITS):
            q[i] = min(max(q[i], lo), hi)
    return np.rad2deg(q), err


def grip_and_ik_targets(q_start_deg: list[float], step: dict, scale: float) -> tuple[dict, str]:
    """把 PLAN 里一段的 ik/deltas 解析成「相对起始位的关节偏移(deg)」。"""
    offsets = {}
    note = ""
    if step.get("ik"):
        q1 = math.radians(q_start_deg[0])
        fwd = float(step["ik"].get("forward", 0.0)) * scale / 1000.0
        lat = float(step["ik"].get("lateral", 0.0)) * scale / 1000.0
        up = float(step["ik"].get("up", 0.0)) * scale / 1000.0
        p0 = np.array(fk_tcp(q_start_deg))
        tw = np.array([fwd * math.cos(q1) - lat * math.sin(q1),
                       fwd * math.sin(q1) + lat * math.cos(q1),
                       up])
        q_sol, err = ik_delta(q_start_deg[:6], p0 + tw)
        for i in range(6):
            offsets[NAMES[i]] = float(q_sol[i] - q_start_deg[i])
        note = f"IK 残差 {err*1000:.1f} mm"
    for k, v in (step.get("deltas") or {}).items():
        offsets[k] = float(v) * scale
    return offsets, note


# ---- URDF 惯性参数（重力补偿用）：(链路, mass, com_local[m]) ----
_LINKS = (
    (1, 0.1613, (0.000113615, -0.000616320, 0.023647637)),
    (2, 1.3266, (-0.132256223, -0.003061704, -0.030830697)),
    (3, 0.8353, (0.121040036, -0.053621108, -0.031013785)),
    (4, 0.5200, (0.060820096, -0.051171191, -0.030299459)),
    (5, 0.3830, (-0.005028021, 0.000001739, 0.038623324)),
    (6, 0.3663, (0.000003764, -0.000100909, 0.025330861)),
    (7, 0.5000, (-0.073765430, -0.000009508, 0.000007043)),   # end_link（含夹爪）
)
G_ACC = 9.81


def _frames(q_deg6):
    """返回 ({joint_index: (axis_world, point_world)}, {link_index: (R,t)}) —— 重力补偿用。"""
    q = [math.radians(v) for v in q_deg6]
    M = np.eye(4)
    axes = {}
    links = {}
    for i, (xyz, r, axis) in enumerate(_URDF):
        R0 = _rpy(*r)
        A = np.eye(4)
        A[:3, :3] = R0
        A[:3, 3] = xyz
        M = M @ A
        ax = np.array(axis, float)
        ax /= np.linalg.norm(ax)
        axes[i] = (M[:3, :3] @ ax, M[:3, 3].copy())        # 该关节在世界系的方向 + 轴上一点
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + math.sin(q[i]) * K + (1 - math.cos(q[i])) * (K @ K)
        R4 = np.eye(4)
        R4[:3, :3] = R
        M = M @ R4
        links[i + 1] = (M[:3, :3].copy(), M[:3, 3].copy())  # link(i+1) 的坐标系
    A = np.eye(4)
    A[:3, :3] = _rpy(*_END[1])
    A[:3, 3] = _END[0]
    Me = M @ A
    links[7] = (Me[:3, :3].copy(), Me[:3, 3].copy())        # end_link
    return axes, links


def gravity_tau(q_deg7) -> np.ndarray:
    """按 URDF 质量/质心算 6 个关节的重力力矩（Nm），用于浮动示教。"""
    q6 = list(q_deg7)[:6]
    axes, links = _frames(q6)
    tau = np.zeros(7)
    g_vec = np.array([0.0, 0.0, -G_ACC])
    for link_idx, mass, com_local in _LINKS:
        R, t = links[link_idx]
        com = R @ np.array(com_local) + t
        F = mass * g_vec
        for j in range(6):                     # 只累加该链路的上游关节
            if link_idx <= j:                  # link(j+1) 及其下游才受 joint j 承载
                continue
            ax, pt = axes[j]
            tau[j] += float(np.dot(ax, np.cross(com - pt, F)))
    return tau


TEACH_HZ = 50.0
TEACH_DEADBAND_DEG = 0.8    # 小于这个偏差认为没在推（防止自发漂移）
TEACH_FOLLOW = 0.35         # 每拍设定值向实测位置靠拢的比例（越大越"软"）
TEACH_VLIM = 1.2            # 拖拽时允许的最大速度 rad/s
TEACH_GRIP_STEP = 0.4       # 夹爪每拍增加的角度（50Hz → 20°/s）
TEACH_GRIP_FORCE = 1.5      # Nm，夹紧时力矩到这么多就认为"夹到东西/到位"
TEACH_OPEN_FORCE = 2.5      # Nm，张开时遇到阻力就停
TEACH_OPEN_SPAN = 110.0     # deg，张开最多走这么多
TEACH_CLOSE_SPAN = 300.0    # deg，夹紧的兜底上限（基本等于"一直夹到底"）
TEACH_GRIP_SQUEEZE = 0.5    # deg，停住后再多压一点，保证持续有力
TEACH_STALL_LAG = 12.0      # deg，设定值超前实测这么多 = 夹不动了
TEACH_STALL_TICKS = 8       # 0.16s 连续判定
TEACH_SPAN_MAX_DEG = 90.0   # deg，拖拽时设定值相对起点的最大行程（硬保护）
TEACH_ASSIST_KD = 0.8       # assist 模式的 MIT 阻尼（kp 恒为 0）
TEACH_MODES = {"free": "完全卸力（徒手搬）", "assist": "助力（抵消当前负重）",
               "follow": "按住跟随（位置环）",
               "leader": "主从示教（拖另一台臂，本臂跟随）"}
TEACH_VEL_ABORT = 4.0       # rad/s，超过立即退出示教
TEACH_DEV_ABORT = 150.0     # deg，任一关节偏离起点超过此值立即退出


def teach_loop(arm: "Arm", ctl: dict, emit=None) -> dict:
    """
    示教 + 录制。三种模式（ctl["mode"]）：
      free   完全卸力：电机失能，随便徒手搬（重力全由你托着）；编码器照常读，夹爪动作时只临时给 J7 上电
      assist 助力：MIT(kp=0, kd 小, tau=进示教时实测的静态力矩偏置) —— 抵消掉当前姿态的负重
      follow 按住跟随：位置环不动，只在你"按住拖动"时设定值追随（deadman）
    """
    mode = ctl.get("mode", "free")
    ctl.setdefault("samples", [])
    ctl.setdefault("events", [])
    ctl.setdefault("notes", [])
    ctl.setdefault("sp6", None)
    t0 = time.time()
    period = 1.0 / TEACH_HZ
    reason = "stopped"

    if mode == "free":
        arm.disable()
        snap = arm.read()
    else:
        arm.enable()
        time.sleep(0.15)
        snap = arm.read()
    q0 = np.array([d if d is not None else 0.0 for d in snap.deg])
    sp = q0.copy()
    bias = np.zeros(7)
    if mode == "assist":
        acc = []
        for _ in range(20):
            s = arm.read_fast()
            if s.torq and all(v is not None for v in s.torq):
                acc.append(s.torq)
            time.sleep(0.01)
        if acc:
            bias = np.array(acc).mean(axis=0)
        arm.set_mode_mit()

    i = 0
    while not ctl["stop"].is_set():
        tick = time.time()
        snap = arm.read_fast()
        q = np.array([d if d is not None else 0.0 for d in snap.deg])
        vmax = max([abs(v) for v in (snap.vel or []) if v is not None] or [0.0])
        torq7 = float(snap.torq[6]) if snap.torq and snap.torq[6] is not None else 0.0

        if mode != "free":
            dev = float(np.max(np.abs(q - q0)))
            if vmax > TEACH_VEL_ABORT or dev > TEACH_DEV_ABORT:
                reason = f"安全退出：速度 {vmax:.1f} rad/s / 偏离 {dev:.0f}°"
                break

        # ---- 关节指令 ----
        if mode == "assist":
            for j in range(6):
                arm.motors[j].send_mit(float(np.deg2rad(q[j])), 0.0, 0.0,
                                       TEACH_ASSIST_KD, float(bias[j]))
        elif mode == "follow":
            follow_on = bool(ctl.get("follow"))
            for j in range(6):
                if not follow_on:
                    continue
                d = q[j] - sp[j]
                if abs(d) > TEACH_DEADBAND_DEG:
                    step = TEACH_FOLLOW * (d - math.copysign(TEACH_DEADBAND_DEG, d))
                    sp_new = sp[j] + step
                    if abs(sp_new - q0[j]) <= TEACH_SPAN_MAX_DEG:
                        sp[j] = sp_new
            for j in range(6):
                arm.motors[j].send_pos_vel(float(np.deg2rad(sp[j])), TEACH_VLIM)

        # ---- 夹爪 ----
        gt_mode = ctl.get("grip_mode")
        gh = ctl.get("grip_hold")
        sp6 = ctl.get("sp6")
        if sp6 is None:
            sp6 = q[6]
        grip_needs_power = bool(gt_mode) or (gh is not None)
        if mode == "free":
            if grip_needs_power and not ctl.get("j7_on"):
                arm.motors[6].enable()
                try:
                    arm.motors[6].ensure_mode(Mode.POS_VEL, 500)
                except Exception:
                    pass
                ctl["j7_on"] = True
            elif not grip_needs_power and ctl.get("j7_on"):
                arm.motors[6].disable()
                ctl["j7_on"] = False
            elif not grip_needs_power and snap.status and snap.status[6] == 1:
                arm.motors[6].disable()      # 之前有帧丢失，补发一次断电

        if gt_mode == "close":
            if ctl.get("close_from") is None:
                ctl["close_from"] = sp6
            sp6 += TEACH_GRIP_STEP
            lag = sp6 - q[6]
            ctl["lag_cnt"] = (ctl.get("lag_cnt", 0) + 1) if lag > TEACH_STALL_LAG else 0
            done = (abs(torq7) > TEACH_GRIP_FORCE
                    or ctl["lag_cnt"] > TEACH_STALL_TICKS
                    or (sp6 - ctl["close_from"]) > TEACH_CLOSE_SPAN)
            if done:
                why = ("力矩" if abs(torq7) > TEACH_GRIP_FORCE
                       else ("夹不动" if ctl["lag_cnt"] > TEACH_STALL_TICKS else "走满行程"))
                # 停在「再压一点」的位置：位置环持续输出力矩，保证夹持稳定
                ctl.update(grip_mode=None, grip_hold=sp6 + TEACH_GRIP_SQUEEZE,
                           lag_cnt=0, close_from=None)
                ctl["events"].append({"t": round(time.time() - t0, 2), "action": "close",
                                      "deg": round(float(q[6]), 1),
                                      "hold": round(float(ctl["grip_hold"]), 1),
                                      "torque": round(torq7, 2), "why": why})
        elif gt_mode == "open":
            ctl["open_ticks"] = ctl.get("open_ticks", 0) + 1
            sp6 -= TEACH_GRIP_STEP
            if ctl.get("open_from") is None:
                ctl["open_from"] = sp6
            lag = q[6] - sp6
            ctl["lag_cnt"] = (ctl.get("lag_cnt", 0) + 1) if lag > TEACH_STALL_LAG else 0
            # 开始 0.4s 内不判力矩（此刻还在夹持，力矩本来就是高的）
            armed = ctl["open_ticks"] > (0.4 * TEACH_HZ)
            if (armed and abs(torq7) > TEACH_OPEN_FORCE) \
               or (armed and ctl["lag_cnt"] > TEACH_STALL_TICKS) \
               or (ctl["open_from"] - sp6) > TEACH_OPEN_SPAN:
                ctl.update(grip_mode=None, grip_hold=sp6 + TEACH_GRIP_STEP,
                           open_from=None, lag_cnt=0, open_ticks=0)
                ctl["events"].append({"t": round(time.time() - t0, 2), "action": "open",
                                      "deg": round(float(q[6]), 1),
                                      "torque": round(torq7, 2)})
        elif gh is not None:
            sp6 = gh
        elif mode == "follow":
            sp6 = q[6]
        ctl["sp6"] = sp6
        if mode != "free" or ctl.get("j7_on"):
            arm.motors[6].send_pos_vel(float(np.deg2rad(sp6)), 0.4)

        ctl["samples"].append([round(time.time() - t0, 4)] + [round(float(v), 3) for v in q])
        if emit and i % 10 == 0:
            emit(time.time() - t0, float(q[6]), [0.0], vmax)
        i += 1
        dt = period - (time.time() - tick)
        if dt > 0:
            time.sleep(dt)
    ctl["notes"].append(reason)
    return {"reason": reason, "mode": mode, "samples": ctl["samples"],
            "events": ctl["events"], "notes": ctl["notes"]}


# ------------------------------------------------- 主从示教（另一台臂当主臂拖动）
# 主臂：Feetech STS3215 总线舵机 ×6（USB-CH343 @1M），失能状态下徒手拖
# 从臂：本文件里的 reBot B601-DM（6 轴 + 夹爪）
# 两台臂 DH / 零位 / 减速比都不同，所以不做绝对角拷贝，而是「相对增量映射」：
#     start 时记下 主臂 L0、从臂 S0
#     target_j = S0_j + sign_j * gain_j * (L_j - L0_j)
# 逐个关节可单独改 sign/gain（leader_map.json），一上手不会突跳；配好绝对行程
# 标定后也可以把 use_abs 打开走比例映射（L 的 [lmin,lmax] → S 的 [smin,smax]）。
LEADER_GAIN_DEFAULT = 1.0
LEADER_MAX_TRAVEL_DEG = 130.0    # 相对起点的最大行程（硬保护）
LEADER_VLIM_SCALE = 0.55         # 从臂速度上限 = vlims × 该比例
LEADER_LAG_ABORT_DEG = 45.0      # 从臂跟不上主臂这么多度，持续 0.5s → 退出
LEADER_TORQUE_ABORT = 15.0       # Nm，任一关节超过 → 退出
LEADER_READ_TIMEOUT = 0.35       # s，主臂读不到数超过这个时间 → 退出
LEADER_HOLD_TICKS = 25           # 开始后先原地保持 ~0.5s（等从臂稳）


def leader_follow_loop(arm: "Arm", ctl: dict, leader, mapper, emit=None) -> dict:
    """主从示教：拖主臂 → 从臂按关节增量跟随，同时按老格式录制样本。

    ctl 里用到的键：
        stop        threading.Event
        gains       [6] 每关节增益（默认 1.0）
        signs       [6] 每关节方向（默认 +1）
        max_travel  相对起点最大行程 deg
        vlim_scale  从臂速度上限比例
        use_abs     True 时走绝对比例映射（需要 leader_map.json 已标定 lmin/lmax）
        j7_hold     True 时夹爪保持当前角（否则不碰 J7）
    """
    ctl.setdefault("samples", [])
    ctl.setdefault("events", [])
    ctl.setdefault("notes", [])
    gains = list(ctl.get("gains") or [LEADER_GAIN_DEFAULT] * 6)
    signs = list(ctl.get("signs") or [1] * 6)
    max_travel = float(ctl.get("max_travel", LEADER_MAX_TRAVEL_DEG))
    vscale = float(ctl.get("vlim_scale", LEADER_VLIM_SCALE))
    use_abs = bool(ctl.get("use_abs"))
    period = 1.0 / TEACH_HZ

    t0 = time.time()
    reason = "stopped"

    # 从臂使能并读完当前位姿
    # 关键：DM 电机使能后会先跑到「上一次下发的目标」，可能与当前姿态差很多（卸力后会下垂）。
    # 所以使能前先按实测角下发一拍 pos_vel（此时没使能，不会动），使能后就不会弹回旧目标。
    pre = arm.read_fast(spins=2)
    for j, d in enumerate(pre.deg):
        if d is not None:
            try:
                arm.motors[j].send_pos_vel(float(np.deg2rad(d)), 0.3)
            except Exception:
                pass
    arm.enable()
    time.sleep(0.25)
    snap = arm.read()
    if any(d is None for d in snap.deg):
        raise RuntimeError("从臂有关节读不到反馈，先检查 DM 总线")
    q0 = np.array([float(d) for d in snap.deg])          # 含 J7
    sp = q0.copy()                                        # 从臂指令（deg）
    urdf_lo, urdf_hi = core_urdf_limits_deg()

    # 主臂起点
    l0 = None
    t_read = time.time()
    while l0 is None:
        lead = mapper.lead_deg(leader.read_counts())
        if all(v is not None for v in lead):
            l0 = np.array(lead, float)
        elif time.time() - t_read > 2.0:
            raise RuntimeError("主臂读不到角度：检查 COM 口 / 波特率 / 供电")
        else:
            time.sleep(0.01)
        if ctl["stop"].is_set():
            return {"reason": "stopped", "mode": "leader", "samples": [],
                    "events": [], "notes": ["未开始就停止"]}

    ctl["events"].append({"t": 0.0, "action": "leader_start",
                          "leader0": [round(float(v), 2) for v in l0],
                          "slave0": [round(float(v), 2) for v in q0]})

    i = 0
    bad_read_since = None
    lag_since = None
    mp = ctl.get("map")            # 指向 server 的 leader_map（可被 /api/leader/map 实时改）
    sign_key = None

    def _signs_now():
        """方向/增益：优先取实时 map，其次 ctl 里的初值。"""
        if mp and mp.get("joints"):
            js = mp["joints"]
            sg = [int(js[j].get("sign", 1)) if j < len(js) else 1 for j in range(6)]
            gn = [float(js[j].get("gain", 1.0)) if j < len(js) else 1.0 for j in range(6)]
            return sg, gn
        return list(signs), list(gains)

    try:
        while not ctl["stop"].is_set():
            tick = time.time()
            lead = mapper.lead_deg(leader.read_counts())
            ok_lead = all(v is not None for v in lead)

            if not ok_lead:
                bad_read_since = bad_read_since or time.time()
                if time.time() - bad_read_since > LEADER_READ_TIMEOUT:
                    reason = "主臂掉线（读不到角度），已安全退出"
                    break
            else:
                bad_read_since = None

            signs, gains = _signs_now()

            # ---- 目标角 ----
            if ok_lead:
                L = np.array(lead, float)
                # 方向/增益被改过 → 以「此刻」为新锚点，避免姿态突跳
                key = (tuple(signs), tuple(round(g, 4) for g in gains))
                if sign_key is None:
                    sign_key = key
                elif key != sign_key:
                    sign_key = key
                    l0 = L.copy()
                    q0[:6] = np.array([d if d is not None else sp[j]
                                       for j, d in enumerate(snap.deg)])[:6] if snap else sp[:6]
                    sp[:6] = q0[:6]
                    ctl["events"].append({"t": round(time.time() - t0, 2),
                                          "action": "reanchor",
                                          "signs": signs,
                                          "leader": [round(float(v), 2) for v in L],
                                          "slave": [round(float(v), 2) for v in sp[:6]]})
                if use_abs:
                    tgt6 = np.array([v if v is not None else sp[j]
                                     for j, v in enumerate(mapper.to_slave(lead)[:6])], float)
                else:
                    dlt = (L - l0) * np.array(signs, float) * np.array(gains, float)
                    dlt = np.clip(dlt, -max_travel, max_travel)
                    tgt6 = q0[:6] + dlt
                tgt6 = np.clip(tgt6, urdf_lo, urdf_hi)
                # 速度限幅（每拍最多走 vlim*dt）
                step_max = arm.vlims[:6] * vscale * period * 57.29578
                tgt6 = sp[:6] + np.clip(tgt6 - sp[:6], -step_max, step_max)
                sp[:6] = tgt6
                # 给前端/诊断看：主臂角与本拍下发的指令角
                ctl["lead_deg"] = [round(float(v), 2) for v in L]
                ctl["sp_deg"] = [round(float(v), 2) for v in sp[:6]]

            # 前 0.5s 只保持，不发跳变指令
            if i >= LEADER_HOLD_TICKS:
                for j in range(6):
                    arm.motors[j].send_pos_vel(float(np.deg2rad(sp[j])),
                                               float(arm.vlims[j] * vscale))

            snap = arm.read_fast(spins=1)
            q = np.array([d if d is not None else sp[j] for j, d in enumerate(snap.deg)])
            tau = [abs(v) for v in (snap.torq or []) if v is not None]
            tau_max = max(tau) if tau else 0.0
            lag = float(np.max(np.abs(q[:6] - sp[:6])))
            vmax = max([abs(v) for v in (snap.vel or []) if v is not None] or [0.0])

            # ---- 夹爪 J7：和徒手示教同一套"爬行 + 力矩/滞停判定" ----
            q7 = float(q[6])
            torq7 = float((snap.torq or [None] * 7)[6] or 0.0)
            sp6 = grip_tick(arm, ctl, q7, torq7, t0, mode="leader")

            if tau_max > LEADER_TORQUE_ABORT:
                reason = f"从臂顶住：力矩 {tau_max:.1f} Nm > {LEADER_TORQUE_ABORT}"
                break
            if lag > LEADER_LAG_ABORT_DEG:
                lag_since = lag_since or time.time()
                if time.time() - lag_since > 0.5:
                    reason = f"从臂跟不上：最大差 {lag:.0f}°（可能被挡住或限幅）"
                    break
            else:
                lag_since = None

            # ---- 录制：跟随模式和录制解耦，只有 ctl["record"] 为真才攒样本 ----
            if ctl.get("record"):
                if ctl.get("rec_t0") is None:
                    ctl["rec_t0"] = time.time()
                    ctl["events"].append({"t": round(time.time() - t0, 2),
                                          "action": "record_start",
                                          "slave0": [round(float(v), 2) for v in q[:7]]})
                ctl["samples"].append(
                    [round(time.time() - ctl["rec_t0"], 4)]
                    + [round(float(v), 3) for v in q[:7]])
            elif ctl.get("rec_t0") is not None:
                ctl["events"].append({"t": round(time.time() - t0, 2),
                                      "action": "record_stop",
                                      "samples": len(ctl["samples"])})
                ctl["rec_t0"] = None
            if emit and i % 5 == 0:
                emit(time.time() - t0, float(q[6]), [tau_max], vmax)
            i += 1
            dt = period - (time.time() - tick)
            if dt > 0:
                time.sleep(dt)
    finally:
        ctl["notes"].append(reason)
        # 收尾：默认不卸力，原地锁住（stop_teach 里也会再兜一次）
        if ctl.get("keep_enabled", True):
            try:
                hold_current(arm)
            except Exception:
                pass
        else:
            try:
                arm.disable()
            except Exception:
                pass
    return {"reason": reason, "mode": "leader", "samples": ctl["samples"],
            "events": ctl["events"], "notes": ctl["notes"]}


def grip_tick(arm: "Arm", ctl: dict, q7: float, torq7: float, t0: float,
              mode: str = "leader") -> float:
    """夹爪 J7 的单拍控制：crawl 到闭合/张开，遇力矩或滞停就停并保持。

    ctl 键：grip_mode("close"/"open"/None)、grip_hold、sp6、close_from、open_from、
            lag_cnt、open_ticks、events。返回本拍下发的 J7 指令角(deg)。
    徒手示教(free/assist/follow)与主从示教(leader)共用，保证手感一致。
    """
    gt_mode = ctl.get("grip_mode")
    gh = ctl.get("grip_hold")
    sp6 = ctl.get("sp6")
    if sp6 is None:
        sp6 = q7
    grip_needs_power = bool(gt_mode) or (gh is not None)

    if mode == "free":
        # free 模式电机本来是断电的，夹爪动作时才临时给 J7 上电
        if grip_needs_power and not ctl.get("j7_on"):
            arm.motors[6].enable()
            try:
                arm.motors[6].ensure_mode(Mode.POS_VEL, 500)
            except Exception:
                pass
            ctl["j7_on"] = True
        elif not grip_needs_power and ctl.get("j7_on"):
            arm.motors[6].disable()
            ctl["j7_on"] = False

    if gt_mode == "close":
        if ctl.get("close_from") is None:
            ctl["close_from"] = sp6
        sp6 += TEACH_GRIP_STEP
        lag = sp6 - q7
        ctl["lag_cnt"] = (ctl.get("lag_cnt", 0) + 1) if lag > TEACH_STALL_LAG else 0
        done = (abs(torq7) > TEACH_GRIP_FORCE
                or ctl["lag_cnt"] > TEACH_STALL_TICKS
                or (sp6 - ctl["close_from"]) > TEACH_CLOSE_SPAN)
        if done:
            why = ("力矩" if abs(torq7) > TEACH_GRIP_FORCE
                   else ("夹不动" if ctl["lag_cnt"] > TEACH_STALL_TICKS else "走满行程"))
            ctl.update(grip_mode=None, grip_hold=sp6 + TEACH_GRIP_SQUEEZE,
                       lag_cnt=0, close_from=None)
            ctl["events"].append({"t": round(time.time() - t0, 2), "action": "close",
                                  "deg": round(float(q7), 1),
                                  "hold": round(float(ctl["grip_hold"]), 1),
                                  "torque": round(torq7, 2), "why": why})
    elif gt_mode == "open":
        ctl["open_ticks"] = ctl.get("open_ticks", 0) + 1
        sp6 -= TEACH_GRIP_STEP
        if ctl.get("open_from") is None:
            ctl["open_from"] = sp6
        lag = q7 - sp6
        ctl["lag_cnt"] = (ctl.get("lag_cnt", 0) + 1) if lag > TEACH_STALL_LAG else 0
        armed = ctl["open_ticks"] > (0.4 * TEACH_HZ)
        if (armed and abs(torq7) > TEACH_OPEN_FORCE) \
           or (armed and ctl["lag_cnt"] > TEACH_STALL_TICKS) \
           or (ctl["open_from"] - sp6) > TEACH_OPEN_SPAN:
            ctl.update(grip_mode=None, grip_hold=sp6 + TEACH_GRIP_STEP,
                       open_from=None, lag_cnt=0, open_ticks=0)
            ctl["events"].append({"t": round(time.time() - t0, 2), "action": "open",
                                  "deg": round(float(q7), 1), "torque": round(torq7, 2)})
    elif gh is not None:
        sp6 = gh

    ctl["sp6"] = sp6
    if mode != "free" or ctl.get("j7_on"):
        arm.motors[6].send_pos_vel(float(np.deg2rad(sp6)), 0.4)
    return sp6


def hold_current(arm: "Arm", seconds: float = 0.5, vscale: float = 0.5) -> None:
    """按当前实测角在 POS_VEL 上原地锁住（不改变姿态，只是别让它掉）。"""
    snap = arm.read()
    q = np.deg2rad([d if d is not None else 0.0 for d in snap.deg])
    t_end = time.time() + seconds
    while time.time() < t_end:
        for m, t, v in zip(arm.motors, q, arm.vlims):
            try:
                m.send_pos_vel(float(t), float(v))
            except Exception:
                pass
        for m in arm.motors:
            try:
                m.request_feedback()
            except Exception:
                pass
        try:
            arm.ctrl.poll_feedback_once()
        except Exception:
            pass
        time.sleep(0.02)


def core_urdf_limits_deg():
    lo = np.array([a * 57.29578 for a, _b in URDF_LIMITS], float)
    hi = np.array([b * 57.29578 for _a, b in URDF_LIMITS], float)
    return lo, hi


def goto_pose(arm: "Arm", target_deg, safe_deg=None, stop=None, emit=None,
              label: str = "移动", near_deg: float = 6.0, far_deg: float = 15.0) -> None:
    """安全地移动到目标位姿：如果离得远，先经过「安全转场点」再下去，
    避免像"从零位直接扫到低姿态"那样贴桌面刮过去。"""
    emit = emit or (lambda *a, **k: None)
    tgt = np.deg2rad(np.array(target_deg, float)[:7])
    cur = arm.read()
    q = np.deg2rad([d if d is not None else 0.0 for d in cur.deg])
    if float(np.max(np.abs(q - tgt))) < np.deg2rad(near_deg):
        return
    if safe_deg is not None:
        sp = np.deg2rad(np.array(safe_deg, float)[:7])
        if float(np.max(np.abs(q - sp))) > np.deg2rad(far_deg):
            emit(label + " · 到安全转场点", 0.0, None)
            arm.stream(sp, 5.0, stop, on_tick=_pct_tick(emit, label + " · 到安全转场点", 5.0))
    # 分两段：先把 J3~J6 折到位（缩短力臂，减小肩部重力矩），再动 J1/J2 转肩
    mid = tgt.copy()
    mid[0] = q[0]
    mid[1] = q[1]
    if float(np.max(np.abs(mid - q))) > np.deg2rad(4.0):
        emit(label + " · 先折肘", 0.0, None)
        arm.stream(mid, 5.0, stop, on_tick=_pct_tick(emit, label + " · 先折肘", 5.0))
    emit(label + " · 到位", 0.0, None)
    arm.stream(tgt, 5.0, stop, on_tick=_pct_tick(emit, label + " · 到位", 5.0))


def playback_run(arm: "Arm", data: dict, speed: float = 1.0,
                 stop: threading.Event | None = None, emit=None,
                 on_phase=None) -> dict:
    """回放录制的关节轨迹（pos-vel 逐点跟随）。
    开始前会先把臂平滑移动到「录制第一帧」的位置（即你开始录时的姿态）。"""
    samples = data.get("samples") or []
    if len(samples) < 5:
        raise RuntimeError("录制数据太短")
    arm.enable()                      # 确保 POS_VEL 模式
    tgt0 = np.deg2rad([samples[0][1:][i] for i in range(7)])
    n0 = [0]
    APPROACH_S = 6.0

    def tick_start():
        n0[0] += 1
        if emit:
            emit(min(100.0, n0[0] / (APPROACH_S * RATE_HZ) * 100.0), float(samples[0][7]))

    # 已经很接近录制首帧就不用再"长途移动到起点"（避免从零位扫过去压到桌面）
    cur0 = arm.read()
    q_c0 = np.deg2rad([d if d is not None else 0.0 for d in cur0.deg])
    need_move = float(np.max(np.abs(q_c0 - tgt0))) > np.deg2rad(3.0)
    if need_move:
        if on_phase:
            on_phase("移到录制起点")
        arm.stream(tgt0, APPROACH_S, stop, on_tick=tick_start)
    else:
        # 已在起点：只做一次轻推到位
        for m, t, v in zip(arm.motors, tgt0, arm.vlims):
            m.send_pos_vel(float(t), float(v))
        time.sleep(0.3)
    if on_phase:
        on_phase("回放动作链")
    t_prev = samples[0][0]
    n = len(samples)
    for k, s in enumerate(samples):
        if stop is not None and stop.is_set():
            raise ChainAborted("收到中止指令")
        dt = max(0.0, (s[0] - t_prev)) / max(speed, 0.05)
        t_prev = s[0]
        tgt = np.deg2rad([s[i + 1] for i in range(7)])
        # 每个采样点重复下发，直到臂跟上（或超时），避免指令跑到实际前面 → 累积误差
        deadline = time.time() + max(0.25, dt * 4.0)
        while True:
            for m, t, v in zip(arm.motors, tgt, arm.vlims):
                m.send_pos_vel(float(t), float(v))
            for m in arm.motors:
                m.request_feedback()
            try:
                arm.ctrl.poll_feedback_once()
            except Exception:
                pass
            snap_q = arm.read_fast(spins=1)
            q_now = np.deg2rad([d if d is not None else 0.0 for d in snap_q.deg])
            lag = float(np.max(np.abs(q_now - tgt)))
            if lag < np.deg2rad(2.5) or time.time() > deadline:
                break
            time.sleep(0.02)
        if emit and k % 5 == 0:
            emit(k / n * 100.0, float(s[7]))
        # 回放时的力矩保护：顶住就停，别等 30° 跟踪误差
        snap = arm.read_fast(spins=1)
        tq = max([abs(v) for v in (snap.torq or []) if v is not None] or [0.0])
        if tq > PLAYBACK_TORQUE_ABORT:
            hang = globals().get("_pb_hang", 0) + 1
            globals()["_pb_hang"] = hang
            if hang >= 3:
                globals()["_pb_hang"] = 0
                raise RuntimeError(f"回放中顶住：力矩 {tq:.1f} Nm（阈值 {PLAYBACK_TORQUE_ABORT}），已停止")
        else:
            globals()["_pb_hang"] = 0
        if dt > 0:
            time.sleep(dt)
    return {"ok": True, "samples": n}


# ---------------------------------------------------------------- 任务层（抓取-装配）
PLAYBACK_TORQUE_ABORT = 20.0
MAX_LEAD_DEG = 8.0          # 指令最多领先实测多少度（防止误差累积触发保护）
STREAM_TORQUE_ABORT = 22.0  # Nm，运动中任一关节超过它认为举不动/顶住   # Nm，回放时任一关节力矩超过它就算顶住
TASK_APPROACH_MM = 40.0     # 抓取/放置前的悬停高度
TASK_LIFT_MM = 60.0         # 夹起后抬升高度
TASK_GRIP_FORCE = 1.5       # Nm，夹紧到位判据
TASK_GRIP_SPAN = 340.0      # deg，夹紧兜底行程（实测总行程>300°）
TASK_OPEN_SPAN = 320.0      # deg，张开兜底行程（留 40° 余量，不顶机械端点）


def grip_close_to_force(arm: "Arm", stop=None, torque=TASK_GRIP_FORCE,
                        span=TASK_GRIP_SPAN, emit=None) -> dict:
    """持续闭合直到力矩到阈值 / 夹不动 / 走满行程，然后停在那儿保持夹持力。"""
    m7 = arm.motors[6]
    arm.motors[6].enable()
    try:
        m7.ensure_mode(Mode.POS_VEL, 500)
    except Exception:
        pass
    s = arm.read_fast()
    q6 = s.deg[6] if s.deg[6] is not None else 0.0
    tq = s.torq[6] if s.torq and s.torq[6] is not None else 0.0
    sp = q6
    lag_cnt = 0
    start = q6
    why = "走满行程"
    k = 0
    grace = 20                       # 起手 0.4s 不判力矩（上一拍的夹持力会误判成"到位"）
    for _ in range(int(span / 0.4) + 50):
        if stop is not None and stop.is_set():
            raise ChainAborted("收到中止指令")
        k += 1
        sp += 1.2
        m7.send_pos_vel(float(np.deg2rad(sp)), 1.2)
        m7.request_feedback()
        try:
            arm.ctrl.poll_feedback_once()
        except Exception:
            pass
        time.sleep(0.02)
        st = m7.get_state()
        if st is None:
            continue
        tq = st.torq
        if k > grace and abs(tq) > torque:
            why = "力矩"
            break
        if sp - np.rad2deg(st.pos) > 12.0:
            lag_cnt += 1
            if lag_cnt > 8:
                why = "夹不动"
                break
        else:
            lag_cnt = 0
        if sp - start > span:
            break
    hold = sp + 1.0
    for _ in range(5):
        m7.send_pos_vel(float(np.deg2rad(hold)), 0.4)
        time.sleep(0.02)
    return {"hold": round(float(hold), 1), "why": why, "torque": round(float(tq), 2),
            "span_used": round(float(sp - start), 1)}


def _keep_grip(target, hold_deg):
    """搬运动作里把 J7 固定成夹持角；否则 stream 会用槽位里教的 J7 把爪子张开（会掉件）。"""
    a = np.array(target, float).copy()
    a[6] = float(hold_deg)
    return a


def grip_open(arm: "Arm", stop=None, span=TASK_OPEN_SPAN) -> dict:
    m7 = arm.motors[6]
    arm.motors[6].enable()
    try:
        m7.ensure_mode(Mode.POS_VEL, 500)
    except Exception:
        pass
    s = arm.read_fast()
    sp = s.deg[6] if s.deg[6] is not None else 0.0
    start = sp
    k = 0
    grace = 20                       # 起手 0.4s 不判力矩
    why = "走满行程"
    for _ in range(int(span / 0.4) + 50):
        if stop is not None and stop.is_set():
            raise ChainAborted("收到中止指令")
        k += 1
        sp -= 1.2
        m7.send_pos_vel(float(np.deg2rad(sp)), 1.2)
        m7.request_feedback()
        try:
            arm.ctrl.poll_feedback_once()
        except Exception:
            pass
        time.sleep(0.02)
        if k > grace:
            st = m7.get_state()
            if st is not None and abs(st.torq) > 2.5:
                why = "力矩"
                break
        if start - sp > span:
            break
    hold = sp - 1.0
    for _ in range(5):
        m7.send_pos_vel(float(np.deg2rad(hold)), 0.4)
        time.sleep(0.02)
    return {"hold": round(float(hold), 1), "span_used": round(float(start - sp), 1)}


def run_task(arm: "Arm", p: dict, stop=None, emit=None) -> dict:
    """
    抓取-装配任务。p = {
        "part": "A"/"B",
        "form": "upright"/"side"/"flipped",
        "angle": 绕竖直轴角度(deg),
        "grasp": [7 关节角deg],  "place": [7 关节角deg],
        "dry_run": bool
    }
    序列：张开 → 悬停(抓取位+40) → 沉到抓取位 → 夹到力 → 抬 60
          → 悬停(装配位+40, 已按 angle 旋转) → 沉下 → 张开 → 抬 40
    """
    emit = emit or (lambda *a, **k: None)
    grasp = np.array(p["grasp"], float)
    place = np.array(p["place"], float)
    ang = float(p.get("angle", 0.0))
    log = []

    def push(tag, **kw):
        snap = arm.read()
        rec = {"tag": tag, **snap.as_dict(), **kw}
        log.append(rec)
        emit(tag, None, snap)
        return rec

    # 角度补偿：绕各自 TCP 自转 ang
    g_q, gpe, gre = pose_shift(grasp, 0.0, ang)
    p_q, ppe, pre = pose_shift(place, 0.0, ang)
    # 悬停位：把位姿抬高 approach / lift（只算，不下降）
    g_pre, g2pe, g2re = pose_shift(g_q, TASK_APPROACH_MM, 0.0)
    g_up, g3pe, g3re = pose_shift(g_q, TASK_LIFT_MM, 0.0)
    p_pre, p2pe, p2re = pose_shift(p_q, TASK_APPROACH_MM, 0.0)
    plan = {
        "grasp_deg": [round(float(v), 2) for v in g_q],
        "grasp_hover_deg": [round(float(v), 2) for v in g_pre],
        "grasp_lift_deg": [round(float(v), 2) for v in g_up],
        "place_deg": [round(float(v), 2) for v in p_q],
        "place_hover_deg": [round(float(v), 2) for v in p_pre],
        "ik_resid": {"grasp": [round(gpe * 1000, 2), round(math.degrees(gre), 3)],
                     "place": [round(ppe * 1000, 2), round(math.degrees(pre), 3)]},
        "angle_deg": ang,
    }
    if p.get("dry_run"):
        return {"ok": True, "dry_run": True, "plan": plan}

    arm.enable()          # 关键：任务开始前给关节上电（否则指令不执行，必报跟踪失败）
    time.sleep(0.2)

    # 真的动
    emit("张开夹爪", 0.0, None)
    grip_open(arm, stop)
    push("张开夹爪")

    emit("到抓取悬停位", 0.0, None)
    safe_g = (p.get("slots") or {}).get(f"safe_{part}") or (p.get("slots") or {}).get("safe")
    goto_pose(arm, g_pre, safe_g, stop, emit, "到抓取悬停位")
    push("到抓取悬停位", target=list(plan["grasp_hover_deg"]))

    emit("沉到抓取位", 0.0, None)
    arm.stream(np.deg2rad(g_q), 4.0, stop, on_tick=_pct_tick(emit, "沉到抓取位", 4.0))
    push("沉到抓取位", target=list(plan["grasp_deg"]))

    emit("夹紧", 0.0, None)
    g = grip_close_to_force(arm, stop)
    push("夹紧", grip=g)

    emit("抬起", 0.0, None)
    arm.stream(g_up, 3.0, stop, on_tick=_pct_tick(emit, "抬起", 3.0))
    push("抬起", target=list(plan["grasp_lift_deg"]))

    emit("到装配悬停位", 0.0, None)
    arm.stream(np.deg2rad(p_pre), 6.0, stop, on_tick=_pct_tick(emit, "到装配悬停位", 6.0))
    push("到装配悬停位", target=list(plan["place_hover_deg"]))

    emit("沉到装配位", 0.0, None)
    arm.stream(np.deg2rad(p_q), 4.0, stop, on_tick=_pct_tick(emit, "沉到装配位", 4.0))
    push("沉到装配位", target=list(plan["place_deg"]))

    emit("张开夹爪", 0.0, None)
    o = grip_open(arm, stop)
    push("张开夹爪", grip=o)

    emit("抬起离开", 0.0, None)
    arm.stream(np.deg2rad(p_pre), 3.0, stop, on_tick=_pct_tick(emit, "抬起离开", 3.0))
    push("抬起离开")

    return {"ok": True, "plan": plan, "log": log,
            "part": p.get("part"), "form": p.get("form"), "angle": ang}


def zero_calibrate(arm: "Arm", joints: list[int] | None = None,
                   stop=None, emit=None, dry_run: bool = False) -> dict:
    """
    零校准：把「当前物理姿态」写成各电机的零点（达妙 0xFE 指令，写进电机自身）。
    joints: 要校的关节索引 0..6（默认全部 7 个）
    dry_run: 只读当前角度并报告，不发指令
    注意：校准后所有已存的示教槽位/录制都会失配（因为绝对角变了）。
    """
    emit = emit or (lambda *a, **k: None)
    ids = list(range(7)) if joints is None else [int(j) for j in joints]
    for j in ids:
        if not (0 <= j <= 6):
            raise RuntimeError(f"关节索引越界: {j}")

    arm.disable()
    time.sleep(0.25)
    before = arm.read()
    if any(v is None for v in before.deg):
        raise RuntimeError("读不到全部关节角度（检查供电/CAN）")

    rows = []
    for j in ids:
        rows.append({"joint": NAMES[j], "index": j,
                     "before_deg": round(float(before.deg[j]), 3)})
    if dry_run:
        return {"ok": True, "dry_run": True, "before_deg": before.deg, "rows": rows,
                "note": "dry_run：未发送任何指令"}

    for j in ids:
        if stop is not None and stop.is_set():
            raise ChainAborted("收到中止指令")
        emit(f"零校准 {NAMES[j]}", None, None)
        try:
            arm.motors[j].set_zero_position()
        except Exception as e:
            rows[[r["index"] for r in rows].index(j)]["error"] = str(e)
        time.sleep(0.18)

    time.sleep(0.35)
    after = arm.read()
    for r in rows:
        j = r["index"]
        r["after_deg"] = None if after.deg[j] is None else round(float(after.deg[j]), 3)
        r["ok"] = (r["after_deg"] is not None and abs(r["after_deg"]) < 1.5)
    return {"ok": all(r.get("ok") for r in rows),
            "before_deg": [None if v is None else round(float(v), 2) for v in before.deg],
            "after_deg": [None if v is None else round(float(v), 2) for v in after.deg],
            "rows": rows,
            "note": "零点已写入电机；已存的示教槽位/录制需要重新示教"}


# ---------------------------------------------------------------- 任务层 v2（推正 → 抓 → 装）
PUSH_SPEED = 0.35           # rad/s，推的时候速度上限（慢，便于观察）
PUSH_CONTACT_TORQUE = 8.0   # Nm，推的过程中顶住就停（避免把工件顶飞/顶坏臂）

# 每种形态对应的前置动作序列；最后统一是 "grasp"（抓正件）
# 按工件分别配置：B 件流程还没定，先沿用 A 的，之后改这里即可
PART_PROGRAMS: dict[str, dict[str, list[str]]] = {
    "A": {"upright": ["grasp"], "side": ["push2", "grasp"],
          "flipped": ["push1", "push2", "grasp"]},
    "B": {"upright": ["grasp"], "side": ["push2", "grasp"],
          "flipped": ["push1", "push2", "grasp"]},   # TODO: B 件流程待定
}


def slot_names(part: str, step: str) -> list[str]:
    """推倒/推立用「工件专属」槽位，找不到时回落到通用名。"""
    return [f"{step}_{part}_a", f"{step}_{part}_b", f"{step}_a", f"{step}_b"]


def push_line(arm: "Arm", a_deg, vec_mm, dur: float = 8.0, stop=None,
              emit=None, label: str = "直线推", torque_abort: float = 5.0) -> dict:
    """从位姿 a 出发，保持末端姿态不变，沿 vec_mm（臂基座系，mm）走直线推。
    比关节插值好控制方向：给 (前, 侧, 上) 三个数就是一条直线。"""
    emit = emit or (lambda *a, **k: None)
    a = np.array(a_deg, float)
    vec = np.array(vec_mm, float) / 1000.0
    arm.enable()
    time.sleep(0.15)
    p0, R0 = fk_full(a)
    n = max(int(dur * RATE_HZ), 1)
    aborted = None
    q_prev = a[:6].copy()
    for k in range(1, n + 1):
        if stop is not None and stop.is_set():
            raise ChainAborted("收到中止指令")
        s = s_curve(k / n)
        pt = p0 + vec * s
        q6, perr, rerr = ik_pose(q_prev, pt, R0, iters=120, max_step=0.06, pull=0.02)
        q_prev = q6
        tgt = np.r_[np.deg2rad(q6), np.deg2rad(a[6])]
        for m, c, v in zip(arm.motors, tgt, arm.vlims):
            m.send_pos_vel(float(c), 0.5)
        for m in arm.motors:
            m.request_feedback()
        try:
            arm.ctrl.poll_feedback_once()
        except Exception:
            pass
        time.sleep(1.0 / RATE_HZ)
        snap = arm.read_fast()
        tq = max([abs(v) for v in (snap.torq or []) if v is not None] or [0.0])
        if k > 10 and tq > torque_abort:
            aborted = f"力矩 {tq:.1f} Nm 超阈值，停止推进"
            break
        if k % 10 == 0:
            emit(label, k / n * 100.0, None)
    end = arm.read()
    return {"aborted": aborted, "vec_mm": [round(v, 1) for v in vec_mm],
            "dur": dur,
            "end_deg": [None if v is None else round(v, 1) for v in end.deg],
            "end_tcp_mm": [round(v * 1000, 1) for v in fk_tcp([x if x is not None else 0 for x in end.deg])]}


def run_push(arm: "Arm", p: dict, stop=None, emit=None) -> dict:
    """单独测试一次推件动作（不抓不放），用于现场调推力/行程。
    p = {"part":"A", "which":"push1"|"push2", "a":[...], "b":[...], "dur":8.0}
    """
    emit = emit or (lambda *a, **k: None)
    a = np.array(p["a"], float)
    b = np.array(p["b"], float)
    label = "推倒（倒→侧）" if p.get("which") == "push1" else "推立（侧→正）"
    arm.enable()
    time.sleep(0.2)
    emit(label, 0.0, None)
    res = push_action(arm, a, b, float(p.get("dur", 8.0)), stop, emit, label)
    snap = arm.read()
    emit(label, 100.0, snap)
    ab = (res or {}).get("aborted")
    if ab:
        raise RuntimeError(f"{label} 中断：{ab}")
    return {"ok": True, "label": label, "result": res,
            "end_deg": [None if v is None else round(v, 1) for v in snap.deg]}


def push_action(arm: "Arm", a_deg, b_deg, dur: float = 6.0,
                stop=None, emit=None, label: str = "推") -> dict:
    """沿关节插值把工件推倒/推正；途中力矩超限即停（说明顶住了）。"""
    emit = emit or (lambda *a, **k: None)
    a_d = np.array(a_deg, float)
    b = np.deg2rad(np.array(b_deg, float))          # 终点：度 -> 弧度
    arm.enable()
    time.sleep(0.15)
    # 先平滑移到「起点」，否则会从当前位置直接扫向终点（曾因此撞到东西 9.2Nm 停住）
    cur = arm.read()
    q_now = np.deg2rad([d if d is not None else 0.0 for d in cur.deg])
    if float(np.max(np.abs(np.deg2rad(a_d[:7]) - q_now))) > np.deg2rad(2.0):
        emit(label + " · 移到起点", 0.0, None)
        arm.stream(np.deg2rad(a_d[:7]), 4.0, stop,
                   on_tick=_pct_tick(emit, label + " · 移到起点", 4.0))
    start = arm.read()
    q0 = np.deg2rad([d if d is not None else 0.0 for d in start.deg])
    n = max(int(dur * RATE_HZ), 1)
    aborted = None
    for k in range(1, n + 1):
        if stop is not None and stop.is_set():
            raise ChainAborted("收到中止指令")
        cmd = q0 + (b - q0) * s_curve(k / n)
        for m, c, v in zip(arm.motors, cmd, arm.vlims):
            m.send_pos_vel(float(c), PUSH_SPEED)
        for m in arm.motors:
            m.request_feedback()
        try:
            arm.ctrl.poll_feedback_once()
        except Exception:
            pass
        time.sleep(1.0 / RATE_HZ)
        snap = arm.read_fast()
        tq = max([abs(v) for v in (snap.torq or []) if v is not None] or [0.0])
        if tq > PUSH_CONTACT_TORQUE:
            aborted = f"力矩 {tq:.1f} Nm 超阈值，停止推进"
            break
        if k % 10 == 0:
            emit(label, k / n * 100.0, None)
    return {"aborted": aborted, "dur": dur}


def run_task(arm: "Arm", p: dict, stop=None, emit=None) -> dict:
    """
    任务 v2：按形态推正 → 抓（带角度补偿）→ 归正朝向 → 搬到装配位。
    p = {
      "part": "A"/"B", "form": "upright"/"side"/"flipped", "angle": deg,
      "slots": { 名字: [7 关节角] },
      "dry_run": bool
    }
    需要的槽位：
      push1_a / push1_b   倒 → 侧（推到位的起点与终点）
      push2_a / push2_b   侧 → 正
      grasp_{part}_upright  抓"正"件（TCP 尽量在孔轴线上）
      place_{part}          装配位
    """
    emit = emit or (lambda *a, **k: None)
    S = p.get("slots") or {}
    part = p.get("part", "A")
    form = p.get("form", "upright")
    ang = float(p.get("angle", 0.0))
    seq = (PART_PROGRAMS.get(part) or PART_PROGRAMS["A"]).get(form)
    if seq is None:
        raise RuntimeError(f"未知形态 {form}")
    grasp_name = f"grasp_{part}_upright"
    place_name = f"place_{part}"

    def pushpairs(step: str):
        names = slot_names(part, step)
        for i in range(0, len(names), 2):
            a, b = names[i], names[i + 1]
            if a in S and b in S:
                return a, b
        return None

    missing = []
    if grasp_name not in S:
        missing.append(grasp_name)
    if place_name not in S:
        missing.append(place_name)
    pushes = {}
    for st in seq:
        if st.startswith("push"):
            if f"{st}_{part}" in (p.get("push_recs") or {}):
                continue          # 有录制就不用两点槽位
            pr = pushpairs(st)
            if pr is None:
                missing.append(f"{st}_{part}_a/{st}_{part}_b")
            else:
                pushes[st] = pr
    if missing:
        raise RuntimeError("缺少槽位：" + ", ".join(missing))

    log = []

    def push(tag, **kw):
        snap = arm.read()
        rec = {"tag": tag, **snap.as_dict(), **kw}
        log.append(rec)
        emit(tag, None, snap)
        return rec

    # 计算抓取位（按 angle 绕 TCP 自转）与各悬停位
    g_q, gpe, gre = pose_shift(S[grasp_name], 0.0, ang)
    g_pre, _, _ = pose_shift(g_q, TASK_APPROACH_MM, 0.0)
    g_lift, _, _ = pose_shift(g_q, TASK_LIFT_MM, 0.0)
    # 抓起来后把朝向归到"标定朝向"：再转 -ang
    g_norm, _, _ = pose_shift(g_lift, 0.0, -ang)
    p_q, ppe, pre = pose_shift(S[place_name], 0.0, 0.0)
    p_pre, _, _ = pose_shift(p_q, TASK_APPROACH_MM, 0.0)

    plan = {
        "part": part, "form": form, "angle": ang,
        "sequence": seq,
        "_debug": {"n_slots": len(S),
                   "push_recordings": {k: v.get("file") for k, v in (p.get("push_recs") or {}).items()}},
        "grasp_deg": [round(float(v), 2) for v in g_q],
        "grasp_hover_deg": [round(float(v), 2) for v in g_pre],
        "grasp_lift_normalized_deg": [round(float(v), 2) for v in g_norm],
        "place_deg": [round(float(v), 2) for v in p_q],
        "place_hover_deg": [round(float(v), 2) for v in p_pre],
        "ik_resid": {"grasp": [round(gpe * 1000, 2), round(math.degrees(gre), 3)],
                     "place": [round(ppe * 1000, 2), round(math.degrees(pre), 3)]},
    }
    if p.get("dry_run"):
        return {"ok": True, "dry_run": True, "plan": plan}

    arm.enable()          # 关键：任务开始前给关节上电（否则指令不执行，必报跟踪失败）
    time.sleep(0.2)

    # ---- 1) 推正（优先回放录制，没有录制才用两点插值） ----
    recs = p.get("push_recs") or {}
    for step in seq:
        if not step.startswith("push"):
            continue
        label = "推倒 → 侧" if step == "push1" else "推侧 → 正"
        emit(label, 0.0, None)
        rec = recs.get(f"{step}_{part}")
        if rec:
            emit(label + "（回放录制 " + str(rec.get("file")) + "）", 0.0, None)
            safe = (p.get("slots") or {}).get(f"safe_{part}") or (p.get("slots") or {}).get("safe")
            goto_pose(arm, rec["samples"][0][1:8], safe, stop, emit, label)
            playback_run(arm, {"samples": rec["samples"]}, 1.0, stop,
                         emit=lambda pct, grip: emit(label, pct, None),
                         on_phase=lambda name: emit(label + " · " + name, 0.0, None))
            push(label + "（录制回放）")
        else:
            a, b = pushes[step]
            r = push_action(arm, S[a], S[b], 8.0, stop, emit, label)
            push(label, result=r)

    # ---- 2) 抓 ----
    emit("张开夹爪", 0.0, None)
    grip_open(arm, stop)
    emit("到抓取悬停位", 0.0, None)
    arm.stream(np.deg2rad(g_pre), 4.0, stop, on_tick=_pct_tick(emit, "到抓取悬停位", 4.0))
    push("到抓取悬停位")
    emit("沉到抓取位", 0.0, None)
    arm.stream(np.deg2rad(g_q), 4.0, stop, on_tick=_pct_tick(emit, "沉到抓取位", 4.0))
    push("沉到抓取位")
    emit("夹紧", 0.0, None)
    gres = grip_close_to_force(arm, stop)
    push("夹紧", grip=gres)
    # 夹住之后：后面所有搬运动作都把 J7 锁在夹持角，避免被槽位里的 J7 张开
    hold7 = float(gres.get("hold", 0.0))
    g_norm = _keep_grip(g_norm, hold7)
    p_pre = _keep_grip(p_pre, hold7)
    p_q = _keep_grip(p_q, hold7)
    emit("抬起并归正朝向", 0.0, None)
    arm.stream(np.deg2rad(g_norm), 4.0, stop, on_tick=_pct_tick(emit, "抬起并归正朝向", 4.0))
    push("抬起并归正朝向")

    # ---- 3) 搬 → 放 ----
    emit("到装配悬停位", 0.0, None)
    arm.stream(np.deg2rad(p_pre), 6.0, stop, on_tick=_pct_tick(emit, "到装配悬停位", 6.0))
    push("到装配悬停位")
    emit("沉到装配位", 0.0, None)
    arm.stream(np.deg2rad(p_q), 4.0, stop, on_tick=_pct_tick(emit, "沉到装配位", 4.0))
    push("沉到装配位")
    emit("张开夹爪", 0.0, None)
    ores = grip_open(arm, stop)
    push("张开夹爪", grip=ores)
    emit("抬起离开", 0.0, None)
    arm.stream(np.deg2rad(_keep_grip(p_pre, ores.get("hold", p_pre[6]))), 3.0,
               stop, on_tick=_pct_tick(emit, "抬起离开", 3.0))
    push("抬起离开")

    return {"ok": True, "plan": plan, "log": log, "grip": gres}


def run_chain(arm: Arm, scale: float = 1.0, stay: bool = False,
              stop: threading.Event | None = None, emit=None) -> dict:
    """按 PLAN 执行动作链。emit(phase, pct, snapshot) 向前端推进度。"""
    emit = emit or (lambda *a, **k: None)
    log: list[dict] = []

    def push(tag: str, snap: Snapshot, **extra) -> None:
        rec = {"t": round(snap.t, 2), "tag": tag, **snap.as_dict(), **extra}
        log.append(rec)
        emit(tag, None, snap)

    snap = arm.read()
    if any(d is None for d in snap.deg):
        raise RuntimeError("有电机读不到反馈，先查线束/CAN")
    push("S0 自检", snap)
    bad = [(NAMES[i], f) for i, f in enumerate(snap.fault) if f]
    if bad:
        raise RuntimeError(f"存在故障码: {bad}")

    arm.enable()
    snap = arm.read()
    push("S1 使能保持", snap)
    q_start = np.deg2rad([d for d in snap.deg])
    tcp_start = fk_tcp(snap.deg)

    # 夹爪现场标定（零点变了也能跑）；卡滞则后续夹爪段直接跳过，臂照常动
    emit("S1.5 夹爪标定", 0.0, None)
    grip = None
    grip_down = False
    try:
        grip = gripper_calibrate(arm, stop, emit)
        snap = arm.read()
        push("S1.5 夹爪标定", snap, **{"gripper_deg": grip["deg"]})
    except SegmentStalled as e:
        grip_down = True
        snap = arm.read()
        push("S1.5 夹爪标定失败（卡滞，跳过夹爪段）", snap, stalled=str(e))
        emit("夹爪卡滞，跳过夹爪段", 100.0, None)

    offsets = {n: 0.0 for n in NAMES}
    peak_deg = list(snap.deg)
    peak_tcp = tcp_start
    best_dz = -1e9
    executed: list[dict] = []
    grip_skipped: list[str] = []
    cmd_prev = q_start.copy()

    for st in PLAN:
        if stay and st.get("home"):
            emit("停在目标位（跳过回程）", 100.0, None)
            break
        tag = f"{st['id']} {st['name']}"
        if st.get("grip") and grip_down:
            grip_skipped.append(st["id"])
            emit(f"{tag}（跳过：夹爪卡滞）", 100.0, None)
            continue
        # 每段先算「相对起始位的关节偏移(deg)」
        note = ""
        if st.get("home"):
            offsets = {n: 0.0 for n in NAMES}
        else:
            new, note = grip_and_ik_targets([math.degrees(v) for v in q_start], st, float(scale))
            offsets.update(new)
            if st.get("grip"):
                gkey = {"open": "open", "close": "closed", "release": "release"}[st["grip"]]
                offsets["J7"] = math.degrees(grip[gkey] - q_start[NAMES.index("J7")])
        target = q_start + np.deg2rad([offsets[n] for n in NAMES])
        try:
            if st.get("hold"):
                hold_s = float(st.get("dur") or T_HOLD)
                arm.hold(target, hold_s, stop, on_tick=_pct_tick(emit, tag, hold_s))
            else:
                # 时长必须按「此刻真实位置」算：上一段（含夹爪标定）可能已经挪动了关节
                now = arm.read()
                q_now = np.deg2rad([d if d is not None else 0.0 for d in now.deg])
                dur = _need_duration(q_now, target, st.get("dur"), arm.vlims)
                arm.stream(target, dur, stop, on_tick=_pct_tick(emit, tag, dur),
                           stall_joint=6 if st.get("grip") else None)
        except SegmentStalled as e:
            grip_down = True
            grip_skipped.append(st["id"])
            push(f"{tag}（卡滞，后续夹爪段跳过）", arm.read(), stalled=str(e))
            emit("夹爪卡滞，跳过后续夹爪段", 100.0, None)
            snap_err = arm.read()
            cmd_prev = np.deg2rad([d if d is not None else 0.0 for d in snap_err.deg])
            continue
        cmd_prev = target
        snap = arm.read()
        tcp = fk_tcp([d if d is not None else 0.0 for d in snap.deg])
        push(tag, snap, tcp_mm=[round(v * 1000, 1) for v in tcp],
             offsets_deg={k: round(v, 2) for k, v in offsets.items()})
        executed.append({"id": st["id"], "name": st["name"],
                         "offsets_deg": dict(offsets)})
        # 峰值姿态 = 抬得最高的那一段（排除回程/只动夹爪的段也会自然落选）
        if not st.get("home") and (tcp[2] - tcp_start[2]) >= best_dz:
            best_dz = tcp[2] - tcp_start[2]
            peak_deg = [d if d is not None else 0.0 for d in snap.deg]
            peak_tcp = tcp

    # 位移换算到「臂自身坐标系」：J1(底座偏航) 会转掉世界系的 x/y，
    # 所以把世界位移绕 z 反转 q1，得到真正的"前/侧/上"
    dvec = [peak_tcp[i] - tcp_start[i] for i in range(3)]
    q1 = float(q_start[0])
    c1, s1 = math.cos(q1), math.sin(q1)
    fwd = dvec[0] * c1 + dvec[1] * s1
    lat = -dvec[0] * s1 + dvec[1] * c1
    up = dvec[2]
    return {
        "ok": True,
        "log": log,
        "executed": executed,
        "gripper_deg": (grip["deg"] if grip else None),
        "gripper_status": ("stalled" if grip_down else "ok"),
        "gripper_skipped": grip_skipped,
        "q_start_deg": [round(math.degrees(v), 2) for v in q_start],
        "q_peak_deg": [round(v, 2) for v in peak_deg],
        "tcp_start_mm": [round(v * 1000, 1) for v in tcp_start],
        "tcp_peak_mm": [round(v * 1000, 1) for v in peak_tcp],
        "displacement_mm": {
            "forward": round(fwd * 1000, 1),
            "lateral": round(lat * 1000, 1),
            "up": round(up * 1000, 1),
            "world": [round(v * 1000, 1) for v in dvec],
            "base_yaw_deg": round(math.degrees(q1), 1),
        },
        "stayed": bool(stay),
    }
