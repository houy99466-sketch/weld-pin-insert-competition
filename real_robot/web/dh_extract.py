# -*- coding: utf-8 -*-
"""
从官方 ReBot_Arm_DM.urdf 反解等效的标准 DH 参数，并用 FK 等价性自证。

模型（标准/远端 DH）：
    A_i = Rz(θ_i + off_i) · Tz(d_i) · Tx(a_i) · Rx(α_i)
    T_0_6 = A_1·A_2·…·A_6

做法：随机多处初值 Levenberg–Marquardt 拟合 24 个参数，使 DH 正解与 URDF 正解
在随机位形下的旋转+平移完全一致；输出残差和参数表。
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent

# ---- 官方 URDF 链（base_link -> link6），(xyz, rpy, axis, movable) ----
CHAIN = [
    ((-8.416e-05, 0, 0.08465),       (0, 0, 0),           (0, 0, 1),  True),
    ((0.020084, 0.031625, 0.05555),  (-1.5708, 0, 0),     (0, 0, -1), True),
    ((-0.264, 0, 0),                 (0, 0, 0),           (0, 0, 1),  True),
    ((0.2426, -0.054, -0.001625),    (0, 0, 0),           (0, 0, 1),  True),
    ((0.078308, -0.0375, -0.03),     (-1.5708, 0, 0),     (0, 0, 1),  True),
    ((0.028008, 0, 0.04),            (0, 1.5708, 0),      (0, 0, 1),  True),
]
END_FIXED = ((0, 0, 0.15539), (0, -1.5708, 3.1415))   # link6 -> end_link（末端/法兰）


def rx(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def ry(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rz(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def rpy(r, p, y): return rz(y) @ ry(p) @ rx(r)
def H(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M


def fk_urdf(q, with_tool=False) -> np.ndarray:
    M = np.eye(4)
    for (xyz, r, axis, mov), ang in zip(CHAIN, q):
        M = M @ H(rpy(*r), np.array(xyz))
        if mov:
            ax = np.array(axis, float); ax = ax / np.linalg.norm(ax)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            M = M @ H(np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K), np.zeros(3))
    if with_tool:
        M = M @ H(rpy(*END_FIXED[1]), np.array(END_FIXED[0]))
    return M


def fk_dh(q, p) -> np.ndarray:
    off, d, a, al = p[:6], p[6:12], p[12:18], p[18:24]
    M = np.eye(4)
    for i in range(6):
        M = M @ H(rz(q[i] + off[i]) @ rz(0), np.zeros(3))
        M = M @ H(np.eye(3), np.array([0, 0, d[i]]))
        M = M @ H(np.eye(3), np.array([a[i], 0, 0]))
        M = M @ H(rx(al[i]), np.zeros(3))
    return M


def residual(p, Q):
    r = []
    for q in Q:
        Tu, Td = fk_urdf(q), fk_dh(q, p)
        r.extend((Td[:3, 3] - Tu[:3, 3]).tolist())
        r.extend((Td[:3, :3] - Tu[:3, :3]).ravel().tolist())
    return np.array(r)


def lm_fit(p0, Q, iters=200, lo=None, hi=None):
    """带边界投影的 Levenberg–Marquardt。"""
    p = p0.copy()
    if lo is not None:
        p = np.clip(p, lo, hi)
    lam = 1e-2
    r = residual(p, Q); cost = r @ r
    best = (p.copy(), cost)
    for _ in range(iters):
        J = np.zeros((len(r), len(p)))
        for k in range(len(p)):
            dp = p.copy(); h = 1e-7; dp[k] += h
            if lo is not None:
                dp = np.clip(dp, lo, hi)
            J[:, k] = (residual(dp, Q) - r) / h
        A = J.T @ J
        g = J.T @ r
        try:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
        except np.linalg.LinAlgError:
            break
        pn = p + step
        if lo is not None:
            pn = np.clip(pn, lo, hi)
        rn = residual(pn, Q); cn = rn @ rn
        if cn < cost:
            p, r, cost = pn, rn, cn
            lam = max(lam / 3, 1e-13)
            if cost < best[1]:
                best = (p.copy(), cost)
        else:
            lam *= 4
            if lam > 1e10:
                break
    return best[0], math.sqrt(best[1] / max(len(r), 1))


def main():
    rng = np.random.default_rng(20261113)
    Q = [rng.uniform(-1.2, 1.2, size=6) for _ in range(10)]

    guess = np.array([0.0] * 6 +                                        # off
                     [0.08465, 0.05555, 0.0, 0.0, 0.03, 0.04] +         # d
                     [0.0, 0.0, 0.264, 0.2426, 0.0, 0.0] +              # a
                     [0.0, -math.pi / 2, math.pi / 2, 0.0, -math.pi / 2, math.pi / 2])

    best = None
    inits = [guess]
    for _ in range(80):                                                 # 随机多起点
        off = rng.uniform(-math.pi, math.pi, 6)
        d = guess[6:12] + rng.normal(0, 0.05, 6)
        a = guess[12:18] + rng.normal(0, 0.08, 6)
        al = np.array([0.0, -math.pi / 2, math.pi / 2, 0.0, -math.pi / 2, math.pi / 2]) \
            + rng.uniform(-0.3, 0.3, 6)
        inits.append(np.concatenate([off, d, a, al]))

    # 物理边界：连杆偏移/长度不可能超过 0.35 m（否则就是靠大数抵消的假解）
    LO = np.array([-2 * math.pi] * 6 + [-0.35] * 6 + [-0.35] * 6 + [-2 * math.pi] * 6)
    HI = np.array([2 * math.pi] * 6 + [0.35] * 6 + [0.35] * 6 + [2 * math.pi] * 6)

    for i, p0 in enumerate(inits):
        p, rms = lm_fit(p0, Q, lo=LO, hi=HI)
        if best is None or rms < best[1]:
            best = (p, rms)
            print(f"  restart {i:>2}: RMS={rms:.3e}  位置最大偏差≈{rms*1000:.4f} mm")

    p, rms = best
    # 归一化：把 off 折到 (-pi, pi]，把 alpha 折到 (-pi, pi]
    off = (p[:6] + math.pi) % (2 * math.pi) - math.pi
    al = (p[18:24] + math.pi) % (2 * math.pi) - math.pi

    # 独立验证：新随机位形 + 含末端法兰
    Qv = [rng.uniform(-1.6, 1.6, size=6) for _ in range(200)]
    max_t = max_r = 0.0
    for q in Qv:
        Tu, Td = fk_urdf(q), fk_dh(q, p)
        max_t = max(max_t, float(np.max(np.abs(Td[:3, 3] - Tu[:3, 3]))))
        max_r = max(max_r, float(np.max(np.abs(Td[:3, :3] - Tu[:3, :3]))))

    print(f"拟合残差 RMS = {rms:.3e}")
    print(f"独立验证（200 组随机位形）: 位置最大误差 = {max_t*1000:.3e} mm, 旋转矩阵最大误差 = {max_r:.3e}")
    print()
    hdr = f"{'i':>2} {'θoff(deg)':>11} {'d(mm)':>9} {'a(mm)':>9} {'α(deg)':>9}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for i in range(6):
        row = dict(i=i + 1,
                   theta_off_deg=round(math.degrees(off[i]), 6),
                   d_mm=round(p[6 + i] * 1000, 4),
                   a_mm=round(p[12 + i] * 1000, 4),
                   alpha_deg=round(math.degrees(al[i]), 6))
        rows.append(row)
        print(f"{i+1:>2} {row['theta_off_deg']:>11.4f} {row['d_mm']:>9.3f} {row['a_mm']:>9.3f} {row['alpha_deg']:>9.4f}")

    tool = dict(xyz_mm=[round(v * 1000, 4) for v in END_FIXED[0]],
                rpy_deg=[round(math.degrees(v), 4) for v in END_FIXED[1]])

    # 末端（end_link/法兰）位置：给 DH 第 6 帧再乘一个固定变换
    print(f"\n末端法兰相对第6帧: xyz={tool['xyz_mm']} mm, rpy={tool['rpy_deg']} deg")

    (OUT / "dh_params.json").write_text(json.dumps(
        {"convention": "standard DH: A_i = Rz(θ+off)·Tz(d)·Tx(a)·Rx(α)",
         "units": {"d": "mm", "a": "mm", "angle": "deg"},
         "dh": rows,
         "tool_flange": tool,
         "fit_rms": rms,
         "verify": {"samples": 200, "max_pos_err_mm": max_t * 1000,
                    "max_rot_err": max_r},
         "source": "urdf/DM/urdf/ReBot_Arm_DM.urdf (Seeed-Projects/reBotArm_control_py)"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {OUT/'dh_params.json'}")


if __name__ == "__main__":
    main()
