# -*- coding: utf-8 -*-
"""
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
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
import traceback
import numpy as np
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chain_core as core  # noqa: E402
import leader_scs as leadmod  # noqa: E402  主臂(另一台臂)读取层

WEB_DIR = Path(__file__).resolve().parent
HOME_FILE = WEB_DIR / "home_pose.json"
import cam as cammod

INDEX_FILE = WEB_DIR / "index.html"
REC_DIR = WEB_DIR / "recordings"
SLOT_FILE = WEB_DIR / "slots.json"
REC_DIR.mkdir(exist_ok=True)
_VER_CACHE: dict = {"mtime": None, "ver": "", "html": ""}


def page_version() -> tuple[str, str]:
    """返回 (版本号, 已注入版本号的 html)。文件变了版本号就变，前端据此自动重载。"""
    import hashlib
    mtime = INDEX_FILE.stat().st_mtime
    if _VER_CACHE["mtime"] != mtime:
        raw = INDEX_FILE.read_text(encoding="utf-8")
        ver = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        _VER_CACHE.update(mtime=mtime, ver=ver,
                          html=raw.replace("__PAGE_VER__", ver))
    return _VER_CACHE["ver"], _VER_CACHE["html"]


class Rig:
    def __init__(self, port: str):
        self.port = port
        self.lock = threading.RLock()
        self.arm: core.Arm | None = None
        self.stop = threading.Event()
        self.home_deg: list | None = None
        if HOME_FILE.exists():
            try:
                self.home_deg = json.loads(HOME_FILE.read_text(encoding="utf-8")).get("deg")
            except Exception:
                self.home_deg = None
        self.job: dict = {
            "running": False, "kind": None, "phase": None, "pct": None,
            "log": [], "result": None, "error": None,
            "started": None, "finished": None,
        }
        self.last_state: dict | None = None
        self.last_state_t: float = 0.0
        self.teach: dict = {
            "running": False, "t0": 0.0, "samples": [], "events": [], "notes": [],
            "grip_target": None, "grip_hold": None, "gripper": None,
            "thread": None, "stop": threading.Event(), "file": None,
            "last_q": None, "err": None,
        }
        # ---- 主臂（另一台 6 轴，Feetech STS3215 总线舵机） ----
        self.leader = None                 # leadmod.LeaderBus
        self.leader_port = "COM22"
        self.leader_baud = 1_000_000
        self.leader_err: str | None = None
        self.leader_map: dict = leadmod.load_map()
        self.leader_mapper: leadmod.LeaderMapper | None = None
        self.leader_hz = 0.0

    # ---------- 夹爪（独立，随时可用） ----------
    def grip(self, action: str) -> dict:
        """不需要示教也能用：只驱动 J7，其他关节不动。
        示教进行中则交给示教循环（避免两条指令抢同一个电机）。"""
        if self.teach["running"]:
            return self.teach_grip(action)
        if self.job["running"]:
            raise RuntimeError("任务运行中：夹爪由任务控制，先中止任务")
        if action not in ("close", "open"):
            raise RuntimeError("action 只能是 close / open")
        self.open()
        arm = self.require()
        if action == "close":
            res = core.grip_close_to_force(arm, None)
        else:
            res = core.grip_open(arm, None)
        return {"ok": True, "action": action, **res}

    # ---------- 单独测试推件 ----------
    def run_push(self, part: str, which: str, dur: float) -> dict:
        self._teach_guard()
        if self.job["running"]:
            raise RuntimeError("有任务在跑，先中止")
        d = self._load_slots()
        flat = {k: v["deg"] for k, v in d.items() if isinstance(v, dict) and "deg" in v}
        names = core.slot_names(part, which)
        pair = None
        for i in range(0, len(names), 2):
            if names[i] in flat and names[i + 1] in flat:
                pair = (names[i], names[i + 1]); break
        if pair is None:
            raise RuntimeError(f"缺少槽位 {which}_{part}_a / _b（先示教起点和终点）")
        p = {"part": part, "which": which, "a": flat[pair[0]], "b": flat[pair[1]], "dur": dur}
        self.start_job("push", p)
        return {"ok": True, "slots": list(pair)}
    # ---------- 单关节点动（调试用） ----------
    def jog(self, joint: int, delta_deg: float) -> dict:
        self._teach_guard()
        if self.job["running"]:
            raise RuntimeError("有任务在跑，先中止")
        self.open()
        arm = self.require()
        cur = arm.read()
        q = np.deg2rad([d if d is not None else 0.0 for d in cur.deg])
        q[joint] += np.deg2rad(delta_deg)
        arm.enable()
        arm.stream(q, 4.0, None, on_tick=None)
        s2 = arm.read()
        return {"ok": True, "joint": joint, "delta_deg": delta_deg,
                "deg": [None if v is None else round(v, 1) for v in s2.deg],
                "torq": s2.torq}

    # ---------- 零校准 ----------
    def zero_calibrate(self, joints, dry_run: bool) -> dict:
        self._teach_guard()
        if self.job["running"]:
            raise RuntimeError("有任务在跑，先中止")
        self.open()
        return core.zero_calibrate(self.require(), joints, None, None, dry_run)

    # ---------- 位姿槽位 / 任务 ----------
    def _load_slots(self) -> dict:
        try:
            return json.loads(SLOT_FILE.read_text(encoding="utf-8-sig")) if SLOT_FILE.exists() else {}
        except Exception:
            return {}

    def _save_slots(self, d: dict) -> None:
        SLOT_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    def slots(self) -> dict:
        return self._load_slots()

    def save_slot(self, name: str) -> dict:
        name = str(name or "").strip()
        if not name:
            raise RuntimeError("槽位名不能为空")
        st = self.state()
        if any(v is None for v in st["deg"]):
            raise RuntimeError("读不到全部关节反馈（机械臂没上电 / CAN 没通），先点『读状态』确认")
        deg = [float(v) for v in st["deg"]]
        d = self._load_slots()
        d[name] = {"deg": [round(v, 2) for v in deg],
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        self._save_slots(d)
        return {"ok": True, "name": name, "deg": d[name]["deg"]}

    def delete_slot(self, name: str) -> dict:
        d = self._load_slots()
        if name not in d:
            raise RuntimeError(f"没有槽位 {name}")
        d.pop(name)
        self._save_slots(d)
        return {"ok": True, "deleted": name}

    def _push_recording(self, name: str):
        """找一个名为 push2_A*.json 的录制，用于"整条轨迹回放"式推件。"""
        cands = sorted(REC_DIR.glob(name + "*.json"),
                       key=lambda f: f.stat().st_mtime, reverse=True)   # 优先用最新的
        for p in cands:
            try:
                d = json.loads(p.read_text(encoding="utf-8-sig"))
                sm = d.get("samples") or []
                if len(sm) >= 5:
                    return {"file": p.name, "samples": sm}
            except Exception:
                continue
        return None

    def task_plan(self, part: str, form: str, angle: float, dry_run: bool) -> dict:
        d = self._load_slots()
        flat = {k: v["deg"] for k, v in d.items() if isinstance(v, dict) and "deg" in v}
        recs = {}
        for w in ("push1", "push2"):
            r = self._push_recording(f"{w}_{part}")
            if r:
                recs[f"{w}_{part}"] = r
        return {"slots": flat, "part": part, "form": form,
                "angle": float(angle), "dry_run": bool(dry_run),
                "push_recs": recs}

    def run_task(self, part: str, form: str, angle: float, dry_run: bool) -> dict:
        self._teach_guard()
        p = self.task_plan(part, form, angle, dry_run)
        if dry_run:
            self.open()
            return {"ok": True, **core.run_task(self.require(), p, None, None)}
        self.start_job("task", p)
        return {"ok": True, "started": True}

    # ---------- 主臂（另一台 6 轴臂，Feetech STS3215 总线舵机） ----------
    def leader_open(self, port: str = "", baud: int = 0) -> dict:
        if self.teach["running"]:
            raise RuntimeError("示教进行中：先『停止并保存』")
        self.leader_close()
        p = (str(port) or self.leader_port).strip()
        b = int(baud or self.leader_baud)
        try:
            bus = leadmod.LeaderBus(p, b)
            ids = bus.scan()
            if not ids:
                bus.close()
                raise RuntimeError(f"{p} @{b} 上没有扫到 STS 舵机（查波特率/接线/供电）")
            info = bus.info()
            self.leader = bus
            self.leader_port, self.leader_baud = p, b
            self.leader_err = None
            self.leader_mapper = leadmod.LeaderMapper(self.leader_map)
        except Exception as e:
            self.leader_err = str(e)
            raise
        return {"ok": True, "port": p, "baud": b, "ids": ids, "motors": info}

    def leader_close(self) -> dict:
        if self.leader is not None:
            try:
                self.leader.close()
            except Exception:
                pass
        self.leader = None
        self.leader_mapper = None
        return {"ok": True, "open": False}

    def leader_state(self) -> dict:
        bus = self.leader
        if bus is None:
            return {"ok": False, "open": False, "port": self.leader_port,
                    "err": self.leader_err, "map": self.leader_map}
        # 示教进行中轮询：直接读示教线程缓存，别跟它抢串口
        if self.teach["running"] and (self.teach.get("ctl") or {}).get("mode") == "leader":
            ctl = self.teach["ctl"]
            return {"ok": True, "open": True, "port": bus.port, "baud": bus.baud,
                    "hz": self.leader_hz, "busy": True,
                    "deg": ctl.get("lead_deg") or [],
                    "target": ctl.get("sp_deg") or [],
                    "map": self.leader_map}
        t0 = time.time()
        n = 0
        lead = None
        while time.time() - t0 < 0.06:
            lead = self.leader_mapper.lead_deg(bus.read_counts())
            n += 1
        hz = n / max(1e-6, time.time() - t0)
        self.leader_hz = round(hz, 1)
        tgt = self.leader_mapper.to_slave(lead or [])
        return {
            "ok": True, "open": True, "port": bus.port, "baud": bus.baud,
            "hz": self.leader_hz,
            "deg": [None if v is None else round(float(v), 2) for v in (lead or [])],
            "target": [None if v is None else round(float(v), 2) for v in tgt],
            "map": self.leader_map,
        }

    def leader_setmap(self, patch: dict) -> dict:
        """改映射：signs/gains/use_abs/joints[i].lmin..lmax，写回 leader_map.json。"""
        m = self.leader_map
        if "use_abs" in patch:
            m["use_abs"] = bool(patch["use_abs"])
        if "joints" in patch:
            for i, j in enumerate(patch["joints"]):
                if i >= len(m["joints"]):
                    break
                for k in ("sign", "gain", "lmin", "lmax", "smin", "smax"):
                    if k in j:
                        m["joints"][i][k] = j[k]
        leadmod.save_map(m)
        if self.leader_mapper is not None:
            self.leader_mapper.reload()
        return {"ok": True, "map": m}

    # ---------- 示教 / 回放 ----------
    def _teach_guard(self) -> None:
        if self.teach["running"]:
            raise RuntimeError("示教进行中：先点『停止并保存』")

    def start_teach(self, mode: str = "free", keep_enabled: bool | None = None) -> dict:
        old = self.teach.get("thread")
        if old is not None and old.is_alive():
            raise RuntimeError("上一轮示教线程还没退出，等 2 秒再试")
        self.teach["running"] = False
        self._teach_guard()
        if self.job["running"]:
            raise RuntimeError("有任务在跑，先中止")
        if mode not in core.TEACH_MODES:
            raise RuntimeError(f"mode 只能是 {list(core.TEACH_MODES)}")
        self.open()
        arm = self.require()
        # leader 模式：先把主臂拉起来
        if mode == "leader":
            if self.leader is None:
                self.leader_open()
            if self.leader_mapper is None:
                self.leader_mapper = leadmod.LeaderMapper(self.leader_map)
        # 每次示教用「新的」stop 事件：旧会话线程持有旧事件，clear 会把它们复活
        self.teach["stop"] = threading.Event()
        self.teach.update(running=True, t0=time.time(), samples=[], events=[],
                          notes=[], grip_target=None, grip_hold=None, mode=mode,
                          gripper=None, file=None, err=None, last_q=None,
                          t=0.0, vmax=0.0, saved=False)
        ctl = {
            "stop": self.teach["stop"], "mode": mode,
            "grip_mode": None, "grip_hold": None, "sp6": None, "open_from": None,
            "follow": False, "j7_on": False,
            "samples": [], "events": [], "notes": [],
            # 停止后是否保持使能原地锁住：主从示教默认保持（free 模式默认卸力）
            "keep_enabled": (True if keep_enabled is None else bool(keep_enabled))
            if mode != "free" else (bool(keep_enabled) if keep_enabled else False),
            # 录制与"跟随"解耦：主从模式先只跟随，点『开始录制』才攒样本
            "record": (mode == "free"),
            "rec_t0": time.time() if mode == "free" else None,
        }
        if mode == "leader":
            jl = self.leader_map.get("joints") or []
            ctl.update(
                gains=[float(j.get("gain", 1.0)) for j in jl][:6] or None,
                signs=[int(j.get("sign", 1)) for j in jl][:6] or None,
                use_abs=bool(self.leader_map.get("use_abs", False)),
                max_travel=float(self.leader_map.get("max_travel_deg",
                                                    core.LEADER_MAX_TRAVEL_DEG)),
                vlim_scale=float(self.leader_map.get("vlim_scale",
                                                     core.LEADER_VLIM_SCALE)),
                j7_hold=bool(self.leader_map.get("j7_hold", False)),
                # 引用同一份 map：/api/leader/map 改了它，循环下一拍就生效
                map=self.leader_map,
            )
        self.teach["ctl"] = ctl
        th = threading.Thread(target=self._teach_thread, args=(arm, ctl), daemon=True)
        self.teach["thread"] = th
        th.start()
        return {"ok": True, "mode": mode,
                "deadband_deg": core.TEACH_DEADBAND_DEG, "hz": core.TEACH_HZ,
                "grip_force_nm": core.TEACH_GRIP_FORCE}

    def _teach_thread(self, arm, ctl) -> None:
        t = self.teach
        try:
            def emit(elapsed, grip_deg, tau, vmax):
                t["t"] = round(elapsed, 1)
                t["last_grip"] = round(grip_deg, 1)
                t["vmax"] = round(vmax, 2)
                t["tau_max"] = round(float(max(abs(x) for x in tau)), 2)
            if ctl.get("mode") == "leader":
                res = core.leader_follow_loop(arm, ctl, self.leader,
                                              self.leader_mapper, emit)
            else:
                res = core.teach_loop(arm, ctl, emit)
            t["result"] = {"reason": res["reason"], "samples": len(res["samples"]),
                           "events": res["events"], "notes": res["notes"]}
        except Exception as e:
            t["err"] = str(e)
            t["result"] = {"reason": f"异常: {e}", "samples": len(ctl["samples"])}
        finally:
            t["running"] = False
            t["samples"] = ctl["samples"] or t.get("samples") or []
            t["events"] = ctl["events"] or t.get("events") or []
            # 主从示教：默认保持使能、原地锁住（不要突然卸力掉下来）
            if ctl.get("mode") == "leader" and self.arm is not None:
                try:
                    if ctl.get("keep_enabled", True):
                        core.hold_current(self.arm, 0.6)
                    else:
                        self.arm.disable()
                except Exception:
                    pass

    def stop_teach(self) -> dict:
        t = self.teach
        if not t["running"]:
            raise RuntimeError("当前没有在示教")
        t["stop"].set()
        th = t["thread"]
        if th:
            th.join(timeout=6.0)
        t["running"] = False
        # free 模式下夹爪上过电的，收尾要关掉
        ctl = t.get("ctl") or {}
        if ctl.get("j7_on") and self.arm is not None:
            try:
                self.arm.motors[6].disable()
            except Exception:
                pass
            ctl["j7_on"] = False
        if (ctl.get("mode") == "free") and self.arm is not None:
            try:
                self.arm.disable()
            except Exception:
                pass
        # 主从示教：保持使能→原地锁住一会儿，确认没有下沉再返回
        if (ctl.get("mode") == "leader") and self.arm is not None:
            try:
                if ctl.get("keep_enabled", True):
                    core.hold_current(self.arm, 0.6)
                else:
                    self.arm.disable()
            except Exception:
                pass
        samples = t["samples"] or []
        out = {"ok": True, "samples": len(samples),
               "duration": round(samples[-1][0], 2) if samples else 0.0,
               "events": t["events"]}
        if samples and not t.get("saved"):
            name = self._save_teach_file(samples)
            t["file"] = name
            out["file"] = name
        elif samples:
            out["file"] = t.get("file")
        return out

    def _save_teach_file(self, samples: list) -> str:
        """把样本写成 recordings/teach_*.json，返回文件名。"""
        t = self.teach
        name = time.strftime("teach_%Y%m%d_%H%M%S.json")
        (REC_DIR / name).write_text(json.dumps({
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hz": core.TEACH_HZ, "mode": t.get("mode"),
            "deadband_deg": core.TEACH_DEADBAND_DEG,
            "gripper": t.get("gripper"), "events": t["events"],
            "notes": (t.get("result") or {}).get("notes", []),
            "joints": list(core.NAMES), "samples": samples,
        }, ensure_ascii=False), encoding="utf-8")
        return name

    def teach_record(self, on: bool) -> dict:
        """主从跟随时开/关录制：开=从这一刻起攒样本；关=立刻落盘，跟随继续不动。"""
        t = self.teach
        if not t["running"]:
            raise RuntimeError("先开始主从跟随/示教")
        ctl = t.get("ctl") or {}
        if on:
            ctl["samples"] = []
            t["samples"] = []
            ctl["rec_t0"] = None
            ctl["record"] = True
            return {"ok": True, "recording": True, "samples": 0}
        ctl["record"] = False
        samples = ctl.get("samples") or []
        out = {"ok": True, "recording": False, "samples": len(samples),
               "duration": round(samples[-1][0], 2) if samples else 0.0,
               "events": ctl.get("events") or []}
        if samples:
            name = self._save_teach_file(samples)
            t["file"] = name
            t["samples"] = list(samples)
            t["saved"] = True
            ctl["samples"] = []          # 已落盘，停止跟随时别再存一遍
            out["file"] = name
        return out

    def teach_grip(self, action: str) -> dict:
        t = self.teach
        if not t["running"]:
            raise RuntimeError("先开始示教")
        ctl = t["ctl"]
        if action == "close":
            ctl["grip_mode"] = "close"
        elif action == "open":
            ctl["grip_mode"] = "open"
            ctl["open_from"] = None
        else:
            raise RuntimeError("action 只能是 close / open")
        ctl["grip_hold"] = None
        return {"ok": True, "action": action,
                "note": ("向闭合方向爬，力矩到 1.5Nm 自动停" if action == "close"
                         else "向张开方向爬，遇阻或走满 110° 停")}

    def teach_follow(self, on: bool) -> dict:
        t = self.teach
        if not t["running"]:
            raise RuntimeError("先开始示教")
        t["ctl"]["follow"] = bool(on)
        return {"ok": True, "follow": bool(on)}

    def teach_state(self) -> dict:
        t = self.teach
        ctl = t.get("ctl") or {}
        n = len(ctl.get("samples") or [])
        return {"running": t["running"], "t": t.get("t"), "samples": n,
                "mode": t.get("mode"),
                "recording": bool((t.get("ctl") or {}).get("record")),
                "following": bool(t["running"] and t.get("mode") == "leader"),
                "follow": bool(ctl.get("follow")),
                "grip": t.get("last_grip"), "vmax": t.get("vmax"),
                "tau_max": t.get("tau_max"), "gripper": t.get("gripper"),
                "events": (ctl.get("events") or []), "file": t.get("file"), "err": t.get("err"),
                "result": t.get("result")}

    def _rec_path(self, name: str) -> Path:
        """只允许 recordings 目录下的 .json，防止路径穿越。"""
        raw = str(name or "").strip()
        if not raw:
            raise RuntimeError("缺少文件名")
        p = (REC_DIR / Path(raw).name).resolve()
        if p.parent != REC_DIR.resolve() or p.suffix.lower() != ".json":
            raise RuntimeError("文件名不合法（必须在本目录下的 .json）")
        return p

    def rec_delete(self, name: str) -> dict:
        p = self._rec_path(name)
        if not p.exists():
            raise RuntimeError(f"找不到 {name}")
        p.unlink()
        return {"ok": True, "deleted": p.name}

    def rec_rename(self, old: str, new: str) -> dict:
        src = self._rec_path(old)
        if not src.exists():
            raise RuntimeError(f"找不到 {old}")
        target = self._rec_path(new if new.lower().endswith(".json") else new + ".json")
        if target.exists() and target != src:
            raise RuntimeError(f"{target.name} 已存在")
        src.rename(target)
        return {"ok": True, "from": src.name, "to": target.name}

    def rec_note(self, name: str, note: str) -> dict:
        p = self._rec_path(name)
        if not p.exists():
            raise RuntimeError(f"找不到 {name}")
        d = json.loads(p.read_text(encoding="utf-8-sig"))
        d["note"] = str(note)
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        return {"ok": True, "name": p.name, "note": str(note)}

    def recordings(self) -> list[dict]:
        """扫描 recordings 目录下所有 .json（不限定文件名前缀，改名/手动拷入都能识别）。"""
        out = []
        files = sorted(REC_DIR.glob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files:
            item = {"name": p.name, "mtime": time.strftime("%Y-%m-%d %H:%M",
                                                           time.localtime(p.stat().st_mtime)),
                    "size_kb": round(p.stat().st_size / 1024, 1)}
            try:
                d = json.loads(p.read_text(encoding="utf-8-sig"))
                s = d.get("samples") or []
                if not isinstance(s, list) or not s:
                    raise ValueError("没有 samples")
                first_ok = isinstance(s[0], list) and len(s[0]) >= 8
                item.update({
                    "created": d.get("created"),
                    "note": d.get("note") or "",
                    "samples": len(s),
                    "duration": round(float(s[-1][0]), 1) if first_ok else 0.0,
                    "events": len(d.get("events") or []),
                    "ok": True,
                })
                if not first_ok:
                    item["warn"] = "samples 格式不认识，可能不能回放"
            except Exception as e:
                item.update({"created": None, "samples": 0, "duration": 0.0,
                             "events": 0, "ok": False, "warn": f"无法解析: {e}"})
            out.append(item)
        return out[:80]

    def play(self, name: str, speed: float) -> dict:
        RIG._teach_guard()
        path = REC_DIR / Path(name).name
        if not path.exists():
            raise RuntimeError(f"找不到录制文件 {name}")
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        self.start_job("play", {"data": data, "speed": speed})
        return {"ok": True}

    # ---------- 串口 ----------
    def _busy_hint(self) -> str:
        return ("串口被别的程序占用。先关掉 MotorBridge Studio / ws_gateway / DM_Tools，"
                "或点『释放串口』再关掉它们。查占用："
                "taskkill /IM ws_gateway.exe /F")

    def ports(self) -> list[dict]:
        """系统当前存在的串口。优先 pyserial；没装就退回 WMI（PowerShell 查 PnP）。"""
        out = []
        try:
            from serial.tools import list_ports
            for p in list_ports.comports():
                out.append({
                    "device": p.device,
                    "desc": p.description or "",
                    "hwid": p.hwid or "",
                    "likely_dm": ("2E88" in (p.hwid or "").upper()
                                  or "USB 串行设备" in (p.description or "")
                                  or "USB Serial" in (p.description or "")),
                })
            if out:
                return out
        except Exception:
            pass
        # 退路：只要当前配置的口还在，就至少把它列出来
        try:
            import os
            if os.name == "nt":
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Get-PnpDevice -PresentOnly -Class Ports | "
                     "Select-Object -ExpandProperty FriendlyName"],
                    capture_output=True, text=True, timeout=6)
                for line in (r.stdout or "").splitlines():
                    line = line.strip()
                    m = re.search(r"\((COM\d+)\)", line)
                    if m:
                        out.append({"device": m.group(1), "desc": line, "hwid": "",
                                    "likely_dm": ("USB 串行" in line or "USB Serial" in line)})
        except Exception:
            pass
        if not out and self.port:
            out.append({"device": self.port, "desc": "（当前配置口，未做枚举）",
                        "hwid": "", "likely_dm": True})
        return out

    def set_port(self, com: str) -> dict:
        with self.lock:
            if self.arm is not None:
                try:
                    self.arm.disable()
                except Exception:
                    pass
                self.arm.close()
                self.arm = None
            self.port = com
            self.open()
            return {"ok": True, "port": com}

    def open(self) -> dict:
        with self.lock:
            if self.arm is not None:
                return {"ok": True, "already": True}
            try:
                self.arm = core.Arm(self.port)
            except Exception as e:
                self.arm = None
                msg = str(e)
                low = msg.lower()
                if "拒绝访问" in msg or "access is denied" in low:
                    raise RuntimeError(f"{msg}  →  {self._busy_hint()}")
                if "找不到" in msg or "cannot find" in low or "no such file" in low:
                    have = ", ".join(p["device"] for p in self.ports()) or "（一个都没有）"
                    raise RuntimeError(
                        f"找不到串口 {self.port}：USB 线没插 / 驱动没起来 / 端口号变了。"
                        f"当前可用串口：{have}。插好后点『读状态』重试，或在下拉框里换端口。")
                raise
            return {"ok": True}

    def release(self) -> dict:
        with self.lock:
            if self.arm is not None:
                try:
                    self.arm.disable()
                except Exception:
                    pass
                self.arm.close()
                self.arm = None
            return {"ok": True}

    def require(self) -> core.Arm:
        if self.arm is None:
            self.open()
        return self.arm  # type: ignore[return-value]

    # ---------- 状态 ----------
    def state(self) -> dict:
        if self.teach["running"]:
            # 示教中：真实读一次（free 模式下电机是失能的，status 会如实报 0）
            try:
                snap = self.require().read(spins=1)
                self.last_state = snap.as_dict()
                return self.last_state
            except Exception:
                q = self.teach.get("ctl", {}).get("samples")
                if q:
                    return {"t": q[-1][0], "names": list(core.NAMES),
                            "deg": list(q[-1][1:]), "vel": [], "torq": [],
                            "status": [-1] * 7, "fault": [-1] * 7, "teaching": True}
        if self.job["running"]:
            # 任务进行中也要真实读一次，否则前端显示的是旧缓存（会让人以为臂没动）
            try:
                snap = self.require().read_fast(spins=1)
                return snap.as_dict()
            except Exception:
                if self.last_state:
                    return self.last_state

        # 串口掉线自愈：报错时尝试重连一次（15 秒内只试一次）
        try:
            snap = self.require().read_fast(spins=1)
            self.last_state = snap.as_dict()
            return self.last_state
        except Exception as e:
            msg = str(e)
            bad = ("121" in msg or "超时" in msg or "write failed" in msg
                   or "拒绝访问" in msg or "request_feedback" in msg)
            now = time.time()
            if bad and (now - getattr(self, "_last_reconnect", 0.0)) > 15.0:
                self._last_reconnect = now
                try:
                    self.reconnect()
                    snap = self.require().read_fast(spins=1)
                    self.last_state = snap.as_dict()
                    return self.last_state
                except Exception as e2:
                    raise RuntimeError(f"串口异常且重连失败：{msg} / {e2}")
            raise
        with self.lock:
            arm = self.require()
            snap = arm.read()
            self.last_state = snap.as_dict()
            self.last_state_t = time.time()
            return self.last_state

    def enable(self) -> dict:
        with self.lock:
            arm = self.require()
            arm.enable()
            snap = arm.read()
            self.last_state = snap.as_dict()
            return self.last_state

    def disable(self) -> dict:
        with self.lock:
            arm = self.require()
            arm.disable()
            time.sleep(0.1)
            snap = arm.read()
            self.last_state = snap.as_dict()
            return self.last_state

    # ---------- 起始位 ----------
    def record_home(self) -> dict:
        st = self.state()
        deg = [None if v is None else float(v) for v in st["deg"]]
        if any(v is None for v in deg[:6]):
            raise RuntimeError("有电机无反馈，无法记录起始位")
        self.home_deg = deg
        HOME_FILE.write_text(json.dumps({"deg": deg, "ts": time.time()},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
        return {"ok": True, "home_deg": deg}

    def go_home(self, duration: float = 4.0) -> dict:
        if not self.home_deg:
            raise RuntimeError("还没记录起始位：先点『记录起始位』")
        self.start_job("home", {"duration": duration})
        return {"ok": True}

    # ---------- 动作链 ----------
    def start(self, scale: float, stay: bool) -> dict:
        self.start_job("chain", {"scale": scale, "stay": stay})
        return {"ok": True}

    def start_job(self, kind: str, params: dict) -> None:
        with self.lock:
            if self.job["running"]:
                raise RuntimeError("已有任务在跑，先中止")
            self.open()
            self.stop.clear()
            self.job = {
                "running": True, "kind": kind, "phase": "准备", "pct": 0.0, "log": [],
                "result": None, "error": None,
                "started": time.time(), "finished": None,
            }
        th = threading.Thread(target=self._run, args=(kind, params), daemon=True)
        th.start()

    def _run(self, kind: str, params: dict) -> None:
        def emit(phase: str, pct, snap) -> None:
            self.job["phase"] = phase
            if pct is not None:
                self.job["pct"] = round(float(pct), 1)
            if snap is not None:
                self.last_state = snap.as_dict()
                self.job["log"].append({"t": round(snap.t, 2), "tag": phase,
                                        "deg": snap.deg})

        try:
            with self.lock:
                arm = self.require()
                if kind == "chain":
                    res = core.run_chain(arm, scale=params["scale"], stay=params["stay"],
                                         stop=self.stop, emit=emit)
                elif kind == "play":
                    res = core.playback_run(
                        arm, params["data"], float(params.get("speed", 1.0)), self.stop,
                        emit=lambda pct, grip: emit("回放示教轨迹", pct, None))
                elif kind == "push":
                    res = core.run_push(arm, params, self.stop, emit)
                elif kind == "task":
                    res = core.run_task(arm, params, self.stop, emit)
                else:  # home
                    import numpy as np
                    target = np.deg2rad([float(v) for v in self.home_deg[:6]] +
                                        [self.home_deg[6] if len(self.home_deg) > 6 else 0.0])
                    emit("回到起始位", 0.0, None)
                    arm.enable()
                    arm.stream(target, float(params.get("duration", 4.0)), self.stop,
                               on_tick=_pct_tick(emit, "回到起始位", float(params.get("duration", 4.0))))
                    emit("回到起始位", 100.0, arm.read())
                    res = {"ok": True, "kind": "home", "home_deg": self.home_deg}
            self.job["result"] = res
            self.job["phase"] = "完成" if kind == "home" else (
                "停在目标位" if params.get("stay") else "完成")
            self.job["pct"] = 100.0
        except core.ChainAborted as e:
            self.job["error"] = f"已中止: {e}"
        except Exception as e:
            self.job["error"] = f"{e}"
            self.job["trace"] = traceback.format_exc(limit=4)
            # 出错时【不要】卸力：直接把当前位姿锁住，避免臂突然耷拉下来
            # （要卸力请按『急停 / 卸力矩』或『失能』）
            try:
                if self.arm:
                    import numpy as _np
                    cur = self.arm.read()
                    q = _np.deg2rad([d if d is not None else 0.0 for d in cur.deg])
                    self.arm.hold(q, 0.5)
            except Exception:
                pass
        finally:
            self.job["running"] = False
            self.job["finished"] = time.time()

    def abort(self) -> dict:
        self.stop.set()
        with self.lock:
            if self.arm is not None:
                try:
                    self.arm.disable()
                except Exception:
                    pass
        return {"ok": True, "aborting": True}

    def reconnect(self) -> dict:
        """串口句柄坏掉（os error 121 / 拒绝访问）时，关掉重开。"""
        with self.lock:
            try:
                if self.arm is not None:
                    self.arm.close()
            except Exception:
                pass
            self.arm = None
        time.sleep(0.4)
        self.open()
        st = self.state()
        return {"ok": True, "reconnected": True,
                "deg": [None if v is None else round(float(v), 1) for v in st["deg"]]}

    def interrupt(self) -> dict:
        """中断当前动作但【保持使能、原地锁住】（区别于 abort：不卸力）。"""
        was_running = bool(self.job["running"])
        kind = self.job.get("kind")
        self.stop.set()
        t0 = time.time()
        while self.job["running"] and time.time() - t0 < 4.0:
            time.sleep(0.05)
        held = None
        try:
            if self.arm is not None:
                cur = self.arm.read()
                if all(v is not None for v in cur.deg):
                    q = np.deg2rad([float(v) for v in cur.deg])
                    self.arm.enable()          # 确保仍在使能
                    self.arm.hold(q, 0.8)      # 锁在当前位姿
                    held = [round(float(v), 1) for v in cur.deg]
        except Exception as e:
            return {"ok": False, "error": f"中断后锁位失败：{e}"}
        with self.lock:
            self.job["error"] = self.job.get("error") or "已中断（保持使能，原地锁住）"
            self.job["phase"] = "已中断 · 保持使能"
            self.job["running"] = False
        return {"ok": True, "was_running": was_running, "kind": kind,
                "held_deg": held, "enabled": True}


RIG: Rig | None = None


def _pct_tick(emit, phase: str, duration: float):
    """给 hold/stream 用的百分比回调（按时间估算）。"""
    t0 = time.time()

    def tick():
        pct = min(100.0, (time.time() - t0) / max(duration, 0.1) * 100.0)
        emit(phase, pct, None)

    return tick


class Handler(SimpleHTTPRequestHandler):
    server_version = "reBotDebug/1.0"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB_DIR), **kw)

    # ---------- 工具 ----------
    def _mjpeg(self) -> None:
        """持续输出 multipart MJPEG（浏览器 <img> 直接显示）。"""
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            while True:
                f = cammod.CAM.frame
                if f:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: " + str(len(f)).encode() + b"\r\n\r\n")
                    self.wfile.write(f)
                    self.wfile.write(b"\r\n")
                time.sleep(1.0 / 25.0)
        except Exception:
            return

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except Exception:
            return {}

    def log_message(self, fmt, *args):  # 安静点，只在控制台打短日志
        if "/api/" in (self.path or ""):
            return
        print(f"[web] {self.address_string()} {fmt % args}")

    # ---------- 路由 ----------
    def do_GET(self):
        assert RIG is not None
        p = self.path.split("?")[0]
        # 首页：注入当前版本号 + 禁缓存（否则改了前端、浏览器还在跑旧 JS）
        if p == "/cam":
            path = WEB_DIR / "cam.html"
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)
            return
        if p == "/cam/stream":
            return self._mjpeg()
        if p == "/api/cam/info":
            return self._json(cammod.CAM.info())
        if p == "/api/cam/devices":
            return self._json({"ok": True, "devices": cammod.list_devices()})
        if p in ("/", "/index.html"):
            try:
                ver, html = page_version()
                body = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("X-Page-Version", ver)
                self.end_headers()
                self.wfile.write(body)
                return
            except Exception as e:
                return self._json({"ok": False, "error": f"index.html 读取失败: {e}"}, 500)
        try:
            if p == "/api/info":
                return self._json({
                    "ok": True,
                    "port": RIG.port,
                    "baud": core.BAUD,
                    "held": RIG.arm is not None,
                    "running": RIG.job["running"],
                    "names": list(core.NAMES),
                    "models": [j[3] for j in core.JOINTS],
                    "plan": core.plan_summary(1.0),
                    "joint_limits_deg": [j[4] for j in core.JOINTS],
                    "urdf_limits_deg": [[round(a * 57.29578, 1), round(b * 57.29578, 1)]
                                        for a, b in core.URDF_LIMITS],
                    "home_deg": RIG.home_deg,
                    "ports": RIG.ports(),
                    "page_ver": page_version()[0],
                    "note": "J7 负=张开(自由到-140°)，正=夹紧(+10°即有负载，+18°顶死)；"
                            "主运动段末端约 前+36mm / 侧+1mm / 上+106mm（scale=1）",
                })
            if p == "/api/state":
                return self._json({"ok": True, "state": RIG.state()})
            if p == "/api/job":
                j = dict(RIG.job)
                j.pop("trace", None)
                return self._json({"ok": True, "job": j})
            if p == "/api/teach/state":
                return self._json({"ok": True, "teach": RIG.teach_state()})
            if p == "/api/leader":
                return self._json({"ok": True, **RIG.leader_state()})
            if p == "/api/slots":
                return self._json({"ok": True, "slots": RIG.slots()})
            if p == "/api/recordings":
                return self._json({"ok": True, "recordings": RIG.recordings()})
            return super().do_GET()
        except Exception as e:
            return self._json({"ok": False, "error": str(e)}, 500)

    def do_POST(self):
        assert RIG is not None
        p = self.path.split("?")[0]
        body = self._body()
        try:
            if p == "/api/open":
                RIG._teach_guard()
                return self._json({"ok": True, **RIG.open()})
            if p == "/api/release":
                if RIG.teach["running"]:
                    raise RuntimeError("示教进行中，先停止示教")
                return self._json(RIG.release())
            if p == "/api/enable":
                RIG._teach_guard()
                return self._json({"ok": True, "state": RIG.enable()})
            if p == "/api/disable":
                if RIG.teach["running"]:
                    raise RuntimeError("示教进行中，先停止示教")
                return self._json({"ok": True, "state": RIG.disable()})
            if p == "/api/run":
                RIG._teach_guard()
                scale = float(body.get("scale", 1.0))
                stay = bool(body.get("stay", False))
                if not (0.05 <= scale <= 3.0):
                    raise RuntimeError("scale 需在 0.05~3.0")
                return self._json(RIG.start(scale, stay))
            if p == "/api/reconnect":
                return self._json(RIG.reconnect())
            if p == "/api/interrupt":
                return self._json(RIG.interrupt())
            if p == "/api/abort":
                if RIG.teach["running"]:
                    RIG.teach["stop"].set()
                return self._json(RIG.abort())
            if p == "/api/teach/start":
                ke = body.get("keep_enabled", None)
                return self._json(RIG.start_teach(str(body.get("mode", "free")),
                                                  None if ke is None else bool(ke)))
            if p == "/api/leader/open":
                return self._json(RIG.leader_open(str(body.get("port") or ""),
                                                 int(body.get("baud") or 0)))
            if p == "/api/leader/close":
                return self._json(RIG.leader_close())
            if p == "/api/leader/map":
                return self._json(RIG.leader_setmap(body or {}))
            if p == "/api/teach/stop":
                return self._json(RIG.stop_teach())
            if p == "/api/teach/grip":
                return self._json(RIG.teach_grip(str(body.get("action", "close"))))
            if p == "/api/teach/record":
                return self._json(RIG.teach_record(bool(body.get("on", True))))
            if p == "/api/teach/follow":
                return self._json(RIG.teach_follow(bool(body.get("on", False))))
            if p == "/api/rec/delete":
                return self._json(RIG.rec_delete(str(body.get("name", ""))))
            if p == "/api/rec/rename":
                return self._json(RIG.rec_rename(str(body.get("name", "")),
                                                 str(body.get("new", ""))))
            if p == "/api/rec/note":
                return self._json(RIG.rec_note(str(body.get("name", "")),
                                               str(body.get("note", ""))))
            if p == "/api/grip":
                return self._json(RIG.grip(str(body.get("action", "close"))))
            if p == "/api/zero/run":
                js = body.get("joints")
                joints = None if js in (None, "", "all") else [int(v) for v in js]
                return self._json(RIG.zero_calibrate(joints, bool(body.get("dry_run", False))))
            if p == "/api/slot/save":
                return self._json(RIG.save_slot(str(body.get("name", ""))))
            if p == "/api/slot/delete":
                return self._json(RIG.delete_slot(str(body.get("name", ""))))
            if p == "/api/cam/open":
                return self._json(cammod.CAM.open(str(body.get("device") or ""),
                                                  str(body.get("size") or "1280x720")))
            if p == "/api/cam/close":
                return self._json(cammod.CAM.close())
            if p == "/api/cam/rec":
                act = str(body.get("action", "start"))
                if act == "start":
                    return self._json(cammod.CAM.start_rec(int(body.get("fps", 0) or 0)))
                return self._json(cammod.CAM.stop_rec())
            if p == "/api/jog":
                return self._json(RIG.jog(int(body.get("joint", 1)), float(body.get("delta", 0.0))))
            if p == "/api/push":
                return self._json(RIG.run_push(str(body.get("part", "A")),
                                               str(body.get("which", "push2")),
                                               float(body.get("dur", 8.0))))
            if p == "/api/task/run":
                return self._json(RIG.run_task(str(body.get("part", "A")),
                                               str(body.get("form", "upright")),
                                               float(body.get("angle", 0.0)),
                                               bool(body.get("dry_run", False))))
            if p == "/api/play":
                return self._json(RIG.play(str(body.get("name", "")),
                                           float(body.get("speed", 1.0))))
            if p == "/api/record_home":
                return self._json(RIG.record_home())
            if p == "/api/home":
                return self._json(RIG.go_home(float(body.get("duration", 4.0))))
            if p == "/api/port":
                com = str(body.get("com") or "").strip()
                if not com:
                    raise RuntimeError("缺少 com 参数")
                return self._json(RIG.set_port(com))
            return self._json({"ok": False, "error": "unknown endpoint"}, 404)
        except Exception as e:
            return self._json({"ok": False, "error": str(e)}, 500)


def main() -> None:
    global RIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--com", default=core.DEFAULT_PORT)
    ap.add_argument("--http-port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    RIG = Rig(args.com)
    srv = ThreadingHTTPServer((args.host, args.http_port), Handler)
    print(f"reBot B601-DM 调试前端: http://{args.host}:{args.http_port}")
    print(f"串口: {args.com} @{core.BAUD}（首次读状态时占用；点『释放串口』可交给 MotorBridge）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n退出，释放串口…")
    finally:
        try:
            RIG.release()
        finally:
            srv.server_close()


if __name__ == "__main__":
    main()

