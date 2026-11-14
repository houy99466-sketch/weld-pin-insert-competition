"""
摄像头模块：ffmpeg(DirectShow) 抓流 → MJPEG 预览 + 按需录制 mp4。
不依赖 OpenCV，只用系统已有 ffmpeg。
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path

FFMPEG = r"C:\Users\21104\.local\bin\ffmpeg.exe"
REC_DIR = Path(__file__).resolve().parent / "cam_videos"
REC_DIR.mkdir(exist_ok=True)

PREVIEW_FPS = 30
JPEG_Q = 4


def _sizeof_jpeg(buf: bytes):
    """从 JPEG 头里取宽高。"""
    i = 2
    n = len(buf)
    while i + 9 < n:
        if buf[i] != 0xFF:
            i += 1
            continue
        m = buf[i + 1]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = (buf[i + 5] << 8) | buf[i + 6]
            w = (buf[i + 7] << 8) | buf[i + 8]
            return w, h
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        seg = (buf[i + 2] << 8) | buf[i + 3]
        i += 2 + seg
    return None


def list_devices() -> list[str]:
    try:
        r = subprocess.run([FFMPEG, "-hide_banner", "-list_devices", "true",
                            "-f", "dshow", "-i", "dummy"],
                           capture_output=True, text=True, timeout=15)
        out = (r.stdout or "") + (r.stderr or "")
        return re.findall(r'"([^"]+)"\s*\(video\)', out)
    except Exception:
        return []


class CamRig:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.frame: bytes | None = None
        self.frame_no = 0
        self.fps = 0.0
        self.size = (0, 0)
        self.device = ""
        self.error = ""
        self._stop = threading.Event()
        # 录制
        self.writer: subprocess.Popen | None = None
        self.rec_file: Path | None = None
        self.rec_start = 0.0
        self.rec_frames = 0

    # ---------------- 打开/关闭 ----------------
    def open(self, device: str | None = None, size: str = "1280x720") -> dict:
        self.close()
        dev = device or self.device
        if not dev:
            devs = list_devices()
            dev = devs[0] if devs else ""
        if not dev:
            raise RuntimeError("没有找到摄像头（ffmpeg -list_devices 为空）")
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error",
               "-f", "dshow", "-rtbufsize", "128M",
               "-framerate", str(PREVIEW_FPS)]
        if size:
            cmd += ["-video_size", size]
        cmd += ["-i", f"video={dev}",
                "-f", "mjpeg", "-q:v", str(JPEG_Q), "-r", str(PREVIEW_FPS), "pipe:1"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        self.device = dev
        self.error = ""
        self._stop.clear()
        self.frame = None
        self.frame_no = 0
        self.thread = threading.Thread(target=self._reader, args=(self.proc,),
                                       daemon=True)
        self.thread.start()
        # 等首帧
        t0 = time.time()
        while time.time() - t0 < 6.0:
            if self.frame:
                break
            if self.proc.poll() is not None:
                err = (self.proc.stderr.read() or b"").decode("utf-8", "replace")[-400:]
                self.error = err or "ffmpeg 启动失败"
                raise RuntimeError(self.error)
            time.sleep(0.05)
        if not self.frame:
            raise RuntimeError("摄像头 6 秒内没有出图（可能是分辨率不支持或设备被占用）")
        sz = _sizeof_jpeg(self.frame)
        if sz:
            self.size = sz
        return self.info()

    def close(self) -> dict:
        self.stop_rec()
        self._stop.set()
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None
        self.frame = None
        self.device = ""
        return {"ok": True}

    # ---------------- 取流线程 ----------------
    def _reader(self, proc: subprocess.Popen) -> None:
        buf = bytearray()
        t_last = time.time()
        cnt = 0
        while not self._stop.is_set() and proc.poll() is None:
            chunk = proc.stdout.read(65536) if proc.stdout else b""
            if not chunk:
                time.sleep(0.005)
                continue
            buf += chunk
            while True:
                s = buf.find(b"\xff\xd8")
                if s < 0:
                    buf.clear()
                    break
                e = buf.find(b"\xff\xd9", s + 2)
                if e < 0:
                    if s:
                        del buf[:s]
                    break
                frame = bytes(buf[s:e + 2])
                del buf[:e + 2]
                self.frame = frame
                self.frame_no += 1
                cnt += 1
                with self.lock:
                    w = self.writer
                    if w is not None and w.stdin is not None:
                        try:
                            w.stdin.write(frame)
                            self.rec_frames += 1
                        except Exception:
                            self.writer = None
            now = time.time()
            if now - t_last >= 1.0:
                self.fps = cnt / (now - t_last)
                cnt = 0
                t_last = now
                if self.frame:
                    sz = _sizeof_jpeg(self.frame)
                    if sz:
                        self.size = sz

    # ---------------- 录制 ----------------
    def start_rec(self, fps: int = 0) -> dict:
        if not self.frame:
            raise RuntimeError("摄像头还没出图，无法录制")
        if self.writer is not None:
            raise RuntimeError("已经在录制中")
        sz = _sizeof_jpeg(self.frame) or self.size
        if not sz:
            raise RuntimeError("拿不到画面尺寸")
        w, h = sz
        self.size = (w, h)
        name = time.strftime("cam_%Y%m%d_%H%M%S") + ".mp4"
        path = REC_DIR / name
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "mjpeg", "-framerate", str(int(fps or PREVIEW_FPS)),
               "-i", "pipe:0",
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
               "-pix_fmt", "yuv420p", str(path)]
        self.writer = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE, bufsize=0)
        self.rec_file = path
        self.rec_start = time.time()
        self.rec_frames = 0
        return self.info()

    def stop_rec(self) -> dict:
        w = self.writer
        self.writer = None
        if w is None:
            return {"ok": True, "recording": False}
        try:
            w.stdin.close()
        except Exception:
            pass
        try:
            w.wait(timeout=12)
        except Exception:
            try:
                w.kill()
            except Exception:
                pass
        f = self.rec_file
        size_mb = round(f.stat().st_size / 1e6, 2) if f and f.exists() else 0
        return {"ok": True, "recording": False, "file": f.name if f else None,
                "size_mb": size_mb, "frames": self.rec_frames,
                "duration": round(time.time() - self.rec_start, 1)}

    # ---------------- 状态 ----------------
    def info(self) -> dict:
        rec = self.writer is not None
        return {"ok": True, "device": self.device,
                "opened": self.proc is not None and self.proc.poll() is None,
                "width": self.size[0], "height": self.size[1],
                "fps": round(self.fps, 1), "frames": self.frame_no,
                "recording": rec,
                "rec_file": self.rec_file.name if (rec and self.rec_file) else None,
                "rec_seconds": round(time.time() - self.rec_start, 1) if rec else 0.0,
                "rec_frames": self.rec_frames if rec else 0,
                "videos": sorted([p.name for p in REC_DIR.glob("*.mp4")],
                                 key=lambda s: -os.path.getmtime(REC_DIR / s))[:40],
                "error": self.error}


CAM = CamRig()
