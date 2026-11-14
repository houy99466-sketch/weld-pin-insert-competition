"""
从录制的 mp4 里抽帧，做 YOLO OBB 数据集的原图。
用法：
    python extract_frames.py --src ..\real_robot\web\cam_videos --dst dataset\raw --every 5
依赖：opencv（ultralytics 会顺带装上）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="视频文件夹或单个 mp4")
    ap.add_argument("--dst", required=True, help="输出图片目录")
    ap.add_argument("--every", type=int, default=5, help="每 N 帧取一张（默认 5）")
    ap.add_argument("--max", type=int, default=0, help="每个视频最多取多少张（0=不限）")
    ap.add_argument("--prefix", default="", help="文件名前缀，例如 A_side_")
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        print("缺少 opencv：请先 pip install opencv-python", file=sys.stderr)
        return 2

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    vids = [src] if src.is_file() else sorted(src.glob("*.mp4"))
    if not vids:
        print(f"{src} 下没有 mp4")
        return 1

    total = 0
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f"  跳过（打不开）{v.name}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = 0
        idx = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(1, args.every) == 0:
                name = f"{args.prefix}{v.stem}_{idx:06d}.jpg"
                cv2.imwrite(str(dst / name), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved += 1
                if args.max and saved >= args.max:
                    break
            idx += 1
            n += 1
        cap.release()
        total += saved
        print(f"  {v.name}: {n} 帧 @ {fps:.0f}fps → 取 {saved} 张")
    print(f"合计 {total} 张 → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
