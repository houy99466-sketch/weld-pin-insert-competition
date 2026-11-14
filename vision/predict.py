"""
推理脚本：图片/视频 → 每个工件输出 {cls, conf, center_px, angle_deg, w, h}
角度定义：OBB 的 r（弧度），这里转成「与图像 x 轴夹角，逆时针为正，范围 [-90, 90)」。

用法：
  python predict.py --model runs\obb\train\weights\best.pt --src xxx.mp4
  python predict.py --model best.pt --src image.jpg --show
  python predict.py --model best.pt --src xxx.mp4 --out out.mp4 --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def angle_deg_from_r(r: float) -> float:
    """把 OBB 的旋转角转到 [-90, 90)，方便和机械臂 yaw 对齐。"""
    a = math.degrees(r) % 180.0
    if a >= 90.0:
        a -= 180.0
    return round(a, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="", help="输出视频路径（仅视频输入有效）")
    ap.add_argument("--json", default="", help="把每帧检测结果写成 json")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max", type=int, default=0, help="只处理前 N 帧（0=全部）")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    recs: list[dict] = []

    def to_rec(res, idx: int) -> dict:
        items = []
        obb = getattr(res, "obb", None)
        if obb is not None and obb.xywhr is not None and len(obb.xywhr):
            for xywhr, cls, conf in zip(obb.xywhr.cpu().numpy(),
                                        obb.cls.cpu().numpy(),
                                        obb.conf.cpu().numpy()):
                if float(conf) < args.conf:
                    continue
                cx, cy, w, h, r = [float(v) for v in xywhr]
                items.append({
                    "cls": res.names[int(cls)],
                    "conf": round(float(conf), 3),
                    "center_px": [round(cx, 1), round(cy, 1)],
                    "size_px": [round(w, 1), round(h, 1)],
                    "angle_deg": angle_deg_from_r(r),
                })
        return {"frame": idx, "objects": items}

    src = Path(args.src)
    is_video = src.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov")

    if is_video:
        import cv2
        cap = cv2.VideoCapture(str(src))
        idx = 0
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = model.predict(frame, conf=args.conf, verbose=False)[0]
            rec = to_rec(res, idx)
            recs.append(rec)
            drawn = res.plot()
            if args.out:
                if writer is None:
                    h, w = drawn.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                             25.0, (w, h))
                writer.write(drawn)
            if args.show:
                cv2.imshow("predict", drawn)
                if cv2.waitKey(1) == 27:
                    break
            if rec["objects"]:
                o = rec["objects"][0]
                print(f"  #{idx:5d} {o['cls']:<8} conf={o['conf']:.2f} "
                      f"center={o['center_px']} angle={o['angle_deg']:+.1f}°")
            idx += 1
            if args.max and idx >= args.max:
                break
        cap.release()
        if writer:
            writer.release()
            print("输出视频:", args.out)
    else:
        res = model.predict(str(src), conf=args.conf, verbose=False)[0]
        rec = to_rec(res, 0)
        recs.append(rec)
        for o in rec["objects"]:
            print(f"  {o['cls']:<8} conf={o['conf']:.2f} center={o['center_px']} "
                  f"angle={o['angle_deg']:+.1f}°")
        if args.out:
            import cv2
            cv2.imwrite(args.out, res.plot())
            print("输出图片:", args.out)

    if args.json:
        Path(args.json).write_text(json.dumps(recs, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print("结果 json:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
