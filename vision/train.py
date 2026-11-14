"""
YOLO11-OBB 训练脚本（针对 RTX 4060 8GB 调参）。

用法：
  python train.py --data dataset\data.yaml
  python train.py --data dataset\data.yaml --epochs 200 --batch 8 --model yolo11s-obb.pt
训练完权重：runs\obb\<name>\weights\best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).parent / "dataset" / "data.yaml"))
    ap.add_argument("--model", default="yolo11n-obb.pt",
                    help="预训练起点：yolo11n-obb.pt（快）/ yolo11s-obb.pt（更准）")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="train")
    ap.add_argument("--device", default="0", help="0=GPU，cpu=CPU")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        name=args.name,
        project=str(Path(__file__).parent / "runs" / "obb"),
        exist_ok=True,
        resume=args.resume,
        # 增强：金属件反光/角度多变，适度开
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=180.0,          # OBB 关键：允许任意旋转，学角度不变性
        translate=0.1, scale=0.5, shear=2.0, perspective=0.0005,
        fliplr=0.5, flipud=0.5,
        mosaic=1.0, mixup=0.0, copy_paste=0.1, close_mosaic=15,
        patience=40,
        save=True, plots=True, val=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print("\n训练完成，最佳权重：", best)
    m = model.val(data=args.data, split="val")
    try:
        print(f"mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
