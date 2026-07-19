"""
train.py

Fine-tunes a pretrained YOLOv8 backbone on the single-class person/head
detection dataset in this repo.

Usage:
    python train.py --data dataset.yaml --epochs 50 --batch 16
    python train.py --data dataset.yaml --epochs 100 --batch 8 --model yolov8s.pt --imgsz 960

Before running this, convert the COCO annotations to YOLO labels:
    python utils/convert_coco_to_yolo.py --coco images/train/_annotations.coco.json --images images/train --out labels/train
    python utils/convert_coco_to_yolo.py --coco images/valid/_annotations.coco.json --images images/valid --out labels/valid
    python utils/convert_coco_to_yolo.py --coco images/test/_annotations.coco.json  --images images/test  --out labels/test
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 detector on this repo's dataset.")
    parser.add_argument("--data", default="dataset.yaml", help="Path to dataset YAML config")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="Pretrained backbone to fine-tune, e.g. yolov8n.pt / yolov8s.pt / yolov8m.pt / yolov8l.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs/train", help="Directory to save training runs under")
    parser.add_argument("--name", default="exp", help="Name of this training run (subfolder under --project)")
    parser.add_argument("--device", default=None, help="Device to train on, e.g. '0' for first GPU, 'cpu' for CPU")
    parser.add_argument("--resume", action="store_true", help="Resume the most recent interrupted run")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found at {data_path}. "
            f"Did you run utils/convert_coco_to_yolo.py to generate YOLO labels first?"
        )

    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print("\nTraining complete.")
    print(f"Best weights: {best_weights}")
    print("Run inference with:")
    print(f"  python predict.py {best_weights} path/to/image.jpg")


if __name__ == "__main__":
    main()
