"""
predict.py

Runs inference with a trained YOLOv8 .pt model on a single image (or a
folder / glob of images) and saves annotated output.

Usage:
    python predict.py runs/train/exp/weights/best.pt path/to/image.jpg
    python predict.py runs/train/exp/weights/best.pt path/to/folder --conf 0.35 --out runs/predict
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on an image or folder.")
    parser.add_argument("weights", help="Path to trained model weights (.pt)")
    parser.add_argument("source", help="Path to an image, folder of images, or glob pattern")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default="runs/predict", help="Directory to save annotated results under")
    parser.add_argument("--device", default=None, help="Device to run on, e.g. '0' or 'cpu'")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model = YOLO(str(weights_path))

    predict_kwargs = dict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        project=args.out,
        name="",
        exist_ok=True,
        save=True,
    )
    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)

    for r in results:
        n_boxes = 0 if r.boxes is None else len(r.boxes)
        print(f"{r.path}: {n_boxes} detection(s)")

    print(f"\nAnnotated output saved under: {args.out}")


if __name__ == "__main__":
    main()
