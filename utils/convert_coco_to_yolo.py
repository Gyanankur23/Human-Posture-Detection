"""
convert_coco_to_yolo.py

Converts a Roboflow-style COCO annotation file (_annotations.coco.json) into
YOLO-format .txt label files (one per image, class + normalized xywh).

This dataset ships with a single real object class (all annotations use the
same category_id despite the odd category name inherited from the source
Roboflow project), so this converter maps every category onto class 0
("person") by default. If your dataset later gains real multi-class labels,
edit `mapping` below to map COCO category_id -> YOLO class index.

Usage:
    python utils/convert_coco_to_yolo.py --coco images/train/_annotations.coco.json --images images/train --out images/train/labels
    python utils/convert_coco_to_yolo.py --coco images/valid/_annotations.coco.json --images images/valid --out images/valid/labels
    python utils/convert_coco_to_yolo.py --coco images/test/_annotations.coco.json  --images images/test  --out images/test/labels
"""

import argparse
import json
import os
from collections import defaultdict


def build_mapping(categories):
    """
    Default: collapse every COCO category onto a single YOLO class (0).
    Edit this dict if you introduce real multi-class labels later, e.g.:
        mapping = {0: 0, 1: 0}  # both current categories -> class 0
    """
    mapping = {cat["id"]: 0 for cat in categories}
    return mapping


def convert(coco_path, images_dir, out_dir):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    images_by_id = {img["id"]: img for img in coco["images"]}
    mapping = build_mapping(coco["categories"])

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    n_images = 0
    n_boxes = 0
    n_skipped_categories = set()

    for image_id, img in images_by_id.items():
        width = img["width"]
        height = img["height"]
        file_name = img["file_name"]
        stem = os.path.splitext(file_name)[0]
        label_path = os.path.join(out_dir, stem + ".txt")

        lines = []
        for ann in anns_by_image.get(image_id, []):
            cat_id = ann["category_id"]
            if cat_id not in mapping:
                n_skipped_categories.add(cat_id)
                continue
            cls = mapping[cat_id]

            x, y, w, h = ann["bbox"]  # COCO bbox: top-left x, y, width, height
            # normalize + convert to center-based xywh for YOLO
            cx = (x + w / 2.0) / width
            cy = (y + h / 2.0) / height
            nw = w / width
            nh = h / height

            # clip to [0, 1] in case of any rounding overshoot
            cx, cy, nw, nh = (max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh))

            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_boxes += 1

        # Always write a label file, even if empty (image with no objects)
        with open(label_path, "w") as lf:
            lf.write("\n".join(lines))

        n_images += 1

    print(f"[convert_coco_to_yolo] {coco_path}")
    print(f"  images processed: {n_images}")
    print(f"  boxes written:    {n_boxes}")
    print(f"  labels written to: {out_dir}")
    if n_skipped_categories:
        print(f"  WARNING: skipped unmapped category ids: {sorted(n_skipped_categories)}")


def main():
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO txt labels.")
    parser.add_argument("--coco", required=True, help="Path to _annotations.coco.json")
    parser.add_argument("--images", required=True, help="Path to the image directory for this split")
    parser.add_argument("--out", required=True, help="Output directory for YOLO .txt label files")
    args = parser.parse_args()

    convert(args.coco, args.images, args.out)


if __name__ == "__main__":
    main()
