#!/usr/bin/env python

import os
import json
import argparse
import random
import cv2
import numpy as np

def draw_polygon(img, polygon_points, color=(0, 255, 0), thickness=2):
    """
    Draws a polygon on the image using the provided list of (x, y) points.
    polygon_points is a list of [x, y] coordinates.
    """
    pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def draw_bbox(img, bbox, color=(255, 0, 0), thickness=2):
    """
    Draws a bounding box on the image, where bbox = [x_min, y_min, x_max, y_max].
    """
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

def annotate_images(json_file, images_dir, output_dir,
                    max_labels, random_labels,
                    draw_bbox_flag, draw_polygon_flag,
                    text_scale):
    """
    Reads the label JSON file, draws bounding boxes and/or polygons for each cell,
    then writes annotated images to the output directory.

    If random_labels is True, we shuffle the cells first, and then truncate
    to max_labels. That same subset of cells is used for both bboxes and polygons.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for item in data:
        tile_path = item["tile_path"]
        # Extract the subpath after 'INPUT_prepped_tiles_path/'
        parts = tile_path.split("INPUT_prepped_tiles_path/", 1)
        if len(parts) < 2:
            print(f"Warning: tile_path {tile_path} does not contain 'INPUT_prepped_tiles_path/'. Skipping.")
            continue

        subpath = parts[1].lstrip("/")
        local_image_path = os.path.join(images_dir, subpath)

        if not os.path.exists(local_image_path):
            print(f"Warning: {local_image_path} does not exist. Skipping.")
            continue

        img = cv2.imread(local_image_path)
        if img is None:
            print(f"Warning: Failed to load image {local_image_path}. Skipping.")
            continue

        # Extract cells for this tile
        cells = item["cells"]

        # Possibly shuffle the cells if random_labels is True
        if random_labels:
            random.shuffle(cells)

        # Draw bounding boxes/polygons for the selected cells
        label_count = 0
        for cell in cells:
            pred_class = cell["pred_class"]
            if len(pred_class.split()) > 1:
                # pred_class = "undetermined"
                continue
            bbox = cell["bbox"]       # [x_min, y_min, x_max, y_max]
            polygon = cell["polygon"] # list of [x, y]

            # If both bounding box and polygon are selected, they get drawn for the SAME cell
            if draw_bbox_flag:
                draw_bbox(img, bbox)

            if draw_polygon_flag:
                draw_polygon(img, polygon)

            # If at least one of them is drawn, add the label once
            if draw_bbox_flag or draw_polygon_flag:
                # Only the class label is shown
                label_str = f"C:{pred_class}, id:{cell['label_id']}"
                x_min, y_min, _, _ = bbox
                # Place text just above the top-left of the bounding box
                cv2.putText(img,
                            label_str,
                            (x_min, max(0, y_min - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            text_scale,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA)
            label_count += 1
            if label_count >= max_labels:
                break

        # Save the annotated image
        out_filename = os.path.basename(subpath)
        output_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(output_path, img)
        print(f"Annotated image saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Draw bounding boxes and/or polygons on images.")
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to the label JSON file.")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing the images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the annotated images.")
    parser.add_argument("--max_labels", type=int, default=999999,
                        help="Maximum number of labels to annotate per image.")
    parser.add_argument("--random_labels", action="store_true",
                        help="Pick labels randomly from the full list.")
    parser.add_argument("--draw_bbox", action="store_true",
                        help="If set, draw bounding boxes.")
    parser.add_argument("--draw_polygon", action="store_true",
                        help="If set, draw polygons.")
    parser.add_argument("--text_scale", type=float, default=0.6,
                        help="Scale factor for label text size.")
    args = parser.parse_args()

    annotate_images(
        json_file=args.json_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        max_labels=args.max_labels,
        random_labels=args.random_labels,
        draw_bbox_flag=args.draw_bbox,
        draw_polygon_flag=args.draw_polygon,
        text_scale=args.text_scale
    )

if __name__ == "__main__":
    main()
