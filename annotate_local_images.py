#!/usr/bin/env python

import os
import json
import argparse
import cv2
import numpy as np

def draw_polygon(img, polygon_points, color=(0, 255, 0), thickness=2):
    """
    Draws a polygon on the image using the provided list of (x, y) points.
    """
    pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def draw_bbox(img, bbox, color=(255, 0, 0), thickness=2):
    """
    Draws a bounding box on the image.
    bbox = [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

def annotate_images(json_file, images_dir, output_dir):
    """
    Reads the label JSON file, draws bounding boxes and polygons for each cell,
    writes annotated images to the output directory.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for item in data:
        tile_path = item["tile_path"]
        # Extract the subpath after 'INPUT_prepped_tiles_path/'
        # This handles variable prefixes (e.g. /mnt/...).
        parts = tile_path.split("INPUT_prepped_tiles_path/", 1)
        if len(parts) < 2:
            print(f"Warning: tile_path {tile_path} does not contain 'INPUT_prepped_tiles_path/'. Skipping.")
            continue

        # The part after 'INPUT_prepped_tiles_path/'
        subpath = parts[1].lstrip("/")

        # Construct the local path to the image
        local_image_path = os.path.join(images_dir, subpath)

        if not os.path.exists(local_image_path):
            print(f"Warning: {local_image_path} does not exist.")
            continue

        img = cv2.imread(local_image_path)
        if img is None:
            print(f"Warning: Failed to load image {local_image_path}. Skipping.")
            continue

        # Draw bounding boxes, polygons, and labels
        for cell in item["cells"]:
            label_id = cell["label_id"]
            polygon = cell["polygon"]  # list of [x, y]
            bbox = cell["bbox"]        # [x_min, y_min, x_max, y_max]
            pred_class = cell["pred_class"]

            draw_bbox(img, bbox)
            draw_polygon(img, polygon)
            
            # Place text (label) near the top-left corner of bbox
            label_str = f"Cls:{pred_class},ID:{label_id}"
            x_min, y_min, _, _ = bbox
            cv2.putText(img, label_str, (x_min, max(0, y_min - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save the annotated image
        output_path = os.path.join(output_dir, os.path.basename(subpath))
        cv2.imwrite(output_path, img)
        print(f"Annotated image saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Draw bounding boxes, polygons, and labels on images.")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the label JSON file.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing the images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the annotated images.")
    args = parser.parse_args()

    annotate_images(args.json_file, args.images_dir, args.output_dir)

if __name__ == "__main__":
    main()
