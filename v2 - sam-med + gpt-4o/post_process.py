import argparse
import os
import json
import logging

import numpy as np
from PIL import Image
from skimage import measure

def mask_to_polygon_list(mask):
    """
    Converts a labeled mask (numpy array) into a list of polygons.
    Return format: 
    [
      {
        'label_id': 1,
        'polygon': [(x1, y1), (x2, y2), ...]
      },
      ...
    ]
    """
    polygons = []
    unique_labels = np.unique(mask)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        binary_mask = (mask == lbl).astype(np.uint8)
        contours = measure.find_contours(binary_mask, 0.5)
        if len(contours) > 0:
            main_contour = max(contours, key=lambda x: x.shape[0])
            poly = [(float(pt[1]), float(pt[0])) for pt in main_contour]
            polygons.append({"label_id": int(lbl), "polygon": poly})
    return polygons

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation_path", type=str, help="Path to segmentation masks.")
    parser.add_argument("--classification_path", type=str, help="Path to classification results.")
    parser.add_argument("--output_path", type=str, help="Final output location.")
    args = parser.parse_args()

    logging.info("Starting post-processing with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)

    class_results_file = os.path.join(args.classification_path, "classification_results.json")
    if not os.path.exists(class_results_file):
        logging.warning("No classification_results.json found in input path: %s", args.classification_path)
        return
    
    with open(class_results_file, "r") as f:
        classification_results = json.load(f)

    final_annotations = []

    for tile_result in classification_results:
        tile_path = tile_result["tile_path"]
        parent_dir = tile_result["slide_name"]
        tile_name = tile_result["tile_name"]
        mask_name = tile_name + "_mask.png"
        mask_path = os.path.join(args.segmentation_path, parent_dir, mask_name)
        if not os.path.exists(mask_path):
            logging.warning(f"Mask {mask_path} not found for tile: {tile_path}")
            continue

        mask_img = Image.open(mask_path)
        mask_arr = np.array(mask_img)
        polygons = mask_to_polygon_list(mask_arr)

        cell_records = []
        for poly_info in polygons:
            lbl_id = poly_info["label_id"]
            classified_cell = next(
                (c for c in tile_result["classified_cells"] if c["label_id"] == lbl_id),
                None
            )
            if classified_cell is None:
                logging.info("No classification found for label_id=%d in tile=%s", lbl_id, tile_path)
                continue
            
            cell_records.append({
                "label_id": lbl_id,
                "polygon": poly_info["polygon"],
                "pred_class": classified_cell["pred_class"],
                "cluster_id": classified_cell.get("cluster_id"),
                "cluster_confidence": classified_cell.get("cluster_confidence"),
                "bbox": classified_cell["bbox"]
            })

        final_annotations.append({
            "tile_path": tile_path,
            "cells": cell_records
        })

    final_json = os.path.join(args.output_path, "final_annotations.json")
    with open(final_json, "w") as f:
        json.dump(final_annotations, f, indent=2)
    
    logging.info("Post-processing step done. Output at: %s", final_json)

if __name__ == "__main__":
    main()
