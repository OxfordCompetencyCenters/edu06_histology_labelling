from __future__ import annotations
import argparse, glob, json, logging, os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure

# ─────────────────────────── helpers ─────────────────────────── #

def parse_tile_filename(filename: str) -> dict | None:
    """
    Parse new tile filename format without regex.
    Example: 'slide_id__MAG_1d000__X_2048__Y_1024__IDX_000001.png'
    Returns: {'slide_id': 'slide_id', 'magnification': 1.0, 'x': 2048, 'y': 1024, 'idx': 1}
    """
    # Remove directory path if present
    basename = os.path.basename(filename)
    
    # Remove .png extension if present
    if basename.endswith('.png'):
        basename = basename[:-4]
    
    # Check if it contains the new format markers
    if '__MAG_' not in basename or '__X_' not in basename or '__Y_' not in basename or '__IDX_' not in basename:
        return None
    
    try:
        # Find the positions of the required components
        mag_idx = basename.find('__MAG_')
        x_idx = basename.find('__X_')
        y_idx = basename.find('__Y_')
        idx_idx = basename.find('__IDX_')
        
        if mag_idx == -1 or x_idx == -1 or y_idx == -1 or idx_idx == -1:
            return None
        
        # Extract slide_id (everything before __MAG_)
        slide_id = basename[:mag_idx]
        
        # Parse magnification: MAG_1d000 -> 1.000
        mag_part = basename[mag_idx+2:x_idx]  # Skip '__' and go until next '__X_'
        if not mag_part.startswith('MAG_'):
            return None
        mag_value = mag_part[4:]  # Remove 'MAG_' prefix
        if 'd' not in mag_value:
            return None
        int_part, frac_part = mag_value.split('d')
        magnification = int(int_part) + (int(frac_part) / 1000.0)
        
        # Parse X coordinate: X_2048 -> 2048
        x_part = basename[x_idx+2:y_idx]  # Skip '__' and go until next '__Y_'
        if not x_part.startswith('X_'):
            return None
        x = int(x_part[2:])
        
        # Parse Y coordinate: Y_1024 -> 1024
        y_part = basename[y_idx+2:idx_idx]  # Skip '__' and go until next '__IDX_'
        if not y_part.startswith('Y_'):
            return None
        y = int(y_part[2:])
        
        # Parse index: IDX_000001 -> 1
        idx_part = basename[idx_idx+2:]  # Skip '__' and take rest
        if not idx_part.startswith('IDX_'):
            return None
        idx = int(idx_part[4:])  # Remove 'IDX_' prefix
        
        return {
            "slide_id": slide_id,
            "magnification": magnification,
            "x": x,
            "y": y,
            "idx": idx,
        }
        
    except (ValueError, IndexError):
        return None

def parse_tile_meta(tile_name: str) -> dict[str, float | int] | None:
    """
    Parse tile metadata from filename using the new format only.
    """
    return parse_tile_filename(tile_name)

def mask_to_polygon_list(mask: np.ndarray):
    """Convert a labelled mask to a list of polygons (one per label ID)."""
    polygons = []
    for lbl in np.unique(mask):
        if lbl == 0:
            continue
        binary = (mask == lbl).astype(np.uint8)
        contours = measure.find_contours(binary, 0.5)
        if not contours:
            continue
        # choose longest contour
        contour = max(contours, key=lambda c: c.shape[0])
        # swap (row, col)→(x, y) and cast to float for JSON serialisation
        poly = [(float(pt[1]), float(pt[0])) for pt in contour]
        polygons.append({"label_id": int(lbl), "polygon": poly})
    return polygons

def find_classification_files(cls_root: Path) -> dict:
    """Locate classification_results.json files per slide directory."""
    classification_files = {}
    
    # Look for per-slide structure: slide_name/classification_results.json
    for slide_dir in cls_root.iterdir():
        if slide_dir.is_dir():
            cls_file = slide_dir / "classification_results.json"
            if cls_file.exists():
                classification_files[slide_dir.name] = cls_file
    
    # Fallback: check for single global file
    if not classification_files:
        global_file = cls_root / "classification_results.json"
        if global_file.exists():
            classification_files["global"] = global_file
    
    return classification_files

# ───────────────────────────── main ───────────────────────────── #

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmentation_path", required=True)
    ap.add_argument("--classification_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--param_string", default="")
    args = ap.parse_args()

    seg_root = Path(args.segmentation_path)
    cls_root = Path(args.classification_path)
    out_root = Path(args.output_path)
    out_root.mkdir(parents=True, exist_ok=True)

    classification_files = find_classification_files(cls_root)
    if not classification_files:
        logging.error("No classification_results.json files found under %s", cls_root)
        return

    logging.info("Found classification files for slides: %s", list(classification_files.keys()))
    
    # Process each slide separately
    for slide_name, cls_file in classification_files.items():
        logging.info("Processing slide: %s from %s", slide_name, cls_file)
        
        # Create slide-specific output directory
        slide_out_dir = out_root / slide_name
        slide_out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            classification_results = json.loads(cls_file.read_text())
        except Exception as e:
            logging.error("Failed to load classification file %s: %s", cls_file, e)
            continue

        final_ann = []
        cell_id_counter = 1  # Reset counter for each slide

        for tile in classification_results:
            tile_path = tile["tile_path"]
            tile_name = tile["tile_name"]
            current_slide_name = tile["slide_name"]

            meta = parse_tile_meta(tile_name)
            if meta is None:
                logging.warning("Could not parse magnification from %s – skipping.", tile_name)
                continue

            mask_name = tile_name + "_mask.png"
            mask_path = seg_root / current_slide_name / mask_name
            if not mask_path.exists():
                logging.warning("Mask not found: %s", mask_path)
                continue

            mask = np.array(Image.open(mask_path))
            polys = mask_to_polygon_list(mask)

            cell_records = []
            for poly in polys:
                lbl_id = poly["label_id"]
                cls_cell = next((c for c in tile["classified_cells"] if c["label_id"] == lbl_id), None)
                if cls_cell is None:
                    logging.debug("No class for lbl=%d in %s", lbl_id, tile_name)
                    continue
                cell_records.append({
                    "cell_id": cell_id_counter,
                    "label_id": lbl_id,
                    "polygon": poly["polygon"],
                    "pred_class": cls_cell["pred_class"],
                    "cluster_id": cls_cell.get("cluster_id"),
                    "cluster_confidence": cls_cell.get("cluster_confidence"),
                    "bbox": cls_cell["bbox"],
                })
                cell_id_counter += 1  # Increment counter for next cell

            final_ann.append({
                "tile_path": tile_path,
                "magnification": meta["magnification"],
                "x": meta["x"],
                "y": meta["y"],
                "cells": cell_records,
            })

        # Write output for this slide
        out_name = f"v1_{args.param_string}_annotations.json" if args.param_string else "final_annotations.json"
        out_file = slide_out_dir / out_name
        out_file.write_text(json.dumps(final_ann, indent=2))
        logging.info("Annotations for slide %s written → %s", slide_name, out_file)

# ──────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    main()
