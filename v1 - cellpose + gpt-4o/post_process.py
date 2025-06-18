#!/usr/bin/env python3
"""
Post-processing: merge segmentation masks + GPT classifications
into a single JSON annotation file.

Updated 2025-06-18 – supports new mag-encoded tile filenames.
"""

from __future__ import annotations
import argparse, glob, json, logging, os, re
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure

# ─────────────────────────── helpers ─────────────────────────── #

_TILE_REGEX = re.compile(
    r"""                # e.g.  Slide42_mag0p900_x2048_y1024.png
    _mag(?P<mag>[0-9]+p[0-9]+)
    _x(?P<x>\d+)
    _y(?P<y>\d+)
    \.png$""",
    re.VERBOSE,
)

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

def find_classification_file(cls_root: Path) -> Path | None:
    """Locate classification_results.json (top-level or one directory deep)."""
    direct = cls_root / "classification_results.json"
    if direct.exists():
        return direct
    candidates = list(cls_root.glob("*/classification_results.json"))
    if candidates:
        return candidates[0]
    return None

def parse_tile_meta(tile_name: str) -> dict[str, float | int] | None:
    """
    Given 'Slide_mag1p000_x2048_y1024.png' return:
    {'magnification': 1.0, 'x': 2048, 'y': 1024}
    """
    m = _TILE_REGEX.search(tile_name)
    if not m:
        return None
    mag = float(m.group("mag").replace("p", ".", 1))
    return {
        "magnification": mag,
        "x": int(m.group("x")),
        "y": int(m.group("y")),
    }

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

    cls_file = find_classification_file(cls_root)
    if cls_file is None:
        logging.error("classification_results.json not found under %s", cls_root)
        return

    logging.info("Loading classifications from %s", cls_file)
    classification_results = json.loads(cls_file.read_text())

    final_ann = []

    for tile in classification_results:
        tile_path = tile["tile_path"]
        tile_name = tile["tile_name"]
        slide_name = tile["slide_name"]

        meta = parse_tile_meta(tile_name)
        if meta is None:
            logging.warning("Could not parse magnification from %s – skipping.", tile_name)
            continue

        mask_name = tile_name + "_mask.png"
        mask_path = seg_root / slide_name / mask_name
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
                **meta,                       # magnification, x, y
                "label_id": lbl_id,
                "polygon": poly["polygon"],
                "pred_class": cls_cell["pred_class"],
                "cluster_id": cls_cell.get("cluster_id"),
                "cluster_confidence": cls_cell.get("cluster_confidence"),
                "bbox": cls_cell["bbox"],
            })

        final_ann.append({
            "tile_path": tile_path,
            **meta,
            "cells": cell_records,
        })

    # --------------- write output --------------- #
    out_name = f"v1_{args.param_string}_annotations.json" if args.param_string else "final_annotations.json"
    out_file = out_root / out_name
    out_file.write_text(json.dumps(final_ann, indent=2))
    logging.info("Annotations written → %s", out_file)

# ──────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    main()
