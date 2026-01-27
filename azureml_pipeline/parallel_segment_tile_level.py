"""
Azure ML Parallel Entry Script for Cell Segmentation (Tile-Level Parallelism).

This script processes INDIVIDUAL TILE FILES in parallel.
Input: FLAT tile structure where mini-batch contains tile file paths.

Output: Slide subfolders with segmentation results (for downstream slide-level steps).
    output_path/
      slideA/
        slideA__MAG_1d000__X_0__Y_0__IDX_000001_mask.png
        slideA__MAG_1d000__X_0__Y_0__IDX_000001_bboxes.json
      slideB/
        slideB__MAG_1d000__X_0__Y_0__IDX_000002_mask.png
        ...
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


# Global state
_args = None
_model = None
_output_base = None


def extract_slide_id_from_filename(filename: str) -> Optional[str]:
    """Extract slide_id from tile filename: {slide_id}__MAG_..."""
    match = re.match(r'^(.+?)__MAG_', filename)
    if match:
        return match.group(1)
    return None


def init():
    """Initialize Cellpose model."""
    global _args, _model, _output_base
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, default="cpsam")
    parser.add_argument("--flow_threshold", type=float, default=0.4)
    parser.add_argument("--cellprob_threshold", type=float, default=0.0)
    parser.add_argument("--segment_use_gpu", action="store_true", default=False)
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--resample", action="store_true", default=True)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--do_3D", action="store_true")
    parser.add_argument("--stitch_threshold", type=float, default=0.0)
    
    _args, _ = parser.parse_known_args()
    _output_base = Path(_args.output_path)
    _output_base.mkdir(parents=True, exist_ok=True)
    
    # Load Cellpose model once
    from cellpose import models
    
    logging.info(f"Loading Cellpose model: {_args.pretrained_model}")
    logging.info(f"  GPU: {_args.segment_use_gpu}")
    
    try:
        _model = models.CellposeModel(
            pretrained_model=_args.pretrained_model,
            gpu=_args.segment_use_gpu
        )
        logging.info(f"Successfully loaded model: {_args.pretrained_model}")
    except Exception as e:
        logging.error(f"Failed to load model '{_args.pretrained_model}': {e}")
        logging.info("Falling back to cyto2 model")
        _model = models.CellposeModel(pretrained_model="cyto2", gpu=_args.segment_use_gpu)
    
    logging.info("Parallel segmentation (Tile-Level) initialized")
    logging.info(f"  Output path: {_output_base}")


def segment_single_tile(img_path: Path) -> dict:
    """Run Cellpose segmentation on a single tile."""
    tile_name = img_path.stem
    
    # Extract slide_id to create slide subfolder (for downstream slide-level steps)
    slide_id = extract_slide_id_from_filename(img_path.name)
    if not slide_id:
        slide_id = "unknown_slide"
    
    # Output to slide subfolder
    output_dir = _output_base / slide_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    img = np.array(Image.open(img_path))
    
    # Determine normalization
    normalize = _args.normalize and not _args.no_normalize
    
    # Run segmentation
    masks, flows, diams = _model.eval(
        img,
        flow_threshold=_args.flow_threshold,
        cellprob_threshold=_args.cellprob_threshold,
        diameter=_args.diameter,
        resample=_args.resample,
        normalize=normalize,
        do_3D=_args.do_3D,
        stitch_threshold=_args.stitch_threshold
    )
    
    # Save mask
    mask_img = Image.fromarray(masks.astype(np.uint16))
    mask_filename = f"{tile_name}_mask.png"
    mask_path = output_dir / mask_filename
    mask_img.save(mask_path)
    
    # Extract and save bounding boxes
    bboxes = []
    unique_labels = np.unique(masks)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        coords = np.argwhere(masks == lbl)
        if coords.size == 0:
            continue
        min_row, max_row = coords[:, 0].min(), coords[:, 0].max()
        min_col, max_col = coords[:, 1].min(), coords[:, 1].max()
        
        bboxes.append({
            "label_id": int(lbl),
            "bbox": [int(min_col), int(min_row), int(max_col), int(max_row)]
        })
    
    bbox_filename = f"{tile_name}_bboxes.json"
    bbox_path = output_dir / bbox_filename
    with open(bbox_path, "w") as f:
        json.dump(bboxes, f, indent=2)
    
    return {
        "slide_id": slide_id,
        "mask_path": str(mask_path),
        "bbox_path": str(bbox_path),
        "num_cells": len(bboxes)
    }


def run(mini_batch: List[str]) -> List[str]:
    """
    Process a mini-batch of INDIVIDUAL TILE FILES.
    """
    results = []
    
    for tile_file in mini_batch:
        tile_path = Path(str(tile_file).strip())
        if not tile_path.exists():
            continue
        
        tile_name = tile_path.name
        
        # Skip thumbnails, masks, stats, bboxes
        if any(skip in tile_name for skip in ['__THUMBNAIL', '_mask.', '_bboxes.', '_filter_stats']):
            results.append(f"SKIP:{tile_name}:not_a_tile")
            continue
        
        # Skip non-image files
        if not tile_name.lower().endswith(('.png', '.jpg', '.tif')):
            continue
        
        try:
            result_info = segment_single_tile(tile_path)
            result = f"OK:{tile_name}:cells={result_info['num_cells']}"
            results.append(result)
            
        except Exception as e:
            logging.error(f"Error segmenting {tile_path}: {e}")
            results.append(f"ERROR:{tile_name}:{str(e)}")
    
    return results


def shutdown():
    """Cleanup."""
    global _model
    _model = None
    logging.info("Parallel segmentation (Tile-Level) shutdown complete")
