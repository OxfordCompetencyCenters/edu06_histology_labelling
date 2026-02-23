"""
Azure ML Parallel Entry Script for Cell Segmentation (Slide-Level).

For "Slide-Level Parallelism", the input mini-batch is a list of TRIGGER FILES
(one per slide). The tiles are read from a SIDE INPUT folder using the slide ID.

Input structure (mini-batch):
    manifest_path/slide_id_A
    
Side input (images_dir):
    images_dir/slide_id_A/tile1.png
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path
from typing import List, Optional
from cellpose import models

import numpy as np
from PIL import Image

from utils import log_and_print, log_exception, mark_slide_done, is_slide_done, log_checkpoint_status, write_json_atomic, build_slide_filter_set, should_process_slide

# Global state initialized in init()
_args = None
_model = None
_output_base = None
_tiles_base = None
_manifest_base = None
_slide_filter = None


def extract_slide_id_from_filename(filename: str) -> Optional[str]:
    match = re.match(r'^(.+?)__MAG_', filename)
    if match:
        return match.group(1)
    return None


def init():
    """
    Initialize model before processing mini-batches.
    """
    global _args, _model, _output_base, _tiles_base, _manifest_base, _slide_filter
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True,
                        help="Base output directory for segmentation results")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Side input directory containing tiles (mounted)")
    parser.add_argument("--output_manifest", type=str, required=True,
                        help="Output directory for trigger files (one per slide)")
    parser.add_argument("--pretrained_model", type=str, default="cpsam",
                        help="Cellpose pretrained model")

    parser.add_argument("--flow_threshold", type=float, default=0.4)
    parser.add_argument("--cellprob_threshold", type=float, default=0.0)
    parser.add_argument("--segment_use_gpu", action="store_true", default=False)
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--resample", action="store_true", default=True)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--tile_batch_size", type=int, default=1)
    parser.add_argument("--slide_filter", type=str, default=None,
                        help="Comma-separated list of slide names to process (others are skipped)")
    
    _args, _ = parser.parse_known_args()
    _output_base = Path(_args.output_path)
    _output_base.mkdir(parents=True, exist_ok=True)
    
    _tiles_base = Path(_args.input_path) # Side input mounted folder
    
    _manifest_base = Path(_args.output_manifest)
    _manifest_base.mkdir(parents=True, exist_ok=True)
    
    _slide_filter = build_slide_filter_set(_args.slide_filter)
    
    log_and_print(f"Loading Cellpose model: {_args.pretrained_model}")
    log_and_print(f"  GPU: {_args.segment_use_gpu}")
    
    try:
        _model = models.CellposeModel(
            pretrained_model=_args.pretrained_model,
            gpu=_args.segment_use_gpu
        )
        log_and_print(f"Successfully loaded model: {_args.pretrained_model}")
    except Exception as e:
        log_exception(f"Failed to load model '{_args.pretrained_model}'", e)
    
    log_and_print("Parallel segmentation (Slide-Level) initialized")
    log_and_print(f"  Input Source: {_tiles_base}")
    log_and_print(f"  Output path: {_output_base}")
    if _slide_filter:
        log_and_print(f"  Slide filter: {_slide_filter}")


def save_segmentation_result(masks: np.ndarray, tile_name: str, output_dir: Path) -> dict:
    """Save mask and extract bounding boxes for a single tile."""
    mask_img = Image.fromarray(masks.astype(np.uint16))
    mask_path = output_dir / f"{tile_name}_mask.png"
    mask_img.save(mask_path)
    
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
    
    bbox_path = output_dir / f"{tile_name}_bboxes.json"
    write_json_atomic(bbox_path, bboxes)
    
    return {"mask_path": str(mask_path), "bbox_path": str(bbox_path), "num_cells": len(bboxes)}


def segment_batch(img_paths: List[Path], output_dir: Path) -> List[dict]:
    """Segment a batch of tiles in one model call."""
    normalize = _args.normalize and not _args.no_normalize
    images = [np.array(Image.open(p)) for p in img_paths]
    
    masks_list, flows_list, diams = _model.eval(
        images,
        flow_threshold=_args.flow_threshold,
        cellprob_threshold=_args.cellprob_threshold,
        diameter=_args.diameter,
        resample=_args.resample,
        normalize=normalize
    )
    
    results = []
    for img_path, masks in zip(img_paths, masks_list):
        results.append(save_segmentation_result(masks, img_path.stem, output_dir))
    return results


def run(mini_batch: List[str]) -> List[str]:
    """
    Process a mini-batch of SLIDE TRIGGER FILES.
    """
    results = []
    
    for trigger_file in mini_batch:
        trigger_path = Path(trigger_file)
        slide_id = trigger_path.name
        
        # Apply slide filter
        if not should_process_slide(slide_id, _slide_filter):
            log_and_print(f"Skipping (slide filter): {slide_id}")
            results.append(f"SKIP:{slide_id}:filtered")
            continue
        
        slide_src_dir = _tiles_base / slide_id
        slide_dst_dir = _output_base / slide_id
        slide_dst_dir.mkdir(parents=True, exist_ok=True)
        
        log_and_print(f"Processing slide: {slide_id}")

        # Skip slides already completed (checkpoint resume across preempted jobs)
        if is_slide_done(_output_base, slide_id):
            log_and_print(f"Skipping (already done): {slide_id}")
            results.append(f"SKIP:{slide_id}:already_done")
            # Still create trigger file so downstream steps can see it
            trigger_file = _manifest_base / slide_id
            trigger_file.touch()
            continue

        if not slide_src_dir.exists():
            log_and_print(f"Slide folder not found: {slide_src_dir}", level="warning")
            results.append(f"MISSING:{slide_id}")
            continue
            
        # Enumerate tiles in slide folder
        tile_files = list(slide_src_dir.glob("*.png")) + \
                     list(slide_src_dir.glob("*.jpg")) + \
                     list(slide_src_dir.glob("*.tif"))
        
        valid_tiles = [f for f in tile_files if '_mask.' not in f.name and '_bboxes.' not in f.name]
        
        count = 0
        batch_size = _args.tile_batch_size
        
        for i in range(0, len(valid_tiles), batch_size):
            batch = valid_tiles[i:i + batch_size]
            try:
                batch_results = segment_batch(batch, slide_dst_dir)
                for r in batch_results:
                    count += r['num_cells']
            except Exception as e:
                log_exception(f"Error segmenting batch starting at {batch[0]}", e)
                for file_path in batch:
                    try:
                        tile_results = segment_batch([file_path], slide_dst_dir)
                        count += tile_results[0]['num_cells']
                    except Exception as inner_e:
                        log_exception(f"Error segmenting {file_path}", inner_e)
        
        result = f"OK:{slide_id}:total_cells={count}"
        results.append(result)
        log_and_print(result)
        mark_slide_done(_output_base, slide_id, result)
        
        # Create trigger file for downstream stages (cluster, classify)
        trigger_file = _manifest_base / slide_id
        trigger_file.touch()
        log_and_print(f"Created trigger file: {trigger_file}")
    
    return results


def shutdown():
    """Cleanup after all mini-batches processed."""
    global _model
    _model = None
    log_and_print("Parallel segmentation shutdown complete")


# --------------------------------------------------------------------------- #
# Sequential (single-node) entry point
# --------------------------------------------------------------------------- #
def run_all() -> None:
    """
    Process ALL slide folders sequentially.
    Used when running as a command() job (max_nodes=1) for easier debugging.
    The input tiles base (_tiles_base) is set by init().
    """
    init()
    # Find all slide subdirectories in the tiles input folder
    slide_dirs = [d for d in _tiles_base.iterdir() if d.is_dir()]
    
    # Apply slide filter
    if _slide_filter:
        before = len(slide_dirs)
        slide_dirs = [d for d in slide_dirs if should_process_slide(d.name, _slide_filter)]
        log_and_print(f"[Sequential] Slide filter matched {len(slide_dirs)}/{before} slide folders")
    
    log_and_print(f"[Sequential] Found {len(slide_dirs)} slide folders in {_tiles_base}")

    # --- checkpoint resume ---
    already_done = sum(1 for d in slide_dirs if is_slide_done(_output_base, d.name))
    log_checkpoint_status("Segmentation", len(slide_dirs), already_done)

    all_results = []
    for slide_dir in sorted(slide_dirs):
        slide_id = slide_dir.name
        if is_slide_done(_output_base, slide_id):
            log_and_print(f"[Sequential] Skipping (already done): {slide_id}")
            continue
        log_and_print(f"[Sequential] Processing slide: {slide_id}")
        # Create a fake trigger file path (run() only uses the filename)
        fake_trigger = str(_tiles_base / slide_id)
        results = run([fake_trigger])
        all_results.extend(results)
        for r in results:
            log_and_print(f"  -> {r}")
            if r.startswith("OK:"):
                mark_slide_done(_output_base, slide_id, r)

    log_and_print(f"[Sequential] Segmentation complete. {len(all_results)} slides processed.")
    shutdown()


if __name__ == "__main__":
    # All args are parsed by init() via parse_known_args
    run_all()
