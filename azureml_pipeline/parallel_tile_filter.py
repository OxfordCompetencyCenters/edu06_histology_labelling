"""
Azure ML Parallel Entry Script for Tile Filtering (Slide-Level).

This script implements the init()/run(mini_batch) interface required by
Azure ML parallel_run_function.

For "Slide-Level Parallelism", the input mini-batch is a list of TRIGGER FILES
(one per slide) representing the slide ID. The actual tiles are read from a
SIDE INPUT folder (mounted read-only).

Input structure (mini-batch items):
    manifest_path/slide_id_A
    manifest_path/slide_id_B

Side input structure (images_dir):
    images_dir/slide_id_A/tile1.png
    images_dir/slide_id_A/tile2.png
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image


# Global state
_args = None
_output_base = None
_tiles_base = None
_manifest_out_base = None


def init():
    """
    Initialize resources.
    """
    global _args, _output_base, _tiles_base, _manifest_out_base
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser()
    # output_path: where filtered tiles go
    parser.add_argument("--output_path", type=str, required=True)
    # input_path: side input where source tiles are located (mounted folder)
    parser.add_argument("--input_path", type=str, required=True)
    # output_manifest_path: where to write trigger files for the next step
    parser.add_argument("--output_manifest_path", type=str, default=None)
    
    parser.add_argument("--min_edge_density", type=float, default=0.02)
    parser.add_argument("--max_bright_ratio", type=float, default=0.8)
    parser.add_argument("--max_dark_ratio", type=float, default=0.8)
    parser.add_argument("--min_std_intensity", type=float, default=10.0)
    parser.add_argument("--min_laplacian_var", type=float, default=50.0)
    parser.add_argument("--min_color_variance", type=float, default=5.0)
    parser.add_argument("--save_stats", action="store_true", default=True)
    
    _args, _ = parser.parse_known_args()
    _output_base = Path(_args.output_path)
    _output_base.mkdir(parents=True, exist_ok=True)
    
    _tiles_base = Path(_args.input_path)  # Side input mounted folder
    
    if _args.output_manifest_path:
        _manifest_out_base = Path(_args.output_manifest_path)
        _manifest_out_base.mkdir(parents=True, exist_ok=True)
    
    logging.info("Parallel tile filter (Slide-Level) initialized")
    logging.info(f"  Input Source: {_tiles_base}")
    logging.info(f"  Output Path: {_output_base}")
    logging.info(f"  Manifest Output: {_manifest_out_base}")


def calculate_tile_stats(image: np.ndarray) -> dict:
    """Calculate comprehensive statistics for a tile to determine quality."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    bright_pixels = np.sum(gray > 240) / gray.size
    dark_pixels = np.sum(gray < 15) / gray.size
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    color_variance = 0
    if len(image.shape) == 3:
        color_variance = np.var([np.mean(image[:, :, i]) for i in range(3)])
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        'mean_intensity': float(mean_intensity),
        'std_intensity': float(std_intensity),
        'bright_pixels_ratio': float(bright_pixels),
        'dark_pixels_ratio': float(dark_pixels),
        'edge_density': float(edge_density),
        'color_variance': float(color_variance),
        'laplacian_var': float(laplacian_var)
    }


def is_tile_useful(image: np.ndarray) -> tuple[bool, dict]:
    """Determine if a tile is worth processing based on quality criteria."""
    stats = calculate_tile_stats(image)
    
    reasons = []
    
    if stats['edge_density'] < _args.min_edge_density:
        reasons.append(f"low_edge_density ({stats['edge_density']:.4f} < {_args.min_edge_density})")
    
    if stats['bright_pixels_ratio'] > _args.max_bright_ratio:
        reasons.append(f"too_bright ({stats['bright_pixels_ratio']:.2f} > {_args.max_bright_ratio})")
    
    if stats['dark_pixels_ratio'] > _args.max_dark_ratio:
        reasons.append(f"too_dark ({stats['dark_pixels_ratio']:.2f} > {_args.max_dark_ratio})")
    
    if stats['std_intensity'] < _args.min_std_intensity:
        reasons.append(f"low_contrast ({stats['std_intensity']:.2f} < {_args.min_std_intensity})")
    
    if stats['laplacian_var'] < _args.min_laplacian_var:
        reasons.append(f"blurry ({stats['laplacian_var']:.2f} < {_args.min_laplacian_var})")
    
    if len(image.shape) == 3 and stats['color_variance'] < _args.min_color_variance:
        reasons.append(f"low_color_var ({stats['color_variance']:.2f} < {_args.min_color_variance})")
    
    is_useful = len(reasons) == 0
    stats['rejection_reasons'] = reasons
    
    return is_useful, stats


def run(mini_batch: List[str]) -> List[str]:
    """
    Process a mini-batch of SLIDE TRIGGER FILES.
    
    Each item in mini_batch is a path to an empty file named `{slide_id}`.
    We iterate over all tiles in `input_path/{slide_id}` and process them.
    """
    results = []
    
    for trigger_file in mini_batch:
        trigger_path = Path(trigger_file)
        slide_id = trigger_path.name  # The file name is the slide ID
        
        # Source directory for this slide (from side input)
        # Note: data_prep now creates subfolders per slide
        slide_src_dir = _tiles_base / slide_id
        
        # Destination directory for this slide
        slide_dst_dir = _output_base / slide_id
        slide_dst_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Processing slide: {slide_id}")
        if not slide_src_dir.exists():
            logging.warning(f"Slide folder not found: {slide_src_dir}")
            results.append(f"MISSING:{slide_id}")
            continue
        
        # List all image files in the slide folder
        # We manually enumerate here, which is fast because it's a single directory (per slide)
        # instead of the root folder (all tiles)
        tile_files = list(slide_src_dir.glob("*.png")) + \
                     list(slide_src_dir.glob("*.jpg")) + \
                     list(slide_src_dir.glob("*.tif"))
                     
        processed_count = 0
        kept_count = 0
        
        for file_path in tile_files:
            tile_name = file_path.name
            # Skip thumbnails/stats
            if '__THUMBNAIL' in tile_name or '_filter_stats' in tile_name:
                continue
                
            try:
                processed_count += 1
                img = np.array(Image.open(file_path))
                is_useful, stats = is_tile_useful(img)
                
                if is_useful:
                    out_path = slide_dst_dir / tile_name
                    shutil.copy2(file_path, out_path)
                    
                    if _args.save_stats:
                        stats_path = slide_dst_dir / f"{file_path.stem}_filter_stats.json"
                        with open(stats_path, 'w') as f:
                            json.dump(stats, f, indent=2)
                    kept_count += 1
                    
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
        
        # Create output trigger file if configured
        if _manifest_out_base:
            ( _manifest_out_base / slide_id ).touch()
            
        result = f"OK:{slide_id}:processed={processed_count}:kept={kept_count}"
        results.append(result)
        logging.info(result)
    
    return results


def shutdown():
    """Cleanup after all mini-batches processed. Optional."""
    logging.info("Parallel tile filter shutdown complete")
