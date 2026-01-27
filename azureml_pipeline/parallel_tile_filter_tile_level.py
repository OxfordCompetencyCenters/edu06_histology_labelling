"""
Azure ML Parallel Entry Script for Tile Filtering (Tile-Level Parallelism).

This script processes INDIVIDUAL TILE FILES in parallel.
Input: FLAT tile structure where mini-batch contains tile file paths.

Input structure (mini-batch items):
    input_path/slideA__MAG_1d000__X_0__Y_0__IDX_000001.png
    input_path/slideB__MAG_1d000__X_512__Y_0__IDX_000002.png

Output: FLAT structure with filtered tiles + slide subfolders for downstream.
    output_path/slideA__MAG_1d000__X_0__Y_0__IDX_000001.png
    output_path/slideB__MAG_1d000__X_512__Y_0__IDX_000002.png
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image


# Global state
_args = None
_output_base = None


def extract_slide_id_from_filename(filename: str) -> Optional[str]:
    """Extract slide_id from tile filename: {slide_id}__MAG_..."""
    match = re.match(r'^(.+?)__MAG_', filename)
    if match:
        return match.group(1)
    return None


def init():
    """Initialize resources."""
    global _args, _output_base
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--min_edge_density", type=float, default=0.02)
    parser.add_argument("--max_bright_ratio", type=float, default=0.8)
    parser.add_argument("--max_dark_ratio", type=float, default=0.8)
    parser.add_argument("--min_std_intensity", type=float, default=10.0)
    parser.add_argument("--min_laplacian_var", type=float, default=50.0)
    parser.add_argument("--min_color_variance", type=float, default=5.0)
    parser.add_argument("--save_stats", action="store_true", default=True)
    parser.add_argument("--output_flat", action="store_true", default=True,
                        help="Output filtered tiles to flat structure")
    
    _args, _ = parser.parse_known_args()
    _output_base = Path(_args.output_path)
    _output_base.mkdir(parents=True, exist_ok=True)
    
    logging.info("Parallel tile filter (Tile-Level) initialized")
    logging.info(f"  Output Path: {_output_base}")
    logging.info(f"  Flat output: {_args.output_flat}")


def calculate_tile_stats(image: np.ndarray) -> dict:
    """Calculate comprehensive statistics for a tile."""
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
    """Determine if a tile is worth processing."""
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
    Process a mini-batch of INDIVIDUAL TILE FILES.
    
    Each item in mini_batch is a path to a tile image file.
    """
    results = []
    
    for tile_file in mini_batch:
        tile_path = Path(str(tile_file).strip())
        if not tile_path.exists():
            continue
        
        tile_name = tile_path.name
        
        # Skip thumbnails and stats files
        if '__THUMBNAIL' in tile_name or '_filter_stats' in tile_name:
            results.append(f"SKIP:{tile_name}:thumbnail_or_stats")
            continue
        
        # Skip non-image files
        if not tile_name.lower().endswith(('.png', '.jpg', '.tif')):
            continue
        
        try:
            img = np.array(Image.open(tile_path))
            is_useful, stats = is_tile_useful(img)
            
            if is_useful:
                # Output to flat structure
                out_path = _output_base / tile_name
                shutil.copy2(tile_path, out_path)
                
                if _args.save_stats:
                    stats_path = _output_base / f"{tile_path.stem}_filter_stats.json"
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=2)
                
                results.append(f"KEEP:{tile_name}")
            else:
                results.append(f"REJECT:{tile_name}:{','.join(stats['rejection_reasons'][:2])}")
                
        except Exception as e:
            logging.error(f"Error processing {tile_path}: {e}")
            results.append(f"ERROR:{tile_name}:{str(e)}")
    
    return results


def shutdown():
    """Cleanup."""
    logging.info("Parallel tile filter (Tile-Level) shutdown complete")
