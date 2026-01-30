"""
Azure ML Parallel Entry Script for Data Preparation (WSI Tiling).

This script implements the init()/run(mini_batch) interface required by
Azure ML parallel_run_function. Each mini-batch contains a list of WSI
file paths (e.g., .ndpi) to tile.

This enables SLIDE-LEVEL parallelization: different nodes process different
whole slide images simultaneously.

Usage in pipeline:
    parallel_run_function(
        task=RunFunction(entry_script="parallel_data_prep.py", ...),
        input_data="${{inputs.input_wsi}}",  # Folder with WSI files
        mini_batch_size="1",  # WSI files per mini-batch (1 is common)
        ...
    )
"""
from __future__ import annotations
import argparse
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Generator, List, Tuple

import openslide
from PIL import Image


# Global state initialized in init()
_args = None
_output_base = None

def init():
    """
    Initialize resources before processing mini-batches.
    Called once per worker process.
    """
    global _args, _output_base
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True,
                        help="Base output directory for tiled images")
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Output directory for slide trigger files (manifest)")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Output tile edge size (default 512)")
    parser.add_argument("--magnifications", type=str, default="1.0",
                        help="Comma-separated magnification factors, e.g., '1.0,0.9,0.8'")
    parser.add_argument("--num_tiles", type=int, default=None,
                        help="Target number of tiles per magnification (grid is thinned)")
    parser.add_argument("--replace_percent_in_names", action="store_true",
                        help="Replace '%' characters in slide names with '_pct_'")
    parser.add_argument("--flat_output", action="store_true",
                        help="Output tiles to flat structure (all at root level)")
    
    _args, _ = parser.parse_known_args()
    _output_base = Path(_args.output_path)
    _output_base.mkdir(parents=True, exist_ok=True)
    
    logging.info("Parallel data prep initialized")
    logging.info(f"  Output path: {_output_base}")
    logging.info(f"  Tile size: {_args.tile_size}")
    logging.info(f"  Magnifications: {_args.magnifications}")
    logging.info(f"  Flat output: {_args.flat_output}")


def parse_magnifications(mag_str: str) -> List[float]:
    """Parse magnification string into sorted list of floats."""
    mag_str = mag_str.replace('"', '').replace("'", "")
    factors = sorted({float(s) for s in mag_str.split(",") if s.strip()}, reverse=True)
    if not factors or any(f <= 0.0 for f in factors):
        raise ValueError("Magnifications must be positive floats > 0.")
    return factors


def generate_slide_id(slide_name: str, replace_percent: bool = False) -> str:
    """Generate a short, filesystem-safe ID from slide name."""
    if replace_percent:
        slide_name = slide_name.replace('%', '_pct_')
    
    clean_name = "".join(c for c in slide_name if c.isalnum() or c in " -_")
    short_name = clean_name[:20].strip()
    hash_suffix = hashlib.md5(slide_name.encode()).hexdigest()[:8]
    return f"{short_name}_{hash_suffix}".replace(" ", "_")


def format_magnification_tag(mag: float) -> str:
    """Convert magnification to readable format: 1.0 -> 1d00, 0.7 -> 0d70."""
    int_part = int(mag)
    frac_part = int((mag - int_part) * 100)
    return f"{int_part}d{frac_part:02d}"


def create_tile_filename(slide_id: str, mag: float, x: int, y: int, idx: int) -> str:
    """Create standardized tile filename using delimiter-based format."""
    mag_tag = format_magnification_tag(mag)
    return f"{slide_id}__MAG_{mag_tag}__X_{x}__Y_{y}__IDX_{idx:06d}.png"


def compute_step_multiplier(total_tiles: int, requested: int | None) -> int:
    """Compute stride multiplier based on requested tile count."""
    if requested is None or requested >= total_tiles:
        return 1
    return math.ceil(math.sqrt(total_tiles / requested))


def iter_grid_positions(w: int, h: int, edge: int, stride: int) -> Generator[Tuple[int, int], None, None]:
    """Iterate over grid positions for tiling."""
    max_x, max_y = w - edge, h - edge
    for y in range(0, max_y + 1, stride):
        for x in range(0, max_x + 1, stride):
            yield x, y


def tile_slide(slide_path: Path) -> dict:
    """
    Tile a single WSI file.
    
    Returns dict with:
        - slide_name: name of the slide
        - slide_id: generated filesystem-safe ID
        - num_tiles: total number of tiles created
        - output_dir: path to output directory for this slide
    """
    magnifications = parse_magnifications(_args.magnifications)
    tile_size = _args.tile_size
    num_tiles = _args.num_tiles
    replace_percent = _args.replace_percent_in_names
    
    slide_name = slide_path.stem
    slide_id = generate_slide_id(slide_name, replace_percent)
    
    # Configure output structure based on flag
    # If flat_output: all tiles at root level (enables tile-level parallelism downstream)
    # Else: tiles in subfolders (enables slide-level parallelism downstream)
    if _args.flat_output:
        out_dir = _output_base
    else:
        out_dir = _output_base / slide_id
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create trigger file in manifest output if requested
    if hasattr(_args, 'manifest_path') and _args.manifest_path:
        manifest_base = Path(_args.manifest_path)
        manifest_base.mkdir(parents=True, exist_ok=True)
        # Create empty trigger file named {slide_id}
        trigger_path = manifest_base / slide_id
        trigger_path.touch()
    
    logging.info(f"Opening slide: {slide_path}")
    logging.info(f"Slide ID: {slide_id}")
    
    slide = openslide.OpenSlide(str(slide_path))
    base_w, base_h = slide.dimensions
    
    tile_idx = 0
    
    for mag in magnifications:
        if mag <= 1.0:
            src_edge = int(round(tile_size / mag))
            if base_w < src_edge or base_h < src_edge:
                logging.info(f"  Skipping mag {mag:.3f}: slide < {src_edge}px at this scale")
                continue
        else:
            src_edge = tile_size
            if base_w < src_edge or base_h < src_edge:
                logging.info(f"  Skipping mag {mag:.3f}: slide < {src_edge}px")
                continue
        
        tiles_x = (base_w - src_edge) // src_edge + 1
        tiles_y = (base_h - src_edge) // src_edge + 1
        stride = src_edge * compute_step_multiplier(tiles_x * tiles_y, num_tiles)
        
        est_tiles = ((base_w - src_edge) // stride + 1) * ((base_h - src_edge) // stride + 1)
        logging.info(f"  Processing mag {mag:.3f}: ~{est_tiles} tiles")
        
        for x, y in iter_grid_positions(base_w, base_h, src_edge, stride):
            region = slide.read_region((x, y), 0, (src_edge, src_edge)).convert("RGB")
            
            if mag <= 1.0:
                if src_edge != tile_size:
                    region = region.resize((tile_size, tile_size), Image.LANCZOS)
            else:
                enlarged_size = int(tile_size * mag)
                region = region.resize((enlarged_size, enlarged_size), Image.LANCZOS)
                left = top = (enlarged_size - tile_size) // 2
                region = region.crop((left, top, left + tile_size, top + tile_size))
            
            out_name = create_tile_filename(slide_id, mag, x, y, tile_idx)
            region.save(out_dir / out_name)
            tile_idx += 1
    
    # Create thumbnail
    thumb_lvl = slide.level_count - 1
    thumb = slide.read_region((0, 0), thumb_lvl, slide.level_dimensions[thumb_lvl]) \
                 .convert("RGB").resize((tile_size, tile_size), Image.BICUBIC)
    
    # Save thumbnail (with prefix if flat, or in folder if nested)
    if _args.flat_output:
        thumb.save(out_dir / f"{slide_id}__THUMBNAIL.png")
    else:
        thumb.save(out_dir / f"{slide_id}__THUMBNAIL.png")
    
    slide.close()
    
    return {
        "slide_name": slide_name,
        "slide_id": slide_id,
        "num_tiles": tile_idx,
        "output_dir": str(out_dir)
    }


def run(mini_batch: List[str]) -> List[str]:
    """
    Process a mini-batch of WSI file paths.
    
    Args:
        mini_batch: List of absolute file paths to WSI files (.ndpi)
        
    Returns:
        List of result strings (one per processed file) - required by Azure ML
    """
    results = []
    
    for file_path in mini_batch:
        file_path = str(file_path).strip()
        if not file_path:
            continue
        
        # Skip non-WSI files
        if not file_path.lower().endswith(('.ndpi')):
            logging.debug(f"Skipping non-WSI file: {file_path}")
            continue
        try:
            path_obj = Path(file_path)
            result_info = tile_slide(path_obj)
            result = f"OK:{file_path}:tiles={result_info['num_tiles']}"
            logging.info(f"Finished {result_info['slide_name']} (ID: {result_info['slide_id']}): {result_info['num_tiles']} tiles")
            results.append(result)
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            results.append(f"ERROR:{file_path}:{str(e)}")
    
    return results


def shutdown():
    """Cleanup after all mini-batches processed. Optional."""
    logging.info("Parallel data prep shutdown complete")
