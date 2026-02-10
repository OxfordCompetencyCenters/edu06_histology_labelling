"""
Shared utility functions for Azure ML parallel processing scripts.
"""
from __future__ import annotations
import glob
import logging
import os
import sys
import traceback
from collections import defaultdict
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Dual logging: print() + logging for Azure ML parallel job visibility
# --------------------------------------------------------------------------- #
# Azure ML parallel jobs scatter output across multiple log files:
#   - ~/logs/user/error/  (stderr / logging)
#   - ~/logs/user/stdout/ (print / stdout)
#   - 70_driver_log.txt   (driver stdout)
# Using both print() and logging increases the odds of seeing messages.
# --------------------------------------------------------------------------- #

def log_and_print(msg: str, level: str = "info") -> None:
    """Log a message via both logging and print (flushed) for maximum Azure ML visibility."""
    logger = logging.getLogger()
    if level == "debug":
        logger.debug(msg)
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    elif level == "critical":
        logger.critical(msg)
    else:
        logger.info(msg)
    # Also print to stdout so it appears in 70_driver_log / stdout logs
    print(f"[{level.upper()}] {msg}", flush=True)


def log_exception(msg: str, exc: Exception | None = None) -> None:
    """Log an error with full traceback via both logging and print."""
    tb = traceback.format_exc()
    full_msg = f"{msg}\n  Exception: {exc}\n  Traceback:\n{tb}" if exc else f"{msg}\n{tb}"
    logging.error(full_msg)
    print(f"[ERROR] {full_msg}", file=sys.stderr, flush=True)


def build_slide_to_bbox_files_mapping(segmentation_path: str) -> Dict[str, List[str]]:
    """
    Build a mapping from slide_id to list of bbox files.
    
    Segmentation outputs are organized in slide subfolders:
        segmentation_path/
          slideA/
            slideA__MAG_1d000__X_0__Y_0__IDX_000001_bboxes.json
            ...
          slideB/
            slideB__MAG_1d000__X_0__Y_0__IDX_000001_bboxes.json
            ...
    
    Args:
        segmentation_path: Base path to segmentation results
        
    Returns:
        Dictionary mapping slide_id to list of bbox file paths
    """
    mapping = defaultdict(list)
    
    # Find all bbox files in slide subfolders
    bbox_files = glob.glob(os.path.join(segmentation_path, "*", "*_bboxes.json"))
    
    for bbox_file in bbox_files:
        # Slide ID is the parent folder name
        slide_id = os.path.basename(os.path.dirname(bbox_file))
        mapping[slide_id].append(bbox_file)
    
    logging.info(f"Built slide mapping: {len(mapping)} slides, {len(bbox_files)} bbox files total")
    return dict(mapping)


def get_tile_path_from_bbox_file(bbox_file: str, prepped_tiles_path: str) -> tuple:
    """
    Helper function to derive tile/slide info from bbox_file path (per-slide subfolder structure).
    
    Args:
        bbox_file: Path to the bbox JSON file
        prepped_tiles_path: Base path to prepped tiles
        
    Returns:
        Tuple of (tile_path, tile_base_name, slide_id) or (None, None, None) on error
    """
    try:
        bbox_basename = os.path.basename(bbox_file)
        tile_base_name = bbox_basename.replace("_bboxes.json", "")
        tile_file_name = tile_base_name + ".png"
        
        # Extract slide_id from the parent folder name (bbox files are in per-slide subfolders)
        slide_id = os.path.basename(os.path.dirname(bbox_file))
        
        # Tiles are in per-slide subfolders: prepped_tiles_path/slide_id/tile.png
        tile_path = os.path.join(prepped_tiles_path, slide_id, tile_file_name)
        
        return tile_path, tile_base_name, slide_id
    except Exception as e:
        logging.error(f"Error parsing info from bbox_file '{bbox_file}': {e}")
        return None, None, None
