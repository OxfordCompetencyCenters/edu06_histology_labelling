"""
Shared utility functions for Azure ML parallel processing scripts.
"""
from __future__ import annotations
import glob
import hashlib
import json
import logging
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set


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


# --------------------------------------------------------------------------- #
# Checkpoint helpers for sequential (single-node) jobs
# --------------------------------------------------------------------------- #
# When running on low-priority / spot VMs, the node can be preempted at any
# time.  These helpers let each stage's run_all() skip slides whose output
# already exists, so progress made before a preemption is preserved.
# --------------------------------------------------------------------------- #

_CHECKPOINT_FILENAME = "_checkpoint_done.json"


def _fsync_file(filepath: Path) -> None:
    """Force-flush a file to the underlying storage (critical for FUSE/blobfuse mounts).

    On Azure ML ``rw_mount`` outputs the filesystem is backed by blobfuse.
    Without an explicit ``fsync`` the data may sit in the kernel page-cache
    and be lost if the VM is preempted before the OS flushes it.
    """
    try:
        fd = os.open(str(filepath), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Best-effort: if the file disappeared or the FS doesn't support
        # fsync (rare), we still want the caller to continue.
        pass


def write_json_atomic(filepath: Path, data, indent: int = 2) -> None:
    """Write JSON to *filepath* with fsync to guarantee durability on FUSE mounts.

    This should be used for ALL important output files (results, checkpoints)
    so that data is persisted to blob storage even under sudden preemption.
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent)
        f.flush()
        os.fsync(f.fileno())


def mark_slide_done(output_dir: Path, slide_id: str, result: str) -> None:
    """Write a small JSON checkpoint file indicating a slide was fully processed."""
    slide_out = output_dir / slide_id
    slide_out.mkdir(parents=True, exist_ok=True)
    ckpt = slide_out / _CHECKPOINT_FILENAME
    write_json_atomic(ckpt, {"slide_id": slide_id, "result": result})


def is_slide_done(output_dir: Path, slide_id: str) -> bool:
    """Return True if the checkpoint file for *slide_id* already exists."""
    return (output_dir / slide_id / _CHECKPOINT_FILENAME).exists()


def log_checkpoint_status(
    stage_name: str, total: int, already_done: int
) -> None:
    """Log a summary of how many items will be skipped due to checkpoints."""
    if already_done:
        log_and_print(
            f"[{stage_name}] Resuming: {already_done}/{total} slides already "
            f"completed — {total - already_done} remaining"
        )
    else:
        log_and_print(f"[{stage_name}] Starting fresh: {total} slides to process")


# --------------------------------------------------------------------------- #
# Slide filter helpers
# --------------------------------------------------------------------------- #
# The --slide_filter argument is a comma-separated list of slide names provided
# by the user.  Slide names may contain spaces but not commas.
#
# Because the pipeline uses generated "slide IDs" (sanitised versions of the
# original name), matching is performed against both the raw slide name and
# the generated slide ID.  This way users can specify either the original
# filename stem or the processed ID.
# --------------------------------------------------------------------------- #

def generate_slide_id(slide_name: str, replace_percent: bool = True) -> str:
    """Generate a short, filesystem-safe ID from a slide name.

    This is the canonical implementation used by all pipeline stages.
    """
    if replace_percent:
        slide_name = slide_name.replace('%', '_pct_')
    clean_name = "".join(c for c in slide_name if c.isalnum() or c in " -_")
    short_name = clean_name[:20].strip()
    hash_suffix = hashlib.md5(slide_name.encode()).hexdigest()[:8]
    return f"{short_name}_{hash_suffix}".replace(" ", "_")


def parse_slide_filter(filter_str: Optional[str]) -> Optional[Set[str]]:
    """Parse a comma-separated slide filter string into a set of names.

    Returns ``None`` when no filter is active (process everything).
    Each entry is stripped of leading/trailing whitespace.
    """
    if not filter_str:
        return None
    names = {name.strip() for name in filter_str.split(",") if name.strip()}
    if not names:
        return None
    return names


def build_slide_filter_set(filter_str: Optional[str]) -> Optional[Set[str]]:
    """Build an expanded filter set that includes both raw names AND their
    generated slide IDs so downstream stages can match on either.

    Returns ``None`` when no filter is active.
    """
    raw_names = parse_slide_filter(filter_str)
    if raw_names is None:
        return None
    expanded: Set[str] = set()
    for name in raw_names:
        expanded.add(name)
        expanded.add(generate_slide_id(name, replace_percent=True))
        expanded.add(generate_slide_id(name, replace_percent=False))
    return expanded


def should_process_slide(slide_id: str, slide_filter: Optional[Set[str]]) -> bool:
    """Return True if *slide_id* passes the filter (or no filter is active)."""
    if slide_filter is None:
        return True
    return slide_id in slide_filter
