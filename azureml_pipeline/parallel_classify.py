"""
Azure ML Parallel Entry Script for Cell Classification (Per-Slide).

This script implements the init()/run(mini_batch) interface required by
Azure ML parallel_run_function. Each mini-batch contains TRIGGER FILES
(one per slide). The actual data is read from SIDE INPUTS.

Input structure (mini-batch):
    trigger_path/slide_id_A  (empty trigger file)
    
Side inputs:
    segmented_path/slide_id_A/tile_bboxes.json
    prepped_tiles_path/slide_id_A/tile.png
    cluster_path/slide_id_A/cluster_assignments.json

Output creates per-slide subfolders with classification_results.json.

Usage in pipeline:
    parallel_run_function(
        task=RunFunction(entry_script="parallel_classify.py", ...),
        input_data="${{inputs.trigger_path}}",  # Trigger files
        mini_batch_size="1",  # Slides per mini-batch
        ...
    )
"""
from __future__ import annotations
import argparse
import base64
import json
import logging
import os
import re
import sys
import traceback
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from openai import OpenAI

from utils import build_slide_to_bbox_files_mapping, get_tile_path_from_bbox_file, log_and_print, log_exception, mark_slide_done, is_slide_done, log_checkpoint_status, write_json_atomic


# Global state initialized in init()
_args = None
_output_base = None
_client = None
_slide_to_bbox_files = None  # Cache: slide_id -> list of bbox files


def init():
    """
    Initialize OpenAI client before processing mini-batches.
    Called once per worker process.
    """
    global _args, _output_base, _client, _slide_to_bbox_files
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmented_path", type=str, required=True,
                        help="Path to segmentation results (flat structure with *_bboxes.json)")
    parser.add_argument("--prepped_tiles_path", type=str, required=True,
                        help="Path to the tiled images (flat structure)")
    parser.add_argument("--clustered_cells_path", type=str, default="",
                        help="Path to clustering output (contains slide subfolders with cluster_assignments.json)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Base output directory for classification results")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of classes")
    parser.add_argument("--classify_per_cluster", type=int, default=10,
                        help="Number of bounding boxes to classify per cluster (per slide)")
    
    _args, _ = parser.parse_known_args()
    _output_base = Path(_args.output_path)
    _output_base.mkdir(parents=True, exist_ok=True)
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OPENAI_API_KEY found in environment!")
    
    _client = OpenAI(api_key=api_key)
    
    # Build slide-to-files mapping from flat structure
    _slide_to_bbox_files = build_slide_to_bbox_files_mapping(_args.segmented_path)
    
    log_and_print("Parallel classification initialized")
    log_and_print(f"  Segmented path: {_args.segmented_path}")
    log_and_print(f"  Prepped tiles path: {_args.prepped_tiles_path}")
    log_and_print(f"  Clustered cells path: {_args.clustered_cells_path}")
    log_and_print(f"  Output path: {_output_base}")
    log_and_print(f"  Classify per cluster: {_args.classify_per_cluster}")
    log_and_print(f"  Discovered {len(_slide_to_bbox_files)} unique slide IDs")


def get_tile_info(bbox_file: str) -> tuple:
    """Helper function to derive tile/slide info from bbox_file path."""
    return get_tile_path_from_bbox_file(bbox_file, _args.prepped_tiles_path)


def classify_bbox_multimodal_llm(tile_image: Image.Image, bbox: List[int]) -> str:
    """
    Classify a single cell image region using multimodal-LLM.
    """
    try:
        cropped_img = tile_image.crop(bbox)
        
        buffer = BytesIO()
        cropped_img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        response = _client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Take the role of a highly experienced histology expert. "
                        "When given an image of a cell, you must identify the correct cell type. "
                        "**Respond with exactly one label from the list and no additional explanation.** "
                        "Example labels: Alpha, Beta, Gamma, Delta, Duct, Acinar, Stellate, Endothelial, Immune, Undetermined. "
                        "Say 'Undetermined' if you cannot confidently identify the cell type."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Identify the cell type from common pancreatic islet types (Alpha, Beta, Gamma, Delta) or surrounding cell types (Duct, Acinar, Stellate, Endothelial, Immune, etc.). If unsure, say Undetermined."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=50
        )
        
        label = response.choices[0].message.content.strip()
        label = re.sub(r'[^\w\s-]', '', label)
        if not label:
            label = "Undetermined"
        return label
    except Exception as e:
        log_exception(f"Error during multimodal-LLM classification for bbox {bbox}", e)
        if "rate" in str(e).lower():
            return "Error - Rate Limited"
        return "Error - Exception"


def load_cluster_assignments_for_slide(slide_name: str) -> Optional[List[Dict]]:
    """Load cluster assignments for a single slide."""
    if not _args.clustered_cells_path:
        return None
    
    # Check per-slide format first
    slide_cluster_path = os.path.join(_args.clustered_cells_path, slide_name, "cluster_assignments.json")
    if os.path.exists(slide_cluster_path):
        try:
            with open(slide_cluster_path, "r") as f:
                return json.load(f)
        except Exception as e:
            log_exception(f"Failed to load cluster assignments from {slide_cluster_path}", e)
    
    # Fallback to global format - filter for this slide
    global_path = os.path.join(_args.clustered_cells_path, "cluster_assignments.json")
    if os.path.exists(global_path):
        try:
            with open(global_path, "r") as f:
                all_assignments = json.load(f)
            # Filter for this slide
            slide_assignments = []
            for entry in all_assignments:
                _, _, entry_slide = get_tile_info(entry["bbox_file"])
                if entry_slide == slide_name:
                    slide_assignments.append(entry)
            if slide_assignments:
                return slide_assignments
        except Exception as e:
            log_exception(f"Failed to load global cluster assignments", e)
    
    return None


def classify_slide(slide_id: str) -> dict:
    """
    Perform classification for a single slide.
    
    Returns dict with:
        - slide_id: ID of the slide
        - num_cells_total: total cells in slide
        - num_cells_classified: number actually classified by multimodal-LLM
        - num_tiles: number of tiles with results
        - output_path: path to classification_results.json
    """
    slide_output_path = _output_base / slide_id
    slide_output_path.mkdir(parents=True, exist_ok=True)
    
    all_cell_results = []
    
    # Try to load cluster assignments for this slide
    clustered_cells = load_cluster_assignments_for_slide(slide_id)
    
    if clustered_cells:
        log_and_print(f"Using cluster assignments for slide {slide_id}: {len(clustered_cells)} cells")
        
        # Group by cluster and select top N per cluster
        cluster_map = defaultdict(list)
        for entry in clustered_cells:
            cluster_id = entry["cluster_id"]
            cluster_map[cluster_id].append(entry)
        
        selected_for_classification = []
        for cluster_id, entries in cluster_map.items():
            sorted_entries = sorted(entries, key=lambda e: e.get("confidence", 0.0), reverse=True)
            top_n = sorted_entries[:_args.classify_per_cluster]
            selected_for_classification.extend(top_n)
        
        log_and_print(f"Selected {len(selected_for_classification)} cells for multimodal-LLM classification")
        
        # Group selected by tile for efficient processing
        bboxes_by_tile = defaultdict(list)
        for entry in selected_for_classification:
            tile_path, _, _ = get_tile_info(entry["bbox_file"])
            if tile_path:
                bboxes_by_tile[tile_path].append(entry)
        
        # Classify selected cells
        multimodal_llm_predictions = {}
        tile_img_cache = {}
        
        for tile_path, entries_in_tile in bboxes_by_tile.items():
            if not os.path.exists(tile_path):
                log_and_print(f"Tile image not found: {tile_path}", level="warning")
                for entry in entries_in_tile:
                    multimodal_llm_predictions[(entry["bbox_file"], entry["label_id"])] = "Error - Tile Not Found"
                continue
            
            try:
                if tile_path not in tile_img_cache:
                    tile_img_cache[tile_path] = Image.open(tile_path).convert("RGB")
                tile_img = tile_img_cache[tile_path]
                
                for entry in entries_in_tile:
                    bbox = entry["bbox"]
                    cell_id = (entry["bbox_file"], entry["label_id"])
                    pred_class = classify_bbox_multimodal_llm(tile_img, bbox)
                    multimodal_llm_predictions[cell_id] = pred_class
                    
            except Exception as e:
                log_exception(f"Error processing tile {tile_path} (classification batch)", e)
                for entry in entries_in_tile:
                    cell_id = (entry["bbox_file"], entry["label_id"])
                    if cell_id not in multimodal_llm_predictions:
                        multimodal_llm_predictions[cell_id] = "Error - Tile Processing Failed"
        
        # Close cached images
        for img in tile_img_cache.values():
            img.close()
        
        # Compile results for all clustered cells
        for entry in clustered_cells:
            bbox_file = entry["bbox_file"]
            label_id = entry["label_id"]
            cell_id = (bbox_file, label_id)
            tile_path, tile_name, _ = get_tile_info(bbox_file)
            
            if not tile_path:
                continue
            
            pred_class = multimodal_llm_predictions.get(cell_id, "Unclassified")
            
            all_cell_results.append({
                "bbox_file": bbox_file,
                "tile_path": tile_path,
                "slide_id": slide_id,
                "tile_name": tile_name,
                "label_id": label_id,
                "bbox": entry["bbox"],
                "cluster_id": entry.get("cluster_id", -1),
                "cluster_confidence": entry.get("confidence", 0.0),
                "pred_class": pred_class
            })
        
        num_classified = len(multimodal_llm_predictions)
        
    else:
        # No cluster assignments - classify all bboxes from flat structure
        log_and_print(f"No cluster assignments found for {slide_id}. Classifying all bboxes.")
        
        # Get bbox files for this slide from the mapping
        bbox_files = _slide_to_bbox_files.get(slide_id, [])
        if not bbox_files:
            log_and_print(f"No bbox files found for slide {slide_id}", level="warning")
            return {
                "slide_id": slide_id,
                "num_cells_total": 0,
                "num_cells_classified": 0,
                "num_tiles": 0,
                "output_path": None,
                "error": "No bbox files found"
            }
        
        tile_img_cache = {}
        num_classified = 0
        
        for bbox_file in bbox_files:
            tile_path, tile_name, _ = get_tile_info(bbox_file)
            
            if not tile_path or not os.path.exists(tile_path):
                continue
            
            try:
                with open(bbox_file, "r") as f:
                    bboxes_in_file = json.load(f)
            except Exception as e:
                log_exception(f"Failed to read bbox file {bbox_file}", e)
                continue
            
            if not bboxes_in_file:
                continue
            
            try:
                if tile_path not in tile_img_cache:
                    tile_img_cache[tile_path] = Image.open(tile_path).convert("RGB")
                tile_img = tile_img_cache[tile_path]
                
                for bbox_entry in bboxes_in_file:
                    label_id = bbox_entry["label_id"]
                    bbox = bbox_entry["bbox"]
                    
                    pred_class = classify_bbox_multimodal_llm(tile_img, bbox)
                    num_classified += 1
                    
                    all_cell_results.append({
                        "bbox_file": bbox_file,
                        "tile_path": tile_path,
                        "slide_id": slide_id,
                        "tile_name": tile_name,
                        "label_id": label_id,
                        "bbox": bbox,
                        "cluster_id": None,
                        "cluster_confidence": None,
                        "pred_class": pred_class
                    })
                    
            except Exception as e:
                log_exception(f"Error processing tile {tile_path} (no-cluster classification)", e)
        
        # Close cached images
        for img in tile_img_cache.values():
            img.close()
    
    # Restructure results by tile
    grouped_by_tile = {}
    for result in all_cell_results:
        tile_path = result["tile_path"]
        
        if tile_path not in grouped_by_tile:
            grouped_by_tile[tile_path] = {
                "tile_path": result["tile_path"],
                "tile_name": result["tile_name"],
                "slide_id": result["slide_id"],
                "classified_cells": []
            }
        
        cell_data = {
            "label_id": result["label_id"],
            "bbox": result["bbox"],
            "bbox_file": result["bbox_file"],
            "pred_class": result["pred_class"]
        }
        if result.get("cluster_id") is not None:
            cell_data["cluster_id"] = result["cluster_id"]
        if result.get("cluster_confidence") is not None:
            cell_data["cluster_confidence"] = result["cluster_confidence"]
        
        grouped_by_tile[tile_path]["classified_cells"].append(cell_data)
    
    final_output_data = list(grouped_by_tile.values())
    
    # Save results (with fsync for preemption safety on FUSE mounts)
    out_file = slide_output_path / "classification_results.json"
    write_json_atomic(out_file, final_output_data)
    
    log_and_print(f"Classification for {slide_id}: {len(all_cell_results)} cells total, {num_classified} classified by multimodal-LLM")
    
    return {
        "slide_id": slide_id,
        "num_cells_total": len(all_cell_results),
        "num_cells_classified": num_classified,
        "num_tiles": len(grouped_by_tile),
        "output_path": str(out_file)
    }


def run(mini_batch: List[str]) -> List[str]:
    """
    Process a mini-batch of TRIGGER FILES from the clustering step.
    
    Each item in mini_batch is a path to a trigger file named after the slide_id.
    The actual bbox files are read from segmented_path side input.
    
    Args:
        mini_batch: List of paths to trigger files (e.g., trigger_path/slide_123)
        
    Returns:
        List of result strings (one per processed slide) - required by Azure ML
    """
    results = []
    
    for trigger_file in mini_batch:
        trigger_path = Path(str(trigger_file).strip())
        if not trigger_path.name:
            continue
        
        # Extract slide_id from trigger filename
        slide_id = trigger_path.name
        
        log_and_print(f"Processing slide from trigger: {slide_id}")
        
        # Skip slides already completed (checkpoint resume across preempted jobs)
        if is_slide_done(_output_base, slide_id):
            log_and_print(f"Skipping (already done): {slide_id}")
            results.append(f"SKIP:{slide_id}:already_done")
            continue
        
        # Verify slide exists in our mapping (built during init)
        if slide_id not in _slide_to_bbox_files:
            log_and_print(f"Slide ID '{slide_id}' not in pre-built mapping, skipping", level="warning")
            results.append(f"SKIP:{slide_id}:not_in_mapping")
            continue
        
        try:
            result_info = classify_slide(slide_id)
            
            if result_info.get("error"):
                result = f"WARN:{slide_id}:{result_info['error']}"
            else:
                result = f"OK:{slide_id}:total={result_info['num_cells_total']},classified={result_info['num_cells_classified']}"
                mark_slide_done(_output_base, slide_id, result)
            
            log_and_print(f"Finished classifying {slide_id}")
            results.append(result)
            
        except Exception as e:
            log_exception(f"Error classifying {slide_id}", e)
            results.append(f"ERROR:{slide_id}:{str(e)}")
    
    return results


def shutdown():
    """Cleanup after all mini-batches processed."""
    global _client
    _client = None
    log_and_print("Parallel classification shutdown complete")


# --------------------------------------------------------------------------- #
# Sequential (single-node) entry point
# --------------------------------------------------------------------------- #
def run_all() -> None:
    """
    Process ALL slides sequentially.
    Used when running as a command() job (max_nodes=1) for easier debugging.
    Slide IDs are discovered from the segmentation output subfolders.
    """
    init()
    slide_ids = sorted(_slide_to_bbox_files.keys())
    log_and_print(f"[Sequential] Found {len(slide_ids)} slides to classify")

    # --- checkpoint resume ---
    already_done = sum(1 for sid in slide_ids if is_slide_done(_output_base, sid))
    log_checkpoint_status("Classification", len(slide_ids), already_done)

    all_results = []
    for slide_id in slide_ids:
        if is_slide_done(_output_base, slide_id):
            log_and_print(f"[Sequential] Skipping (already done): {slide_id}")
            continue
        log_and_print(f"[Sequential] Classifying slide: {slide_id}")
        # Create a fake trigger file path (run() only uses the filename)
        fake_trigger = os.path.join(_args.segmented_path, slide_id)
        results = run([fake_trigger])
        all_results.extend(results)
        for r in results:
            log_and_print(f"  -> {r}")
            if r.startswith("OK:"):
                mark_slide_done(_output_base, slide_id, r)

    log_and_print(f"[Sequential] Classification complete. {len(all_results)} slides processed.")
    shutdown()


if __name__ == "__main__":
    # All args are parsed by init() via parse_known_args
    run_all()
