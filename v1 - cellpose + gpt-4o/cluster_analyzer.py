#!/usr/bin/env python3
import json
import argparse
import os
import shutil
import glob
from collections import defaultdict
from pathlib import Path

def find_annotation_files(json_path):
    """
    Find annotation JSON files from either a direct file path or a directory.
    Returns a dictionary mapping slide names to file paths for per-slide processing.
    
    Args:
        json_path: Path to a JSON file or directory containing JSON files
        
    Returns:
        Dictionary mapping slide names to annotation JSON file paths
    """
    annotation_files = {}
    
    if os.path.isfile(json_path):
        # Single file - extract slide name from parent directory or use 'global'
        parent_dir = os.path.basename(os.path.dirname(json_path))
        slide_name = parent_dir if parent_dir and parent_dir != '.' else 'global'
        annotation_files[slide_name] = json_path
        return annotation_files
    
    if os.path.isdir(json_path):
        # Look for per-slide structure: slide_name/annotation_file.json
        for slide_dir in os.listdir(json_path):
            slide_path = os.path.join(json_path, slide_dir)
            if os.path.isdir(slide_path):
                # Look for common annotation file patterns in this slide directory
                patterns = [
                    "*.json",
                    "*annotations.json",
                    "*_annotations.json",
                    "final_annotations.json",
                    "v1_*_annotations.json"
                ]
                
                for pattern in patterns:
                    matches = glob.glob(os.path.join(slide_path, pattern))
                    if matches:
                        # Prefer files with "annotation" in the name
                        annotation_files_in_dir = [f for f in matches if "annotation" in os.path.basename(f).lower()]
                        if annotation_files_in_dir:
                            annotation_files[slide_dir] = annotation_files_in_dir[0]
                            break
                        else:
                            annotation_files[slide_dir] = matches[0]
                            break
        
        # Fallback: look for a single global file in the root directory
        if not annotation_files:
            patterns = [
                "*.json",
                "*annotations.json", 
                "*_annotations.json",
                "final_annotations.json",
                "v1_*_annotations.json"
            ]
            
            for pattern in patterns:
                matches = glob.glob(os.path.join(json_path, pattern))
                if matches:
                    annotation_files_in_root = [f for f in matches if "annotation" in os.path.basename(f).lower()]
                    if annotation_files_in_root:
                        annotation_files['global'] = annotation_files_in_root[0]
                        break
                    else:
                        annotation_files['global'] = matches[0]
                        break
    
    if not annotation_files:
        raise FileNotFoundError(f"No JSON annotation files found at {json_path}")
    
    return annotation_files

def analyze_cluster_and_tiles_per_slide(
    json_file_path,
    slide_name,
    output_dir,
    confidence_threshold=0.0,
    cluster_analyzer_max_items=None
):
    """Process a single slide's annotation file and extract filtered representative tiles."""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file_path}' is not a valid JSON file.")
        return None
    
    clusters_dict = defaultdict(list)
    for tile in data:
        for cell in tile.get("cells", []):
            if cell.get("cluster_confidence", 0.0) >= confidence_threshold:
                cluster_id = cell.get("cluster_id")
                cell_id = cell.get("cell_id")
                if cluster_id is not None and cell_id is not None:
                    clusters_dict[cluster_id].append(
                        (cell_id, cell.get("cluster_confidence"))
                    )

    for cluster_id in clusters_dict:
        clusters_dict[cluster_id].sort(key=lambda x: x[1], reverse=True)

    if cluster_analyzer_max_items is not None:
        for cluster_id in clusters_dict:
            clusters_dict[cluster_id] = clusters_dict[cluster_id][:cluster_analyzer_max_items]

    filtered_cell_ids = {
        cell_id for cell_list in clusters_dict.values() for cell_id, conf in cell_list
    }

    filtered_tiles_and_cells = []
    tile_map = {}

    for tile in data:
        tile_path = tile.get("tile_path")
        if not tile_path:
            continue
        
        for cell in tile.get("cells", []):
            if cell.get("cell_id") in filtered_cell_ids:
                if tile_path not in tile_map:
                    new_tile = {key: value for key, value in tile.items() if key != 'cells'}
                    new_tile['cells'] = []
                    filtered_tiles_and_cells.append(new_tile)
                    tile_map[tile_path] = new_tile                
                tile_map[tile_path]['cells'].append(cell)

    # Create slide-specific output directory
    slide_output_dir = Path(output_dir) / slide_name
    slide_output_dir.mkdir(parents=True, exist_ok=True)

    # Save filtered data as 'filtered_annotations.json' for this slide
    filtered_annotations_path = slide_output_dir / "filtered_annotations.json"
    with open(filtered_annotations_path, 'w') as f:
        json.dump(filtered_tiles_and_cells, f, indent=4)
    print(f"Filtered annotations for slide {slide_name} saved to: {filtered_annotations_path}")

    tile_summary_list = []
    for tile in filtered_tiles_and_cells:
        cells_per_cluster = defaultdict(int)
        for cell in tile.get("cells", []):
            cells_per_cluster[cell.get("cluster_id")] += 1
        
        tile_summary = {
            "tile_path": tile.get("tile_path"),
            "clusters": sorted(list(cells_per_cluster.keys())),
            "number_of_cells_per_cluster": dict(cells_per_cluster),
        }
        tile_summary_list.append(tile_summary)

    tile_summary_path = slide_output_dir / "tile_summary.json"
    with open(tile_summary_path, 'w') as f:
        json.dump(tile_summary_list, f, indent=4)
    print(f"Tile summary for slide {slide_name} saved to: {tile_summary_path}")

    # Create cluster analysis results for this slide
    cluster_analysis = {
        "analysis_metadata": {
            "slide_name": slide_name,
            "confidence_threshold": confidence_threshold,
            "max_items_per_cluster": cluster_analyzer_max_items,
            "total_clusters": len(clusters_dict),
            "total_filtered_cells": len(filtered_cell_ids),
            "total_tiles_with_filtered_cells": len(filtered_tiles_and_cells)
        },
        "cluster_statistics": {
            cluster_id: {
                "total_cells": len(cell_list),
                "confidence_range": {
                    "min": min(conf for _, conf in cell_list),
                    "max": max(conf for _, conf in cell_list),
                    "avg": sum(conf for _, conf in cell_list) / len(cell_list)
                }
            }
            for cluster_id, cell_list in clusters_dict.items()
        },
        "tiles_summary": tile_summary_list,
        "filtered_annotations": filtered_tiles_and_cells
    }
    
    cluster_analysis_path = slide_output_dir / "cluster_analysis.json"
    with open(cluster_analysis_path, 'w') as f:
        json.dump(cluster_analysis, f, indent=4)
    print(f"Cluster analysis results for slide {slide_name} saved to: {cluster_analysis_path}")

    return filtered_tiles_and_cells


def main():
    """
    Main function to parse command-line arguments and run the script.
    """
    parser = argparse.ArgumentParser(
        description="Analyze clusters, filter cells by confidence, and extract the corresponding tile images per slide.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--json_file",
        type=str,
        help="Path to the input JSON file containing tile, cell, and cluster data. Each cell must have a unique 'cell_id'."
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        help="Directory containing the input JSON files with tile, cell, and cluster data (per slide structure)."
    )
    parser.add_argument(
        "--tiles_dir",
        type=str,
        required=True,
        help="Directory containing the original tile images."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save filtered JSON files and extracted tile images."
    )
    parser.add_argument(
        "-c", "--confidence_threshold",
        type=float,
        default=0.0,
        help="Minimum cluster confidence to include a cell (e.g., 0.75).\nDefault is 0.0 (include all)."
    )
    parser.add_argument(
        "-n", "--max_items",
        type=int,
        default=None,
        help="Maximum number of representative cells per cluster.\nDefault is to include all cells that meet the confidence threshold."
    )

    args = parser.parse_args()

    # Determine JSON file paths (per-slide structure)
    if args.json_file and args.json_dir:
        print("Error: Specify either --json_file or --json_dir, not both.")
        return
    elif args.json_file:
        # Single file - treat as one slide
        slide_name = os.path.basename(os.path.dirname(args.json_file))
        if not slide_name or slide_name == '.':
            slide_name = 'global'
        annotation_files = {slide_name: args.json_file}
    elif args.json_dir:
        try:
            annotation_files = find_annotation_files(args.json_dir)
            print(f"Found annotation files for slides: {list(annotation_files.keys())}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    else:
        print("Error: Must specify either --json_file or --json_dir.")
        return

    print("Analyzing clusters and filtering data per slide...")
    
    total_slides_processed = 0
    for slide_name, json_path in annotation_files.items():
        print(f"\nProcessing slide: {slide_name}")
        filtered_data = analyze_cluster_and_tiles_per_slide(
            json_path,
            slide_name,
            args.output_dir,
            args.confidence_threshold,
            args.max_items
        )

        if filtered_data is None:
            print(f"Analysis failed for slide {slide_name}.")
            continue
        
        total_slides_processed += 1
        print(f"Completed processing slide {slide_name}: {len(filtered_data)} tiles with filtered cells")

    print(f"\nProcess finished. Successfully processed {total_slides_processed} slides.")

if __name__ == "__main__":
    main()