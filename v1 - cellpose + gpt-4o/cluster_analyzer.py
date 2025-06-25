#!/usr/bin/env python3
import json
import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path

def analyze_cluster_and_tiles(
    json_file_path,
    output_dir,
    confidence_threshold=0.0,
    cluster_analyzer_max_items=None
):
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
                # A unique 'cell_id' is required for this logic.
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

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save filtered data as 'filtered_annotations.json' to match what the filtered_annotation component expects
    filtered_annotations_path = output_path / "filtered_annotations.json"
    with open(filtered_annotations_path, 'w') as f:
        json.dump(filtered_tiles_and_cells, f, indent=4)
    print(f"Filtered annotations saved to: {filtered_annotations_path}")

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

    tile_summary_path = output_path / "tile_summary.json"
    with open(tile_summary_path, 'w') as f:
        json.dump(tile_summary_list, f, indent=4)
    print(f"Tile summary saved to: {tile_summary_path}")

    # Always create comprehensive cluster analysis results
    cluster_analysis = {
        "analysis_metadata": {
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
    
    cluster_analysis_path = output_path / "cluster_analysis.json"
    with open(cluster_analysis_path, 'w') as f:
        json.dump(cluster_analysis, f, indent=4)
    print(f"Cluster analysis results saved to: {cluster_analysis_path}")

    return filtered_tiles_and_cells


def main():
    """
    Main function to parse command-line arguments and run the script.
    """
    parser = argparse.ArgumentParser(
        description="Analyze clusters, filter cells by confidence, and extract the corresponding tile images.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "json_file",
        help="Path to the input JSON file containing tile, cell, and cluster data. Each cell must have a unique 'cell_id'."
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

    print("Analyzing clusters and filtering data...")
    filtered_data = analyze_cluster_and_tiles(
        args.json_file,
        args.output_dir,
        args.confidence_threshold,
        args.max_items
    )

    if filtered_data is None:
        print("\nAnalysis failed. Exiting.")
        return

    print("\nProcess finished.")

if __name__ == "__main__":
    main()