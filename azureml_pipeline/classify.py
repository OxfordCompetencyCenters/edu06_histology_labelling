import argparse
import os
import glob
import json
import logging
import base64
import re
from PIL import Image
from io import BytesIO
import openai
from openai import OpenAI
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_cluster_assignments(clustered_cells_path):
    """
    Load cluster assignments from either global or per-slide clustering format.
    
    Args:
        clustered_cells_path: Path to cluster output directory
        
    Returns:
        List of cluster assignment entries, or None if no valid assignments found
    """
    if not clustered_cells_path or not os.path.exists(clustered_cells_path):
        return None
    
    # Try global clustering format first (single cluster_assignments.json)
    global_path = os.path.join(clustered_cells_path, "cluster_assignments.json")
    if os.path.exists(global_path):
        logging.info(f"Loading global cluster assignments from {global_path}")
        try:
            with open(global_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load global cluster assignments: {e}")
            return None
    
    # Try per-slide clustering format (slide_name/cluster_assignments.json)
    slide_dirs = [d for d in os.listdir(clustered_cells_path) 
                  if os.path.isdir(os.path.join(clustered_cells_path, d))]
    
    if not slide_dirs:
        logging.warning(f"No cluster_assignments.json found at {global_path} and no slide directories found in {clustered_cells_path}")
        return None
    
    all_assignments = []
    slides_found = 0
    for slide_name in slide_dirs:
        slide_path = os.path.join(clustered_cells_path, slide_name, "cluster_assignments.json")
        if os.path.exists(slide_path):
            try:
                with open(slide_path, "r") as f:
                    slide_assignments = json.load(f)
                all_assignments.extend(slide_assignments)
                slides_found += 1
                logging.info(f"Loaded {len(slide_assignments)} cluster assignments from {slide_path}")
            except Exception as e:
                logging.error(f"Failed to load cluster assignments from {slide_path}: {e}")
    
    if slides_found > 0:
        logging.info(f"Combined cluster assignments from {slides_found} slides: {len(all_assignments)} total entries")
        return all_assignments
    else:
        logging.warning(f"No cluster_assignments.json files found in any slide directories in {clustered_cells_path}")
        return None

def parse_slide_name(tile_name):
    """
    Extract slide name from tile filename using new format only.
    
    New format: 'slide_id__MAG_1d000__X_2048__Y_1024__IDX_000001.png' -> 'slide_id'
    """
    # Remove .png extension if present
    if tile_name.endswith('.png') or tile_name.endswith('.PNG'):
        base_name = tile_name[:-4]
    else:
        base_name = tile_name
    
    # New format only: slide_id is everything before the first __MAG_
    if "__MAG_" in base_name:
        slide_id = base_name.split("__MAG_")[0]
        return slide_id
    else:
        # If it doesn't match new format, return as-is
        return base_name

def get_tile_info(bbox_file, prepped_tiles_path):
    """Helper function to derive tile/slide info from bbox_file path."""
    try:
        tile_base_name = os.path.basename(bbox_file).replace("_bboxes.json", "")
        tile_file_name = tile_base_name + ".png"
        # Assumes bbox_file is like /path/to/segmented/SLIDE_NAME/TILE_NAME_bboxes.json
        slide_name = os.path.basename(os.path.dirname(bbox_file))
        tile_path = os.path.join(prepped_tiles_path, slide_name, tile_file_name)
        return tile_path, tile_base_name, slide_name
    except Exception as e:
        logging.error(f"Error parsing info from bbox_file '{bbox_file}': {e}")
        return None, None, None


def classify_bbox_gpt4o(client, tile_image, bbox):
    """
    Classify a single cell image region using GPT-4o with vision capabilities.
    Crops the given tile image to the bounding box, sends it along with label options
    to the GPT-4o model, and returns the chosen label.
    """
    try:
        # Crop the tile image to the specified bounding box
        cropped_img = tile_image.crop(bbox)

        # Save the cropped image to an in-memory binary (bytes) object
        buffer = BytesIO()
        cropped_img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        # Base64 encode the image
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Call the GPT-4o Chat Completions API with the image and prompt
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Take the role of a highly experienced histology expert. "
                        "When given an image of a cell, you must identify the correct cell type. "
                        "**Respond with exactly one label from the list and no additional explanation.** "
                        "Example labels might include: Alpha, Beta, Gamma, Delta, Duct, Acinar, Stellate, Endothelial, Immune, Undetermined. " # Added examples for clarity
                        "Say 'Undetermined' if you cannot confidently identify the cell type."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is a cell image, identify the cell type from the common pancreatic islet types (Alpha, Beta, Gamma, Delta) or other surrounding cell types (Duct, Acinar, Stellate, Endothelial, Immune, etc.). If unsure, say Undetermined."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=50 # Limit response length
        )

        # Extract the label from the model's response (first choice)
        label = response.choices[0].message.content.strip()
        # Basic validation/cleanup
        label = re.sub(r'[^\w\s-]', '', label) # Remove potentially strange characters
        if not label:
            label = "Undetermined" # Default if empty response
        return label
    except openai.RateLimitError:
        logging.error("OpenAI rate limit exceeded. Consider adding delays or reducing request frequency.")
        return "Error - Rate Limited"
    except openai.APIError as e:
        logging.error(f"OpenAI API Error during classification: {e}")
        return f"Error - API Error: {e}"
    except Exception as e:
        logging.error(f"Unexpected error during GPT-4o classification for bbox {bbox}: {e}")
        return "Error - Exception"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmented_path", type=str, required=True,
                        help="Path to segmentation results (masks + *_bboxes.json).")
    parser.add_argument("--prepped_tiles_path", type=str, required=True,
                        help="Path to the original prepped tile images.")
    parser.add_argument("--clustered_cells_path", type=str, default="",
                        help="Path to cluster output (containing cluster_assignments.json).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path for classification results.")
    parser.add_argument("--num_classes", type=int, default=4, # This seems less relevant now with GPT-4o but kept for interface consistency
                        help="Number of classes (if you want a consistent interface).")
    parser.add_argument("--classify_per_cluster", type=int, default=10,
                        help="Number of bounding boxes to classify per cluster (per slide), from highest to lowest confidence.")
    args = parser.parse_args()

    logging.info("Starting GPT-4o classification with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Consider allowing key via argument as well, but env var is good practice
        logging.error("FATAL: No OPENAI_API_KEY found in environment!")
        raise ValueError("No OPENAI_API_KEY found in environment!")
    # openai.api_key = api_key # Deprecated way
    client = OpenAI(api_key=api_key) # Recommended way

    # This list will hold dictionaries for ALL cells, classified or not
    all_cell_results = []

    # ------------------------------------------------------------------
    # (A) If clustered_cells_path is provided, use cluster assignments
    # ------------------------------------------------------------------
    all_clustered_cells = load_cluster_assignments(args.clustered_cells_path)
    if all_clustered_cells:
        logging.info("Using cluster assignments to select bounding boxes.")

        # all_clustered_cells already loaded above
        logging.info(f"Loaded {len(all_clustered_cells)} total cell entries from cluster assignments.")

        # Group bounding boxes by (cluster_id, slide_name) to select top N
        cluster_map = defaultdict(list)
        for entry in all_clustered_cells:
            # Need slide_name for grouping
            _, _, slide_name = get_tile_info(entry["bbox_file"], args.prepped_tiles_path)
            if not slide_name:
                 logging.warning(f"Could not determine slide name for bbox_file {entry['bbox_file']}. Skipping entry.")
                 continue # Skip if we can't parse needed info
            cluster_id = entry["cluster_id"]
            # Add slide_name to the entry if needed later (though get_tile_info can derive it)
            entry['_slide_name_temp'] = slide_name # Temporary key for grouping
            cluster_map[(cluster_id, slide_name)].append(entry)

        # Sort each cluster+slide group by descending confidence and pick top N to classify
        selected_bboxes_for_classification = []
        for (cluster_id, slide_name), entries in cluster_map.items():
            sorted_entries = sorted(entries, key=lambda e: e.get("confidence", 0.0), reverse=True) # Use .get for safety
            top_n = sorted_entries[: args.classify_per_cluster]
            selected_bboxes_for_classification.extend(top_n)

        logging.info(f"Selected a total of {len(selected_bboxes_for_classification)} bounding boxes across all clusters/slides for GPT-4o classification.")

        # Create a set of unique identifiers for fast lookup of cells that WILL be classified
        # Using (bbox_file, label_id) as a unique identifier for a cell within a tile
        ids_to_classify = set(
            (entry["bbox_file"], entry["label_id"]) for entry in selected_bboxes_for_classification
        )

        # Store actual GPT-4o predictions here, mapping identifier to predicted class
        gpt_predictions = {}

        # Group selected boxes by tile_path for efficient image loading
        bboxes_grouped_by_tile = defaultdict(list)
        for entry in selected_bboxes_for_classification:
             tile_path, _, _ = get_tile_info(entry["bbox_file"], args.prepped_tiles_path)
             if tile_path:
                 bboxes_grouped_by_tile[tile_path].append(entry)
             else:
                 logging.warning(f"Could not get tile_path for selected bbox_file {entry['bbox_file']}. Skipping classification for this cell.")


        # Perform GPT-4o classification ONLY for the selected cells
        tile_img_cache = {} # Cache loaded images
        for tile_path, entries_in_tile in bboxes_grouped_by_tile.items():
            if not os.path.exists(tile_path):
                logging.warning(f"Tile image not found at {tile_path} (referenced by {entries_in_tile[0]['bbox_file']}). Skipping classification for {len(entries_in_tile)} cells in this tile.")
                # Mark these as errored/skipped in the predictions map
                for entry in entries_in_tile:
                     cell_id = (entry["bbox_file"], entry["label_id"])
                     gpt_predictions[cell_id] = "Error - Tile Not Found"
                continue

            try:
                if tile_path not in tile_img_cache:
                     logging.info(f"Loading tile image: {tile_path}")
                     tile_img_cache[tile_path] = Image.open(tile_path).convert("RGB")
                tile_img = tile_img_cache[tile_path]

                logging.info(f"Classifying {len(entries_in_tile)} selected cells in {os.path.basename(tile_path)}...")
                for entry in entries_in_tile:
                    bbox = entry["bbox"]
                    cell_id = (entry["bbox_file"], entry["label_id"])

                    # Call GPT-4o for classification
                    pred_class = classify_bbox_gpt4o(client, tile_img, bbox)
                    gpt_predictions[cell_id] = pred_class # Store prediction

            except Exception as e:
                logging.error(f"Error processing tile {tile_path}: {e}. Skipping classification for remaining cells in this tile.")
                # Mark remaining cells in this tile as errored if an image loading error occurred
                for entry in entries_in_tile:
                     cell_id = (entry["bbox_file"], entry["label_id"])
                     if cell_id not in gpt_predictions: # Only mark if not already processed
                         gpt_predictions[cell_id] = "Error - Tile Processing Failed"

            # Optional: Clear image from cache if memory is a concern, but likely faster to keep it
            # if tile_path in tile_img_cache: del tile_img_cache[tile_path]


        # Now, iterate through ALL cells from the original cluster data
        # and build the final result list, adding the prediction if available,
        # or 'Unclassified' otherwise.
        logging.info("Compiling final results for all clustered cells...")
        for entry in all_clustered_cells:
            bbox_file = entry["bbox_file"]
            label_id = entry["label_id"]
            cell_id = (bbox_file, label_id)

            tile_path, tile_name, slide_name = get_tile_info(bbox_file, args.prepped_tiles_path)

            if not tile_path: # Skip if we couldn't parse info
                logging.warning(f"Skipping final result for cell {label_id} from {bbox_file} due to parsing error.")
                continue

            # Determine the predicted class
            if cell_id in gpt_predictions:
                pred_class = gpt_predictions[cell_id]
            else:
                # This cell was in cluster_assignments but NOT selected for classification
                pred_class = "Unclassified"

            # Append the comprehensive data for this cell
            all_cell_results.append({
                "bbox_file": bbox_file,
                "tile_path": tile_path,
                "slide_name": slide_name,
                "tile_name": tile_name,
                "label_id": label_id,
                "bbox": entry["bbox"],
                "cluster_id": entry.get("cluster_id", -1), # Use .get for safety
                "cluster_confidence": entry.get("confidence", 0.0), # Use .get for safety
                "pred_class": pred_class
            })
        logging.info(f"Compiled {len(all_cell_results)} total cell results including 'Unclassified'.")

    else:
        # ------------------------------------------------------------------
        # (B) If no cluster assignments, classify *all* bounding boxes
        #     from segmented_path (original logic, but ensure consistent output).
        # ------------------------------------------------------------------
        logging.info("No valid cluster assignments found or path not provided. Classifying ALL bounding boxes from segmented_path.")
        bbox_files = glob.glob(
            os.path.join(args.segmented_path, "**/*_bboxes.json"),
            recursive=True
        )
        if not bbox_files:
            logging.warning("No bounding box JSON files found under: %s", args.segmented_path)
            # No return here, proceed to save empty results if necessary

        tile_img_cache = {} # Cache loaded images
        for bbox_file in bbox_files:
            tile_path, tile_name, slide_name = get_tile_info(bbox_file, args.prepped_tiles_path)

            if not tile_path:
                logging.warning(f"Could not determine tile path for {bbox_file}. Skipping this file.")
                continue

            if not os.path.exists(tile_path):
                logging.warning(f"Tile image not found at {tile_path} (referenced by {bbox_file}). Skipping classification for this tile.")
                # Optionally: add entries with error status if needed, otherwise just skip
                continue

            try:
                 with open(bbox_file, "r") as f:
                    bboxes_in_file = json.load(f)
            except Exception as e:
                 logging.error(f"Failed to read or parse bbox file {bbox_file}: {e}. Skipping.")
                 continue


            if not bboxes_in_file:
                logging.info(f"No bounding boxes found in {bbox_file}. Skipping.")
                continue

            try:
                if tile_path not in tile_img_cache:
                    logging.info(f"Loading tile image: {tile_path}")
                    tile_img_cache[tile_path] = Image.open(tile_path).convert("RGB")
                tile_img = tile_img_cache[tile_path]

                logging.info(f"Classifying {len(bboxes_in_file)} cells in {tile_name}...")
                for bbox_entry in bboxes_in_file:
                    label_id = bbox_entry["label_id"]
                    bbox = bbox_entry["bbox"]

                    # Call GPT-4o for classification - ALL cells are classified here
                    pred_class = classify_bbox_gpt4o(client, tile_img, bbox)

                    # Append result - consistent structure with Mode A
                    all_cell_results.append({
                        "bbox_file": bbox_file, # Added
                        "tile_path": tile_path,
                        "slide_name": slide_name, # Added
                        "tile_name": tile_name, # Added
                        "label_id": label_id,
                        "bbox": bbox,
                        # Add placeholders or nulls for cluster info if needed for perfect schema match
                        "cluster_id": None, # Or -1, or omit if schema allows missing keys
                        "cluster_confidence": None, # Or 0.0, or omit
                        "pred_class": pred_class
                    })

            except Exception as e:
                logging.error(f"Error processing tile {tile_path}: {e}. Skipping classification for cells in this tile.")
                # Optionally add errored entries here if needed


    # ------------------------------------------------------------------
    # Restructure results (Group by tile) - This part remains largely the same,
    # but now operates on `all_cell_results` which contains ALL cells.
    # ------------------------------------------------------------------
    logging.info("Restructuring classification results...")
    grouped_by_tile = {}
    for result in all_cell_results:
        tile_path = result["tile_path"]

        if tile_path not in grouped_by_tile:
            # Initialize the tile entry using info from the first cell found for that tile
            grouped_by_tile[tile_path] = {
                "tile_path": result["tile_path"],
                "tile_name": result["tile_name"],
                "slide_name": result["slide_name"],
                "classified_cells": [] # Renamed from classified_cells for clarity, now holds ALL cells
            }

        # Prepare the cell data dictionary for this specific cell
        cell_data = {
            "label_id": result["label_id"],
            "bbox": result["bbox"],
            "bbox_file": result["bbox_file"], # Ensure this is always present
            "pred_class": result["pred_class"] # Will be 'Unclassified' or GPT prediction or Error
        }
        # Add cluster info if it exists (primarily from Mode A)
        if "cluster_id" in result and result["cluster_id"] is not None:
            cell_data["cluster_id"] = result["cluster_id"]
        if "cluster_confidence" in result and result["cluster_confidence"] is not None:
            cell_data["cluster_confidence"] = result["cluster_confidence"]

        # Append this cell's data to the list for its tile
        grouped_by_tile[tile_path]["classified_cells"].append(cell_data)

    # Convert the dictionary of tiles back into a list and group by slide
    final_output_data = list(grouped_by_tile.values())
    
    # Group results by slide name for per-slide output
    slides_data = defaultdict(list)
    for tile_data in final_output_data:
        slide_name = tile_data["slide_name"]
        slides_data[slide_name].append(tile_data)
    
    # Create output directory structure and save per slide
    os.makedirs(args.output_path, exist_ok=True)
    
    total_tiles_written = 0
    for slide_name, tiles_for_slide in slides_data.items():
        slide_output_dir = os.path.join(args.output_path, slide_name)
        os.makedirs(slide_output_dir, exist_ok=True)
        
        out_file = os.path.join(slide_output_dir, "classification_results.json")
        logging.info(f"Writing {len(tiles_for_slide)} tile entries for slide {slide_name} to {out_file}...")
        try:
            with open(out_file, "w") as f:
                json.dump(tiles_for_slide, f, indent=2)
            total_tiles_written += len(tiles_for_slide)
            logging.info(f"Classification for slide {slide_name} complete. Output: {out_file}")
        except Exception as e:
            logging.error(f"Failed to write output JSON for slide {slide_name} to {out_file}: {e}")
    
    logging.info(f"Classification step complete. Total {total_tiles_written} tile entries written across {len(slides_data)} slides.")

if __name__ == "__main__":
    main()