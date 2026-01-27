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
    """
    if not clustered_cells_path or not os.path.exists(clustered_cells_path):
        return None
    
    global_path = os.path.join(clustered_cells_path, "cluster_assignments.json")
    if os.path.exists(global_path):
        logging.info(f"Loading global cluster assignments from {global_path}")
        try:
            with open(global_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load global cluster assignments: {e}")
            return None
    
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
    Extract slide name from tile filename.
    """
    if tile_name.endswith('.png') or tile_name.endswith('.PNG'):
        base_name = tile_name[:-4]
    else:
        base_name = tile_name
    
    if "__MAG_" in base_name:
        slide_id = base_name.split("__MAG_")[0]
        return slide_id
    else:
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


def classify_bbox_multimodal_llm(client, tile_image, bbox):
    """
    Classify a single cell image region using multimodal-LLM.
    """
    try:
        cropped_img = tile_image.crop(bbox)

        buffer = BytesIO()
        cropped_img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Take the role of a highly experienced histology expert. "
                        "When given an image of a cell, you must identify the correct cell type. "
                        "**Respond with exactly one label from the list and no additional explanation.** "
                        "Example labels might include: Alpha, Beta, Gamma, Delta, Duct, Acinar, Stellate, Endothelial, Immune, Undetermined. "
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
            max_tokens=50
        )

        label = response.choices[0].message.content.strip()
        label = re.sub(r'[^\w\s-]', '', label)
        if not label:
            label = "Undetermined"
        return label
    except openai.RateLimitError:
        logging.error("OpenAI rate limit exceeded. Consider adding delays or reducing request frequency.")
        return "Error - Rate Limited"
    except openai.APIError as e:
        logging.error(f"OpenAI API Error during classification: {e}")
        return f"Error - API Error: {e}"
    except Exception as e:
        logging.error(f"Unexpected error during multimodal_llm classification for bbox {bbox}: {e}")
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
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of classes.")
    parser.add_argument("--classify_per_cluster", type=int, default=10,
                        help="Number of bounding boxes to classify per cluster (per slide), from highest to lowest confidence.")
    args = parser.parse_args()

    logging.info("Starting multimodal-LLM classification with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("FATAL: No OPENAI_API_KEY found in environment!")
        raise ValueError("No OPENAI_API_KEY found in environment!")
    client = OpenAI(api_key=api_key)

    all_cell_results = []

    all_clustered_cells = load_cluster_assignments(args.clustered_cells_path)
    if all_clustered_cells:
        logging.info("Using cluster assignments to select bounding boxes.")

        logging.info(f"Loaded {len(all_clustered_cells)} total cell entries from cluster assignments.")

        cluster_map = defaultdict(list)
        for entry in all_clustered_cells:
            _, _, slide_name = get_tile_info(entry["bbox_file"], args.prepped_tiles_path)
            if not slide_name:
                 logging.warning(f"Could not determine slide name for bbox_file {entry['bbox_file']}. Skipping entry.")
                 continue
            cluster_id = entry["cluster_id"]
            entry['_slide_name_temp'] = slide_name
            cluster_map[(cluster_id, slide_name)].append(entry)

        selected_bboxes_for_classification = []
        for (cluster_id, slide_name), entries in cluster_map.items():
            sorted_entries = sorted(entries, key=lambda e: e.get("confidence", 0.0), reverse=True)
            top_n = sorted_entries[: args.classify_per_cluster]
            selected_bboxes_for_classification.extend(top_n)

        logging.info(f"Selected a total of {len(selected_bboxes_for_classification)} bounding boxes across all clusters/slides for multimodal-LLM classification.")

        ids_to_classify = set(
            (entry["bbox_file"], entry["label_id"]) for entry in selected_bboxes_for_classification
        )

        multimodal_llm_predictions = {}

        bboxes_grouped_by_tile = defaultdict(list)
        for entry in selected_bboxes_for_classification:
             tile_path, _, _ = get_tile_info(entry["bbox_file"], args.prepped_tiles_path)
             if tile_path:
                 bboxes_grouped_by_tile[tile_path].append(entry)
             else:
                 logging.warning(f"Could not get tile_path for selected bbox_file {entry['bbox_file']}. Skipping classification for this cell.")


        tile_img_cache = {}
        for tile_path, entries_in_tile in bboxes_grouped_by_tile.items():
            if not os.path.exists(tile_path):
                logging.warning(f"Tile image not found at {tile_path} (referenced by {entries_in_tile[0]['bbox_file']}). Skipping classification for {len(entries_in_tile)} cells in this tile.")
                for entry in entries_in_tile:
                     cell_id = (entry["bbox_file"], entry["label_id"])
                     multimodal_llm_predictions[cell_id] = "Error - Tile Not Found"
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

                    pred_class = classify_bbox_multimodal_llm(client, tile_img, bbox)
                    multimodal_llm_predictions[cell_id] = pred_class

            except Exception as e:
                logging.error(f"Error processing tile {tile_path}: {e}. Skipping classification for remaining cells in this tile.")
                for entry in entries_in_tile:
                     cell_id = (entry["bbox_file"], entry["label_id"])
                     if cell_id not in multimodal_llm_predictions:
                         multimodal_llm_predictions[cell_id] = "Error - Tile Processing Failed"

        logging.info("Compiling final results for all clustered cells...")
        for entry in all_clustered_cells:
            bbox_file = entry["bbox_file"]
            label_id = entry["label_id"]
            cell_id = (bbox_file, label_id)

            tile_path, tile_name, slide_name = get_tile_info(bbox_file, args.prepped_tiles_path)

            if not tile_path:
                logging.warning(f"Skipping final result for cell {label_id} from {bbox_file} due to parsing error.")
                continue

            if cell_id in multimodal_llm_predictions:
                pred_class = multimodal_llm_predictions[cell_id]
            else:
                pred_class = "Unclassified"

            all_cell_results.append({
                "bbox_file": bbox_file,
                "tile_path": tile_path,
                "slide_name": slide_name,
                "tile_name": tile_name,
                "label_id": label_id,
                "bbox": entry["bbox"],
                "cluster_id": entry.get("cluster_id", -1),
                "cluster_confidence": entry.get("confidence", 0.0),
                "pred_class": pred_class
            })
        logging.info(f"Compiled {len(all_cell_results)} total cell results including 'Unclassified'.")

    else:
        logging.info("No valid cluster assignments found or path not provided. Classifying ALL bounding boxes from segmented_path.")
        bbox_files = glob.glob(
            os.path.join(args.segmented_path, "**/*_bboxes.json"),
            recursive=True
        )
        if not bbox_files:
            logging.warning("No bounding box JSON files found under: %s", args.segmented_path)

        tile_img_cache = {}
        for bbox_file in bbox_files:
            tile_path, tile_name, slide_name = get_tile_info(bbox_file, args.prepped_tiles_path)

            if not tile_path:
                logging.warning(f"Could not determine tile path for {bbox_file}. Skipping this file.")
                continue

            if not os.path.exists(tile_path):
                logging.warning(f"Tile image not found at {tile_path} (referenced by {bbox_file}). Skipping classification for this tile.")
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

                    pred_class = classify_bbox_multimodal_llm(client, tile_img, bbox)

                    all_cell_results.append({
                        "bbox_file": bbox_file,
                        "tile_path": tile_path,
                        "slide_name": slide_name,
                        "tile_name": tile_name,
                        "label_id": label_id,
                        "bbox": bbox,
                        "cluster_id": None,
                        "cluster_confidence": None,
                        "pred_class": pred_class
                    })

            except Exception as e:
                logging.error(f"Error processing tile {tile_path}: {e}. Skipping classification for cells in this tile.")


    logging.info("Restructuring classification results...")
    grouped_by_tile = {}
    for result in all_cell_results:
        tile_path = result["tile_path"]

        if tile_path not in grouped_by_tile:
            grouped_by_tile[tile_path] = {
                "tile_path": result["tile_path"],
                "tile_name": result["tile_name"],
                "slide_name": result["slide_name"],
                "classified_cells": []
            }

        cell_data = {
            "label_id": result["label_id"],
            "bbox": result["bbox"],
            "bbox_file": result["bbox_file"],
            "pred_class": result["pred_class"]
        }
        if "cluster_id" in result and result["cluster_id"] is not None:
            cell_data["cluster_id"] = result["cluster_id"]
        if "cluster_confidence" in result and result["cluster_confidence"] is not None:
            cell_data["cluster_confidence"] = result["cluster_confidence"]

        grouped_by_tile[tile_path]["classified_cells"].append(cell_data)

    final_output_data = list(grouped_by_tile.values())
    
    slides_data = defaultdict(list)
    for tile_data in final_output_data:
        slide_name = tile_data["slide_name"]
        slides_data[slide_name].append(tile_data)
    
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