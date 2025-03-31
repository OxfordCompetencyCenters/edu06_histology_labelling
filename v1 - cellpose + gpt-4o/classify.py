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

def parse_slide_name(tile_name):
    """
    Given a tile name like:
        '202 Human Pancreas Gomori C.A.H.R - 2012-08-08_x0_y0'
    returns the slide portion:
        '202 Human Pancreas Gomori C.A.H.R - 2012-08-08'
    by removing the trailing '_x####_y####'.
    """
    return re.sub(r"_x\d+_y\d+$", "", tile_name)

def classify_bbox_gpt4o(client, tile_image, bbox):
    """
    Classify a single cell image region using GPT-4o with vision capabilities.
    Crops the given tile image to the bounding box, sends it along with label options
    to the GPT-4o model, and returns the chosen label.
    """
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
                    "Take the role of a highly experienced histopathologist expert. "
                    "When given an image of a cell, you must identify the correct cell type. "
                    "**Respond with exactly one label from the list and no additional explanation.**"
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a cell image, identify the cell type."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ]
    )
    
    # Extract the label from the model's response (first choice)
    label = response.choices[0].message.content.strip()
    return label

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--segmented_path", type=str,
                        help="Path to segmentation results (masks + *_bboxes.json).")
    parser.add_argument("--prepped_tiles_path", type=str,
                        help="Path to the original prepped tile images.")
    parser.add_argument("--clustered_cells_path", type=str, default="",
                        help="Path to cluster output (containing cluster_assignments.json).")
    parser.add_argument("--output_path", type=str,
                        help="Path for classification results.")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of classes (if you want a consistent interface).")
    parser.add_argument("--classify_per_cluster", type=int, default=10,
                        help="Number of bounding boxes to classify per cluster (per slide), from highest to lowest confidence.")
    args = parser.parse_args()

    logging.info("Starting GPT-4o classification with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OPENAI_API_KEY found in environment!")
    openai.api_key = api_key
    client = OpenAI()

    classification_results = []

    # ------------------------------------------------------------------
    # (A) If clustered_cells_path is provided, we read cluster_assignments.json
    #     and select bounding boxes from it to classify, grouping by
    #     (cluster_id, slide_name).
    # ------------------------------------------------------------------
    cluster_assignments_path = os.path.join(args.clustered_cells_path, "cluster_assignments.json")
    if args.clustered_cells_path and os.path.exists(cluster_assignments_path):
        logging.info("Using cluster assignments from %s to select bounding boxes.", cluster_assignments_path)

        with open(cluster_assignments_path, "r") as f:
            cluster_data = json.load(f)
        
        # Group bounding boxes by (cluster_id, slide_name)
        cluster_map = defaultdict(list)
        for entry in cluster_data:
            bbox_file = entry["bbox_file"]
            tile_name = os.path.basename(bbox_file).replace("_bboxes.json", "")
            slide_name = parse_slide_name(tile_name)
            cluster_id = entry["cluster_id"]
            cluster_map[(cluster_id, slide_name)].append(entry)

        # Sort each cluster+slide group by descending confidence and pick up to classify_per_cluster
        selected_bboxes = []
        for (cluster_id, slide_name), entries in cluster_map.items():
            sorted_entries = sorted(entries, key=lambda e: e["confidence"], reverse=True)
            top_n = sorted_entries[: args.classify_per_cluster]
            selected_bboxes.extend(top_n)

        logging.info(f"Selected a total of {len(selected_bboxes)} bounding boxes across all clusters/slides.")

        # We'll group them by tile_path for efficiency, so we only open each tile once.
        tile_img_cache = {}

        def get_tile_image(bbox_file):
            """
            Derives tile image path from bbox_file and loads it from prepped_tiles_path.
            E.g. if bbox_file is something like:
                /.../segment_output/subdir/tileXYZ_bboxes.json
            we'll guess tileXYZ.png is in:
                prepped_tiles_path/subdir/tileXYZ.png
            """
            tile_name_local = os.path.basename(bbox_file).replace("_bboxes.json", ".png")
            sub_dir = os.path.basename(os.path.dirname(bbox_file))
            tile_path = os.path.join(args.prepped_tiles_path, sub_dir, tile_name_local)
            if not os.path.exists(tile_path):
                logging.warning("Tile image not found at %s. Bbox file: %s", tile_path, bbox_file)
                return None
            if tile_path not in tile_img_cache:
                tile_img_cache[tile_path] = Image.open(tile_path).convert("RGB")
            return tile_img_cache[tile_path]

        for entry in selected_bboxes:
            bbox_file = entry["bbox_file"]
            label_id = entry["label_id"]
            bbox = entry["bbox"]

            tile_img = get_tile_image(bbox_file)
            if tile_img is None:
                # skip if we can't load the tile
                continue

            # Call GPT-4o for classification
            pred_class = classify_bbox_gpt4o(client, tile_img, bbox)

            classification_results.append({
                "bbox_file": bbox_file,
                "label_id": label_id,
                "bbox": bbox,
                "cluster_id": entry["cluster_id"],
                "cluster_confidence": entry["confidence"],
                "pred_class": pred_class
            })

    else:
        # ------------------------------------------------------------------
        # (B) If no cluster assignments, classify *all* bounding boxes 
        #     from segmented_path (original logic).
        # ------------------------------------------------------------------
        logging.info("No valid cluster assignments found. Classifying all bounding boxes.")
        bbox_files = glob.glob(
            os.path.join(args.segmented_path, "**/*_bboxes.json"), 
            recursive=True
        )
        if not bbox_files:
            logging.warning("No bounding box JSON files found under: %s", args.segmented_path)
            return

        for bbox_file in bbox_files:
            with open(bbox_file, "r") as f:
                bboxes = json.load(f)

            tile_name = os.path.basename(bbox_file).replace("_bboxes.json", ".png")
            parent_dir = os.path.basename(os.path.dirname(bbox_file))
            tile_path = os.path.join(args.prepped_tiles_path, parent_dir, tile_name)

            if not os.path.exists(tile_path):
                logging.warning("Tile image not found at %s. Skipping.", tile_path)
                continue

            logging.info("Classifying bboxes for tile: %s", tile_path)
            tile_img = Image.open(tile_path).convert("RGB")

            for bbox_entry in bboxes:
                label_id = bbox_entry["label_id"]
                bbox = bbox_entry["bbox"]
                pred_class = classify_bbox_gpt4o(client, tile_img, bbox)

                classification_results.append({
                    "tile_path": tile_path,
                    "label_id": label_id,
                    "bbox": bbox,
                    "pred_class": pred_class
                })

    # Write out JSON with classification results
    out_file = os.path.join(args.output_path, "classification_results.json")
    with open(out_file, "w") as f:
        json.dump(classification_results, f, indent=2)

    logging.info("Classification step complete via GPT-4o. Output: %s", out_file)

if __name__ == "__main__":
    main()
