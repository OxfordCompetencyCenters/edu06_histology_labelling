import argparse
import os
import glob
import json
import logging
import base64

from PIL import Image
from io import BytesIO

import openai

def classify_bbox_gpt4o(tile_image, bbox):
    """
    Classify a single cell image region using GPT-4o with vision capabilities.
    Crops the given tile image to the bounding box, sends it along with label options
    to the GPT-4o model, and returns the chosen label.
    """
    # Crop the tile image to the specified bounding box
    cropped_img = tile_image.crop(bbox)
    
    # Save the cropped image to an in-memory binary (bytes) object
    buffer = BytesIO()
    cropped_img.save(buffer, format="PNG")  # Using PNG format; JPEG is also acceptable
    image_bytes = buffer.getvalue()
    
    # Construct the system and user messages for the GPT-4o model
    system_message = {
        "role": "system",
        "content": (
            "Take the role of a highly experienced histopathologist expert. "
            "When given an image of a cell, you must identify the correct cell type. "
            "**Respond with exactly one label from the list and no additional explanation.**"
        )
    }
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": f"Here is a cell image, identify the cell type."
            },
            {
                "type": "input_image",
                "image": image_bytes
            }
        ]
    }
    
    # Call the GPT-4o ChatCompletion API with the image and prompt
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # GPT-4 model with vision capabilities
        messages=[system_message, user_message]
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
    parser.add_argument("--output_path", type=str,
                        help="Path for classification results.")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of classes (if you want a consistent interface).")
    args = parser.parse_args()

    logging.info("Starting GPT-4 classification with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)

    # Gather bounding box files
    bbox_files = glob.glob(
        os.path.join(args.segmented_path, "**/*_bboxes.json"), 
        recursive=True
    )
    if not bbox_files:
        logging.warning("No bounding box JSON files found under: %s", args.segmented_path)
        return

    classification_results = []

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

        tile_results = []
        for bbox_entry in bboxes:
            label_id = bbox_entry["label_id"]
            bbox = bbox_entry["bbox"]

            # Call GPT-4o for classification
            pred_class = classify_bbox_gpt4(tile_img, bbox)

            tile_results.append({
                "label_id": label_id,
                "bbox": bbox,
                "pred_class": pred_class
            })

        classification_results.append({
            "tile_path": tile_path,
            "classified_cells": tile_results
        })

    # Write out JSON with classification results
    out_file = os.path.join(args.output_path, "classification_results.json")
    with open(out_file, "w") as f:
        json.dump(classification_results, f, indent=2)

    logging.info("Classification step complete via GPT-4o. Output: %s", out_file)


if __name__ == "__main__":
    main()
