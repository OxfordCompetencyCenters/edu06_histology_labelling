import argparse
import os
import glob
import json
import numpy as np
from PIL import Image
from cellpose import models
from skimage.measure import regionprops, label
import logging

def segment_and_extract_bboxes(img_path, model, out_dir, channels):
    """
    Runs Cellpose segmentation on a single tile.
    Saves the resulting mask as a PNG or NumPy file.
    Also extracts bounding boxes for each cell for classification.
    """
    tile_name = os.path.splitext(os.path.basename(img_path))[0]
    img = np.array(Image.open(img_path))
    logging.info(f"Segmenting tile: {img_path}")

    masks, flows, styles, diams = model.eval(img, channels=channels)
    mask_img = Image.fromarray(masks.astype(np.uint16))
    mask_filename = f"{tile_name}_mask.png"
    mask_path = os.path.join(out_dir, mask_filename)
    mask_img.save(mask_path)
    logging.info(f"Saved mask to {mask_path}")

    labeled_mask = label(masks, connectivity=1, background=0)
    props = regionprops(labeled_mask)

    bboxes = []
    for prop in props:
        y1, x1, y2, x2 = prop.bbox
        bboxes.append({
            "label_id": int(prop.label),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })

    bbox_filename = f"{tile_name}_bboxes.json"
    bbox_path = os.path.join(out_dir, bbox_filename)
    with open(bbox_path, "w") as f:
        json.dump(bboxes, f, indent=2)
    logging.info(f"Saved bounding boxes to {bbox_path}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to prepped data (tiled images).")
    parser.add_argument("--output_path", type=str, help="Path for segmentation output.")
    parser.add_argument("--model_type", type=str, default="cyto2", help="Cellpose model type.")
    parser.add_argument("--chan", type=int, default=0, help="Channel for cellpose.")
    parser.add_argument("--chan2", type=int, default=0, help="Second channel for cellpose.")
    args = parser.parse_args()

    logging.info("Starting segmentation with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)
    
    logging.info(f"Initializing Cellpose model of type: {args.model_type}")
    model = models.Cellpose(model_type=args.model_type)

    channels = [args.chan, args.chan2]
    logging.info(f"Using channels: {channels}")

    tile_files = glob.glob(os.path.join(args.input_path, "**/*.png"), recursive=True)
    if not tile_files:
        logging.warning("No tile images found in the input path: %s", args.input_path)
        return

    for tile_file in tile_files:
        relative_path = os.path.relpath(tile_file, args.input_path)
        tile_out_dir = os.path.join(args.output_path, os.path.dirname(relative_path))
        os.makedirs(tile_out_dir, exist_ok=True)
        segment_and_extract_bboxes(tile_file, model, tile_out_dir, channels)

    logging.info("Segmentation step done. Output saved to: %s", args.output_path)

if __name__ == "__main__":
    main()
