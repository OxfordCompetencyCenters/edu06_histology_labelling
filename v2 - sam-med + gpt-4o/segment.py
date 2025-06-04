import argparse
import os
import glob
import json
import numpy as np
from PIL import Image
import logging

from segment_sam_med import SAMMedSegmenter

def segment_and_extract_bboxes(img_path: str, model: SAMMedSegmenter, out_dir: str):
    """
    Runs sam_med segmentation on a single tile.
    Saves the resulting mask as a PNG and extracts bounding boxes.
    
    Args:
        img_path: Path to input image tile
        model: SAMMedSegmenter instance
        out_dir: Output directory for masks and bboxes
    """
    tile_name = os.path.splitext(os.path.basename(img_path))[0]
    img = np.array(Image.open(img_path))
    logging.info(f"Segmenting tile: {img_path}")

    # Use sam_med segmentation
    logging.info("Using sam_med segmentation")
    try:
        results = model.segment_image(img_path, use_grid_prompts=True, use_adaptive_prompts=True)
        masks = results.get('combined_mask', np.zeros_like(img[:,:,0] if len(img.shape) == 3 else img))
    except Exception as e:
        logging.warning(f"sam_med segmentation failed: {e}, creating empty mask")
        masks = np.zeros_like(img[:,:,0] if len(img.shape) == 3 else img)

    # Save the mask to disk
    mask_img = Image.fromarray(masks.astype(np.uint16))
    mask_filename = f"{tile_name}_mask.png"
    mask_path = os.path.join(out_dir, mask_filename)
    mask_img.save(mask_path)
    logging.info(f"Saved mask to {mask_path}")

    # Extract bounding boxes from mask
    bboxes = []
    unique_labels = np.unique(masks)
    for lbl in unique_labels:
        if lbl == 0:
            continue  # Skip background
        coords = np.argwhere(masks == lbl)
        if coords.size == 0:
            continue
        min_row, max_row = coords[:, 0].min(), coords[:, 0].max()
        min_col, max_col = coords[:, 1].min(), coords[:, 1].max()

        bboxes.append({
            "label_id": int(lbl),
            "bbox": [int(min_col), int(min_row), int(max_col), int(max_row)]
        })

    # Save bounding boxes as JSON
    bbox_filename = f"{tile_name}_bboxes.json"
    bbox_path = os.path.join(out_dir, bbox_filename)
    with open(bbox_path, "w") as f:
        json.dump(bboxes, f, indent=2)
    logging.info(f"Saved bounding boxes to {bbox_path}")

def main():
    """Main segmentation function using sam_med."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Advanced histology segmentation using sam_med")
    parser.add_argument("--prepped_tiles_path", type=str, required=True, help="Path to preprocessed tiles")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for masks and bboxes")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_med2d_b.pth", 
                       help="SAM checkpoint name")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)

    # Initialize sam_med model
    logging.info(f"Initializing sam_med with checkpoint: {args.sam_checkpoint}")
    model = SAMMedSegmenter(checkpoint_name=args.sam_checkpoint)
    logging.info("sam_med model loaded successfully")

    # Process all slide directories
    slide_dirs = [d for d in os.listdir(args.prepped_tiles_path) 
                  if os.path.isdir(os.path.join(args.prepped_tiles_path, d))]
    
    for slide_name in slide_dirs:
        slide_input_dir = os.path.join(args.prepped_tiles_path, slide_name)
        slide_output_dir = os.path.join(args.output_path, slide_name)
        os.makedirs(slide_output_dir, exist_ok=True)
        
        # Process all PNG tiles in the slide directory
        tile_paths = glob.glob(os.path.join(slide_input_dir, "*.png"))
        logging.info(f"Processing {len(tile_paths)} tiles for slide: {slide_name}")
        
        for tile_path in tile_paths:
            try:
                segment_and_extract_bboxes(
                    img_path=tile_path,
                    model=model,
                    out_dir=slide_output_dir
                )
            except Exception as e:
                logging.error(f"Failed to process tile {tile_path}: {e}")
                continue

    logging.info("Segmentation complete!")

if __name__ == "__main__":
    main()
