import argparse
import os
import glob
import json
import numpy as np
from PIL import Image
from cellpose import models
import logging

def segment_and_extract_bboxes(img_path, model, out_dir, channels, flow_threshold=0.4, 
                               cellprob_threshold=0.0, diameter=None, resample=True, 
                               normalize=True, do_3D=False, stitch_threshold=0.0):
    """
    Runs Cellpose segmentation on a single tile.
    Saves the resulting mask as a PNG.
    Extracts bounding boxes directly from the mask for each label.
    """
    tile_name = os.path.splitext(os.path.basename(img_path))[0]
    img = np.array(Image.open(img_path))
    logging.info(f"Segmenting tile: {img_path} with flow_threshold={flow_threshold}, cellprob_threshold={cellprob_threshold}")

    masks, flows, diams = model.eval(
        img, 
        channels=channels,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        diameter=diameter,
        resample=resample,
        normalize=normalize,
        do_3D=do_3D,
        stitch_threshold=stitch_threshold
    )

    mask_img = Image.fromarray(masks.astype(np.uint16))
    mask_filename = f"{tile_name}_mask.png"
    mask_path = os.path.join(out_dir, mask_filename)
    mask_img.save(mask_path)
    logging.info(f"Saved mask to {mask_path}")

    bboxes = []
    unique_labels = np.unique(masks)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        coords = np.argwhere(masks == lbl)
        if coords.size == 0:
            continue
        min_row, max_row = coords[:, 0].min(), coords[:, 0].max()
        min_col, max_col = coords[:, 1].min(), coords[:, 1].max()

        bboxes.append({
            "label_id": int(lbl),
            "bbox": [int(min_col), int(min_row), int(max_col), int(max_row)]
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
    parser.add_argument("--model_type", type=str, default="cellpose_sam", 
                       help="Cellpose model type.")
    parser.add_argument("--channels", type=str, default="2,1", 
                       help="Comma-separated channel specification.")
    parser.add_argument("--flow_threshold", type=float, default=0.4, 
                       help="Flow threshold for segmentation confidence.")
    parser.add_argument("--cellprob_threshold", type=float, default=0.0,
                       help="Cell probability threshold.")
    parser.add_argument("--segment_use_gpu", action="store_true", default=False,
                       help="Use GPU for segmentation.")
    parser.add_argument("--diameter", type=float, default=None,
                       help="Expected cell diameter in pixels.")
    parser.add_argument("--resample", action="store_true", default=True,
                       help="Enable resampling for better segmentation")
    parser.add_argument("--normalize", action="store_true", default=True,
                       help="Normalize images before segmentation")
    parser.add_argument("--no_normalize", action="store_true",
                       help="Disable image normalization")
    parser.add_argument("--do_3D", action="store_true",
                       help="Enable 3D segmentation for Z-stacks")
    parser.add_argument("--stitch_threshold", type=float, default=0.0,
                       help="Threshold for stitching masks across tiles")
    args = parser.parse_args()

    logging.info("Starting segmentation with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)

    normalize = args.normalize and not args.no_normalize

    try:
        channels = [int(c) for c in args.channels.split(',')]
        if len(channels) != 2:
            raise ValueError("Must specify exactly 2 channels")
    except ValueError as e:
        logging.error(f"Invalid channel specification '{args.channels}': {e}")
        return

    logging.info(f"Initializing Cellpose model of type: {args.model_type}")
    
    try:
        if args.model_type == "cellpose_sam":
            model = models.CellposeSAM(model_type="sam_vit_b", gpu=args.segment_use_gpu)
        else:
            model = models.CellposeModel(model_type=args.model_type, gpu=args.segment_use_gpu)
        logging.info(f"Successfully loaded model: {args.model_type}")
    except Exception as e:
        logging.error(f"Failed to load model '{args.model_type}': {e}")
        logging.info("Falling back to cyto2 model")
        try:
            model = models.CellposeModel(model_type="cyto2", gpu=args.segment_use_gpu)
        except Exception as e2:
            logging.error(f"Failed to load fallback cyto2 model: {e2}")
            raise
    
    logging.info(f"Using GPU: {args.segment_use_gpu}")
    logging.info(f"Using channels: {channels}")
    logging.info(f"Normalization: {normalize}")
    logging.info(f"Diameter: {args.diameter}")
    logging.info(f"Resample: {args.resample}")
    logging.info(f"3D segmentation: {args.do_3D}")
    logging.info(f"Stitch threshold: {args.stitch_threshold}")

    tile_files = glob.glob(os.path.join(args.input_path, "**/*.png"), recursive=True)
    if not tile_files:
        logging.warning("No tile images found in the input path: %s", args.input_path)
        return

    for tile_file in tile_files:
        relative_path = os.path.relpath(tile_file, args.input_path)
        tile_out_dir = os.path.join(args.output_path, os.path.dirname(relative_path))
        os.makedirs(tile_out_dir, exist_ok=True)
        segment_and_extract_bboxes(
            tile_file, 
            model, 
            tile_out_dir, 
            channels, 
            args.flow_threshold, 
            args.cellprob_threshold,
            args.diameter,
            args.resample,
            normalize,
            args.do_3D,
            args.stitch_threshold
        )

    logging.info("Segmentation step done. Output saved to: %s", args.output_path)

if __name__ == "__main__":
    main()
