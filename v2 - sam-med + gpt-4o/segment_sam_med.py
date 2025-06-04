import argparse
import os
import glob
import json
import numpy as np
from PIL import Image
import cv2
import logging
from typing import List, Tuple, Dict
import random

try:
    # Import SAM-Med components
    from segment_anything import SamPredictor, sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    logging.warning("SAM not available. Install segment-anything to use SAM-Med features.")

class SAMMedSegmentor:
    """SAM-Med based segmentation with prompt generation for histology images."""
    
    def __init__(self, model_type="vit_h", checkpoint_path=None, device="cuda"):
        self.device = device
        self.model_type = model_type
        
        if not HAS_SAM:
            raise ImportError("segment-anything is required for SAM-Med segmentation")
            
        # Load SAM model
        if checkpoint_path is None:
            # You'll need to download SAM-Med weights or use regular SAM
            checkpoint_path = "sam_vit_h_4b8939.pth"  # Default SAM checkpoint
            
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)
        self.predictor = SamPredictor(self.sam)
        
        # Transform for preprocessing
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        
    def generate_grid_prompts(self, image_shape: Tuple[int, int], grid_size: int = 16) -> List[Tuple[int, int]]:
        """Generate grid-based point prompts for comprehensive segmentation."""
        h, w = image_shape[:2]
        points = []
        
        step_y = h // grid_size
        step_x = w // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                y = i * step_y + step_y // 2
                x = j * step_x + step_x // 2
                if y < h and x < w:
                    points.append((x, y))
        
        return points
    
    def generate_adaptive_prompts(self, image: np.ndarray, num_points: int = 50) -> List[Tuple[int, int]]:
        """Generate adaptive point prompts based on image features."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use Harris corner detection to find interesting points
        corners = cv2.goodFeaturesToTrack(
            gray, 
            maxCorners=num_points,
            qualityLevel=0.01,
            minDistance=10
        )
        
        points = []
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel().astype(int)
                points.append((x, y))
        
        # Add some random points if we don't have enough
        h, w = image.shape[:2]
        while len(points) < num_points:
            x = random.randint(10, w - 10)
            y = random.randint(10, h - 10)
            points.append((x, y))
            
        return points[:num_points]
    
    def segment_with_prompts(self, image: np.ndarray, point_prompts: List[Tuple[int, int]]) -> np.ndarray:
        """Segment image using point prompts."""
        self.predictor.set_image(image)
        
        # Convert points to numpy array
        input_points = np.array(point_prompts)
        input_labels = np.ones(len(input_points))  # All positive prompts
        
        # Predict masks
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        # Select best mask based on score
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx]
    
    def segment_tile_comprehensive(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Comprehensive segmentation of a tile using multiple prompt strategies.
        Returns combined mask and bounding boxes.
        """
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint16)
        bboxes = []
        current_label = 1
        
        # Strategy 1: Grid-based prompts
        grid_points = self.generate_grid_prompts(image.shape, grid_size=8)
        
        # Strategy 2: Adaptive prompts
        adaptive_points = self.generate_adaptive_prompts(image, num_points=30)
        
        # Combine all prompts
        all_prompts = grid_points + adaptive_points
        
        # Segment with batched prompts (process in chunks to avoid memory issues)
        chunk_size = 20
        for i in range(0, len(all_prompts), chunk_size):
            chunk_prompts = all_prompts[i:i+chunk_size]
            
            try:
                mask = self.segment_with_prompts(image, chunk_prompts)
                
                # Post-process mask to get individual objects
                # Use connected components to separate objects
                num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
                
                for label_id in range(1, num_labels):  # Skip background (0)
                    object_mask = (labels == label_id)
                    
                    # Filter small objects
                    if np.sum(object_mask) < 100:  # Minimum 100 pixels
                        continue
                    
                    # Add to combined mask
                    combined_mask[object_mask] = current_label
                    
                    # Extract bounding box
                    coords = np.argwhere(object_mask)
                    if coords.size > 0:
                        min_row, max_row = coords[:, 0].min(), coords[:, 0].max()
                        min_col, max_col = coords[:, 1].min(), coords[:, 1].max()
                        
                        bboxes.append({
                            "label_id": int(current_label),
                            "bbox": [int(min_col), int(min_row), int(max_col), int(max_row)],
                            "area": int(np.sum(object_mask)),
                            "segmentation_method": "sam_med"
                        })
                    
                    current_label += 1
                    
            except Exception as e:
                logging.warning(f"Error processing prompt chunk {i//chunk_size}: {e}")
                continue
        
        return combined_mask, bboxes

def segment_and_extract_sam_med(img_path: str, segmentor: SAMMedSegmentor, out_dir: str) -> None:
    """
    Run SAM-Med segmentation on a single tile and extract bounding boxes.
    """
    tile_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Load image
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)
    
    logging.info(f"Segmenting tile with SAM-Med: {img_path}")
    
    try:
        # Segment the image
        mask, bboxes = segmentor.segment_tile_comprehensive(img_np)
        
        # Save the mask
        mask_img = Image.fromarray(mask.astype(np.uint16))
        mask_filename = f"{tile_name}_sam_mask.png"
        mask_path = os.path.join(out_dir, mask_filename)
        mask_img.save(mask_path)
        logging.info(f"Saved SAM-Med mask to {mask_path}")
        
        # Save bounding boxes
        bbox_filename = f"{tile_name}_sam_bboxes.json"
        bbox_path = os.path.join(out_dir, bbox_filename)
        with open(bbox_path, "w") as f:
            json.dump(bboxes, f, indent=2)
        logging.info(f"Saved {len(bboxes)} bounding boxes to {bbox_path}")
        
    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser(description="SAM-Med based segmentation for histology tiles")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to prepped data (tiled images).")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path for segmentation output.")
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                       help="Path to SAM checkpoint file.")
    parser.add_argument("--model_type", type=str, default="vit_h",
                       choices=["vit_h", "vit_l", "vit_b"],
                       help="SAM model type.")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu).")
    
    args = parser.parse_args()
    
    if not HAS_SAM:
        logging.error("segment-anything is required but not installed. Please install it first.")
        return
    
    logging.info("Starting SAM-Med segmentation with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize segmentor
    try:
        segmentor = SAMMedSegmentor(
            model_type=args.model_type,
            checkpoint_path=args.sam_checkpoint,
            device=args.device
        )
        logging.info(f"Initialized SAM-Med segmentor with model type: {args.model_type}")
    except Exception as e:
        logging.error(f"Failed to initialize SAM-Med segmentor: {e}")
        return
    
    # Find all tile images
    tile_files = glob.glob(os.path.join(args.input_path, "**/*.png"), recursive=True)
    if not tile_files:
        logging.warning("No tile images found in the input path: %s", args.input_path)
        return
    
    logging.info(f"Found {len(tile_files)} tiles to process")
    
    # Process each tile
    for tile_file in tile_files:
        relative_path = os.path.relpath(tile_file, args.input_path)
        tile_out_dir = os.path.join(args.output_path, os.path.dirname(relative_path))
        os.makedirs(tile_out_dir, exist_ok=True)
        
        segment_and_extract_sam_med(tile_file, segmentor, tile_out_dir)
    
    logging.info("SAM-Med segmentation completed. Output saved to: %s", args.output_path)

if __name__ == "__main__":
    main()
