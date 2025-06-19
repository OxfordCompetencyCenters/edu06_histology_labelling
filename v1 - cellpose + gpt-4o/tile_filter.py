from __future__ import annotations
import argparse, logging, os, shutil, sys
from pathlib import Path
from typing import List, Tuple
import json

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ─────────────────────── filtering algorithms ─────────────────────── #

def calculate_tile_stats(image: np.ndarray) -> dict:
    """Calculate comprehensive statistics for a tile to determine quality."""
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Basic statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # Check for empty/background regions (very bright or very dark)
    bright_pixels = np.sum(gray > 240) / gray.size
    dark_pixels = np.sum(gray < 15) / gray.size
    
    # Edge detection for texture/structure content
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Color analysis (if RGB)
    color_variance = 0
    if len(image.shape) == 3:
        # Calculate variance across color channels
        color_variance = np.var([np.mean(image[:,:,i]) for i in range(3)])
    
    # Laplacian variance for blur detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        'mean_intensity': float(mean_intensity),
        'std_intensity': float(std_intensity),
        'bright_pixels_ratio': float(bright_pixels),
        'dark_pixels_ratio': float(dark_pixels),
        'edge_density': float(edge_density),
        'color_variance': float(color_variance),
        'laplacian_var': float(laplacian_var)
    }


def is_tile_useful(
    image: np.ndarray,
    min_edge_density: float = 0.02,
    max_bright_ratio: float = 0.8,
    max_dark_ratio: float = 0.8,
    min_std_intensity: float = 10.0,
    min_laplacian_var: float = 50.0,
    min_color_variance: float = 5.0
) -> Tuple[bool, dict]:
    """
    Determine if a tile is worth processing based on multiple criteria.
    
    Args:
        image: RGB or grayscale image as numpy array
        min_edge_density: Minimum ratio of edge pixels (structure content)
        max_bright_ratio: Maximum ratio of very bright pixels (background)
        max_dark_ratio: Maximum ratio of very dark pixels (empty space)
        min_std_intensity: Minimum standard deviation of pixel intensities
        min_laplacian_var: Minimum Laplacian variance (focus/sharpness)
        min_color_variance: Minimum variance across color channels
    
    Returns:
        (is_useful, stats_dict)
    """
    stats = calculate_tile_stats(image)
    
    # Define strict criteria for useful tiles
    criteria = {
        'has_structure': stats['edge_density'] >= min_edge_density,
        'not_too_bright': stats['bright_pixels_ratio'] <= max_bright_ratio,
        'not_too_dark': stats['dark_pixels_ratio'] <= max_dark_ratio,
        'has_variation': stats['std_intensity'] >= min_std_intensity,
        'is_sharp': stats['laplacian_var'] >= min_laplacian_var,
        'has_color_content': stats['color_variance'] >= min_color_variance if len(image.shape) == 3 else True
    }
    
    # Tile is useful if it passes ALL criteria (strict filtering)
    is_useful = all(criteria.values())
    
    stats['criteria'] = criteria
    stats['is_useful'] = is_useful
    
    return is_useful, stats


# ─────────────────────── main filtering function ─────────────────────── #

def filter_tiles(
    input_path: Path,
    output_path: Path,
    min_edge_density: float = 0.02,
    max_bright_ratio: float = 0.8,
    max_dark_ratio: float = 0.8,
    min_std_intensity: float = 10.0,
    min_laplacian_var: float = 50.0,
    min_color_variance: float = 5.0,
    save_stats: bool = True
) -> None:
    """
    Filter tiles from input directory, keeping only useful ones.
    
    Args:
        input_path: Directory containing prepared tiles
        output_path: Directory to save filtered tiles
        Various filtering thresholds...
        save_stats: Whether to save filtering statistics
    """
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    all_files = []
    for ext in image_extensions:
        all_files.extend(input_path.rglob(f"*{ext}"))
    
    if not all_files:
        logging.warning("No image files found in %s", input_path)
        return
    
    logging.info("Found %d image files to filter", len(all_files))
    
    # Statistics tracking
    total_files = len(all_files)
    kept_files = 0
    filtered_stats = []
    
    # Process each file
    for img_path in tqdm(all_files, desc="Filtering tiles", unit="tile"):
        try:
            # Load image
            image = np.array(Image.open(img_path).convert('RGB'))
            
            # Check if tile is useful
            is_useful, stats = is_tile_useful(
                image,
                min_edge_density=min_edge_density,
                max_bright_ratio=max_bright_ratio,
                max_dark_ratio=max_dark_ratio,
                min_std_intensity=min_std_intensity,
                min_laplacian_var=min_laplacian_var,
                min_color_variance=min_color_variance
            )
            
            # Add file info to stats
            stats['filename'] = img_path.name
            stats['relative_path'] = str(img_path.relative_to(input_path))
            filtered_stats.append(stats)
            
            if is_useful:
                # Copy useful tile to output directory
                relative_path = img_path.relative_to(input_path)
                output_file = output_path / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, output_file)
                kept_files += 1
                
        except Exception as e:
            logging.warning("Error processing %s: %s", img_path, e)
            continue
    
    # Save filtering statistics
    if save_stats:
        stats_file = output_path / "filtering_stats.json"
        summary = {
            'total_files': total_files,
            'kept_files': kept_files,
            'filtered_out': total_files - kept_files,
            'keep_ratio': kept_files / total_files if total_files > 0 else 0,
            'filtering_params': {
                'min_edge_density': min_edge_density,
                'max_bright_ratio': max_bright_ratio,
                'max_dark_ratio': max_dark_ratio,
                'min_std_intensity': min_std_intensity,
                'min_laplacian_var': min_laplacian_var,
                'min_color_variance': min_color_variance
            },
            'detailed_stats': filtered_stats
        }
        
        with open(stats_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info("Saved filtering statistics to %s", stats_file)
    
    logging.info("Filtering complete: kept %d/%d tiles (%.1f%%)", 
                 kept_files, total_files, 100 * kept_files / total_files if total_files > 0 else 0)


# ───────────────────────────── CLI ───────────────────────────── #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tile_filter",
        description="Filter prepared tiles to remove background noise and low-quality images."
    )
    p.add_argument("--input_path", required=True,
                   help="Directory containing prepared tiles")
    p.add_argument("--output_path", required=True,
                   help="Directory to save filtered tiles")
    
    # Filtering parameters
    p.add_argument("--min_edge_density", type=float, default=0.02,
                   help="Minimum edge density (structure content) [0.02]")
    p.add_argument("--max_bright_ratio", type=float, default=0.8,
                   help="Maximum ratio of bright pixels (background) [0.8]")
    p.add_argument("--max_dark_ratio", type=float, default=0.8,
                   help="Maximum ratio of dark pixels (empty space) [0.8]")
    p.add_argument("--min_std_intensity", type=float, default=10.0,
                   help="Minimum intensity standard deviation [10.0]")
    p.add_argument("--min_laplacian_var", type=float, default=50.0,
                   help="Minimum Laplacian variance (sharpness) [50.0]")
    p.add_argument("--min_color_variance", type=float, default=5.0,
                   help="Minimum color variance across channels [5.0]")
    
    p.add_argument("--save_stats", action="store_true", default=True,
                   help="Save detailed filtering statistics")
    
    return p


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    args = build_arg_parser().parse_args(argv)
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    if not input_path.exists():
        logging.error("Input path does not exist: %s", input_path)
        sys.exit(1)
    
    logging.info("Starting tile filtering...")
    logging.info("Input: %s", input_path)
    logging.info("Output: %s", output_path)
    
    filter_tiles(
        input_path=input_path,
        output_path=output_path,
        min_edge_density=args.min_edge_density,
        max_bright_ratio=args.max_bright_ratio,
        max_dark_ratio=args.max_dark_ratio,
        min_std_intensity=args.min_std_intensity,
        min_laplacian_var=args.min_laplacian_var,
        min_color_variance=args.min_color_variance,
        save_stats=args.save_stats
    )
    
    logging.info("Tile filtering completed successfully!")


if __name__ == "__main__":
    main()
