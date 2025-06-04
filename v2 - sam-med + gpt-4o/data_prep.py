import argparse
import os
import glob
import openslide
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm
import json

def calculate_tissue_ratio(tile_rgb):
    """
    Calculate the ratio of tissue to background in a tile.
    Uses simple thresholding to identify tissue areas.
    """
    # Convert to HSV for better tissue detection
    tile_hsv = Image.fromarray(tile_rgb).convert('HSV')
    hsv_array = np.array(tile_hsv)
    
    # Define tissue detection criteria (adjust as needed)
    # Typically, background is very light (high value, low saturation)
    h, s, v = hsv_array[:, :, 0], hsv_array[:, :, 1], hsv_array[:, :, 2]
    
    # Tissue typically has higher saturation and lower value than background
    tissue_mask = (s > 30) & (v < 230)  # Adjust thresholds as needed
    
    tissue_ratio = np.sum(tissue_mask) / tissue_mask.size
    return tissue_ratio

def tile_slide_multi_resolution(slide_path, out_dir, tile_sizes=[256, 512], overlap=0, target_mpp=0.5):
    """
    sam-med tiling function that creates tiles at multiple resolutions.
    Generates both 256x256 tiles for token clustering and 512x512 for traditional analysis.
    
    Args:
        slide_path: Path to the slide file
        out_dir: Output directory
        tile_sizes: List of tile sizes to generate
        overlap: Overlap between tiles
        target_mpp: Target microns per pixel for normalization
    """
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    slide_out_dir = os.path.join(out_dir, slide_name)
    os.makedirs(slide_out_dir, exist_ok=True)

    logging.debug(f"Opening slide {slide_path} with OpenSlide...")
    slide = openslide.OpenSlide(slide_path)
    
    # Get slide properties
    base_w, base_h = slide.dimensions
    logging.info(f"Slide {slide_path} base level dimensions: {base_w}x{base_h}")
    
    # Try to get microns per pixel from slide properties
    try:
        mpp_x = float(slide.properties.get('openslide.mpp-x', target_mpp))
        mpp_y = float(slide.properties.get('openslide.mpp-y', target_mpp))
        current_mpp = (mpp_x + mpp_y) / 2
        logging.info(f"Slide MPP: {current_mpp:.3f}")
    except (ValueError, TypeError):
        current_mpp = target_mpp
        logging.warning(f"Could not determine slide MPP, using target: {target_mpp}")
    
    # Calculate scaling factor to normalize to target MPP
    scale_factor = current_mpp / target_mpp
    
    level_count = slide.level_count
    downsamples = slide.level_downsamples
    
    # Find best level for target MPP
    best_level = 0
    best_level_mpp = current_mpp
    for level in range(level_count):
        level_mpp = current_mpp * downsamples[level]
        if abs(level_mpp - target_mpp) < abs(best_level_mpp - target_mpp):
            best_level = level
            best_level_mpp = level_mpp
    
    logging.info(f"Using level {best_level} with MPP {best_level_mpp:.3f} (target: {target_mpp})")
    
    # Create tiles for each specified size
    for tile_size in tile_sizes:
        create_tiles_for_size(
            slide, slide_name, slide_out_dir, 
            tile_size, overlap, best_level, 
            downsamples[best_level], scale_factor
        )
    
    # Create tissue mask and metadata
    create_tissue_mask_and_metadata(slide, slide_name, slide_out_dir, best_level)
    
    slide.close()

def create_tiles_for_size(slide, slide_name, out_dir, tile_size, overlap, level, downsample, scale_factor):
    """Create tiles for a specific size."""
    size_dir = os.path.join(out_dir, f"tiles_{tile_size}x{tile_size}")
    os.makedirs(size_dir, exist_ok=True)
    
    level_w, level_h = slide.level_dimensions[level]
    step = tile_size - overlap
    
    # Adjust step size based on scale factor if needed
    effective_step = int(step * scale_factor)
    
    steps_x = max(1, (level_w // effective_step) + (1 if level_w % effective_step != 0 else 0))
    steps_y = max(1, (level_h // effective_step) + (1 if level_h % effective_step != 0 else 0))
    total_tiles = steps_x * steps_y
    
    logging.info(f"Creating {tile_size}x{tile_size} tiles: {total_tiles} total")
    
    tile_metadata = []
    
    with tqdm(total=total_tiles, desc=f"Tiling {tile_size}x{tile_size}: {slide_name}", unit="tile") as pbar:
        x_level = 0
        tile_idx = 0
        
        while x_level < level_w:
            y_level = 0
            while y_level < level_h:
                # Base-level coordinates
                x_base = int(x_level * downsample)
                y_base = int(y_level * downsample)
                
                # Read the tile
                tile_region = slide.read_region(
                    (x_base, y_base),
                    level,
                    (tile_size, tile_size)
                )
                
                tile_rgb = tile_region.convert("RGB")
                
                # Check if tile contains tissue (not just background)
                tissue_ratio = calculate_tissue_ratio(np.array(tile_rgb))
                
                # Only save tiles with significant tissue content
                if tissue_ratio > 0.1:  # At least 10% tissue
                    tile_filename = (f"{slide_name}_L{level}_{tile_size}x{tile_size}_"
                                   f"x{x_base}_y{y_base}_idx{tile_idx:06d}.png")
                    tile_path = os.path.join(size_dir, tile_filename)
                    tile_rgb.save(tile_path)
                    
                    # Store metadata
                    tile_metadata.append({
                        "filename": tile_filename,
                        "tile_idx": tile_idx,
                        "coordinates": {
                            "x_base": x_base,
                            "y_base": y_base,
                            "x_level": x_level,
                            "y_level": y_level
                        },
                        "level": level,
                        "tile_size": tile_size,
                        "tissue_ratio": round(tissue_ratio, 3),
                        "downsample": downsample
                    })
                
                y_level += effective_step
                tile_idx += 1
                pbar.update(1)
            x_level += effective_step
    
    # Save tile metadata
    metadata_path = os.path.join(size_dir, f"{slide_name}_tiles_{tile_size}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "slide_name": slide_name,
            "tile_size": tile_size,
            "level": level,
            "downsample": downsample,
            "total_tiles": len(tile_metadata),
            "tiles": tile_metadata
        }, f, indent=2)
    
    logging.info(f"Created {len(tile_metadata)} tiles of size {tile_size}x{tile_size}")

def create_tissue_mask_and_metadata(slide, slide_name, out_dir, level):
    """Create a tissue mask and slide metadata."""
    # Create low-resolution overview for tissue detection
    overview_level = min(level + 2, slide.level_count - 1)
    overview_w, overview_h = slide.level_dimensions[overview_level]
    
    # Read overview
    overview = slide.read_region(
        (0, 0),
        overview_level,
        (overview_w, overview_h)
    ).convert("RGB")
    
    overview_array = np.array(overview)
    tissue_ratio = calculate_tissue_ratio(overview_array)
    
    # Save overview and tissue mask
    overview_path = os.path.join(out_dir, f"{slide_name}_overview.png")
    overview.save(overview_path)
    
    # Create tissue mask
    tissue_mask = calculate_tissue_ratio(overview_array) > 0.1
    tissue_mask_img = Image.fromarray((tissue_mask * 255).astype(np.uint8))
    mask_path = os.path.join(out_dir, f"{slide_name}_tissue_mask.png")
    tissue_mask_img.save(mask_path)
    
    # Save slide metadata
    metadata = {
        "slide_name": slide_name,
        "dimensions": slide.dimensions,
        "level_count": slide.level_count,
        "level_dimensions": slide.level_dimensions,
        "level_downsamples": slide.level_downsamples,
        "properties": dict(slide.properties),
        "tissue_ratio_overview": round(tissue_ratio, 3),
        "overview_level": overview_level,
        "processing_info": {
            "overview_saved": overview_path,
            "tissue_mask_saved": mask_path
        }
    }
    
    metadata_path = os.path.join(out_dir, f"{slide_name}_slide_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

# Legacy function for backward compatibility
def tile_slide(slide_path, out_dir, tile_size=512, overlap=0):
    """
    Legacy function for backward compatibility.
    Calls the sam-med multi-resolution tiling with single tile size.
    """
    tile_slide_multi_resolution(slide_path, out_dir, [tile_size], overlap)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="sam-med data preparation for histology analysis")
    parser.add_argument("--input_data", type=str, required=True,
                        help="Path to folder with NDPI (or SVS) data.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to store prepped data (tiles).")
    parser.add_argument("--tile_sizes", type=int, nargs='+', default=[256, 512],
                        help="List of tile sizes to generate (default: 256 512).")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Single tile size (legacy compatibility).")
    parser.add_argument("--overlap", type=int, default=0,
                        help="Overlap between tiles, default=0.")
    parser.add_argument("--target_mpp", type=float, default=0.5,
                        help="Target microns per pixel for normalization.")
    
    args = parser.parse_args()

    # Handle legacy single tile_size argument
    if hasattr(args, 'tile_size') and args.tile_size != 512:
        if 512 in args.tile_sizes:
            args.tile_sizes.remove(512)
        if args.tile_size not in args.tile_sizes:
            args.tile_sizes.append(args.tile_size)

    logging.info("Starting sam-med data prep with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)

    # Find slide files
    slide_extensions = ["*.ndpi", "*.svs", "*.tif", "*.tiff"]
    slide_files = []
    for ext in slide_extensions:
        slide_files.extend(glob.glob(os.path.join(args.input_data, "**", ext), recursive=True))
    
    if not slide_files:
        logging.warning(f"No slide files found in the input directory: {args.input_data}")
        return

    logging.info(f"Found {len(slide_files)} slide files")

    # Process slides
    for slide_path in tqdm(slide_files, desc="Processing slides", unit="slide"):
        logging.info(f"Tiling slide: {slide_path}")
        try:
            tile_slide_multi_resolution(
                slide_path,
                args.output_path,
                tile_sizes=args.tile_sizes,
                overlap=args.overlap,
                target_mpp=args.target_mpp
            )
        except Exception as e:
            logging.error(f"Error processing {slide_path}: {e}")
            continue

    logging.info("sam-med data prep step done. Tiles saved to: %s", args.output_path)

if __name__ == "__main__":
    main()
