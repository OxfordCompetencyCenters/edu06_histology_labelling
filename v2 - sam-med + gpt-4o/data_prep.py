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

def tile_slide_multi_resolution(slide_path, out_dir, tile_sizes=[256, 512], overlap=0, target_mpp=0.5, num_tiles=None):
    """
    sam_med tiling function that creates tiles at the highest resolution level.
    Generates tiles at specified sizes.
    
    Args:
        slide_path: Path to the slide file
        out_dir: Output directory
        tile_sizes: List of tile sizes to generate
        overlap: Overlap between tiles
        target_mpp: Target microns per pixel for normalization
        num_tiles: If provided, generate this many tiles in a uniform grid pattern
                  (from top-right to bottom-left) instead of tiling the whole slide.
                  If this exceeds the maximum possible non-overlapping tiles, will use the maximum.
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
    
    # Using highest resolution level
    # Calculate native MPP at highest resolution
    mpp_at_highest_res = current_mpp
    logging.info(f"Using highest resolution with MPP {mpp_at_highest_res:.3f} (target was {target_mpp})")
    
    # Create tiles for each specified size
    for tile_size in tile_sizes:
        create_tiles_for_size(
            slide, slide_name, slide_out_dir, 
            tile_size, overlap,
            scale_factor,
            num_tiles
        )
    
    # Create tissue mask and metadata
    create_tissue_mask_and_metadata(slide, slide_name, slide_out_dir)
    
    slide.close()

def create_tiles_for_size(slide, slide_name, out_dir, tile_size, overlap, scale_factor, num_tiles=None):
    """Create tiles for a specific size."""
    # Using the highest resolution level (level 0)
    # For highest resolution, downsample is always 1.0
    downsample = 1.0
    # Using highest resolution level (level 0)
    level = 0
    size_dir = os.path.join(out_dir, f"tiles_{tile_size}x{tile_size}")
    os.makedirs(size_dir, exist_ok=True)
    
    level_w, level_h = slide.level_dimensions[level]
    step = tile_size - overlap
    
    # Adjust step size based on scale factor if needed
    effective_step = int(step * scale_factor)
    
    steps_x = max(1, (level_w // effective_step) + (1 if level_w % effective_step != 0 else 0))
    steps_y = max(1, (level_h // effective_step) + (1 if level_h % effective_step != 0 else 0))
    total_tiles = steps_x * steps_y        # Generate uniform tiles if requested
    if num_tiles is not None:
        # If num_tiles exceeds total possible non-overlapping tiles, cap it
        max_possible_tiles = total_tiles
        tiles_to_generate = min(num_tiles, max_possible_tiles)
        
        logging.info(f"Creating {tiles_to_generate} uniform {tile_size}x{tile_size} tiles")
        
        tile_metadata = []
        tile_idx = 0
        
        # Calculate grid dimensions to distribute tiles uniformly
        from math import sqrt, ceil
        
        # Determine grid dimensions based on aspect ratio
        aspect_ratio = level_w / level_h
        grid_h = ceil(sqrt(tiles_to_generate / aspect_ratio))
        grid_w = ceil(tiles_to_generate / grid_h)
        
        # Recalculate to handle rounding up
        if grid_w * grid_h > tiles_to_generate:
            grid_actual_tiles = min(grid_w * grid_h, max_possible_tiles)
        else:
            grid_actual_tiles = tiles_to_generate
        
        logging.info(f"Creating a {grid_w}x{grid_h} grid of tiles (max {grid_actual_tiles} tiles)")
        
        # Calculate spacing between tile starting positions
        x_spacing = (level_w - tile_size) / (grid_w - 1) if grid_w > 1 else 0
        y_spacing = (level_h - tile_size) / (grid_h - 1) if grid_h > 1 else 0
        
        # Generate tiles in a grid pattern - starting from top-right moving to bottom-left
        tiles_generated = 0
        
        with tqdm(total=grid_actual_tiles, desc=f"Tiling {tile_size}x{tile_size}: {slide_name} (uniform)", unit="tile") as pbar:
            for i in range(grid_w):
                for j in range(grid_h):
                    if tiles_generated >= grid_actual_tiles:
                        break
                        
                    # Calculate starting position
                    if grid_w > 1:
                        # Start from right (i=0) to left (i=grid_w-1)
                        x_level = int((level_w - tile_size) - (i * x_spacing))
                        # Ensure x is within bounds
                        x_level = max(0, min(level_w - tile_size, x_level))
                        # Align to step size grid
                        x_level = (x_level // effective_step) * effective_step
                    else:
                        # Center horizontally if just one column
                        x_level = ((level_w - tile_size) // 2) // effective_step * effective_step
                    
                    if grid_h > 1:
                        # Start from top (j=0) to bottom (j=grid_h-1)
                        y_level = int(j * y_spacing)
                        # Ensure y is within bounds
                        y_level = max(0, min(level_h - tile_size, y_level))
                        # Align to step size grid
                        y_level = (y_level // effective_step) * effective_step
                    else:
                        # Center vertically if just one row
                        y_level = ((level_h - tile_size) // 2) // effective_step * effective_step
                    
                    # Base-level coordinates (for traceability)
                    x_base = int(x_level * downsample)
                    y_base = int(y_level * downsample)
                    
                    # Read the tile from the given level
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
                        tile_idx += 1
                    
                    tiles_generated += 1
                    pbar.update(1)
            
            logging.info(f"Generated {len(tile_metadata)} uniform grid tiles with sufficient tissue for level {level}")
                
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
                    tile_idx += 1
                    pbar.update(1)

    else:
        # Original tiling logic for the whole slide
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

def create_tissue_mask_and_metadata(slide, slide_name, out_dir):
    # For highest resolution, downsample is always 1.0
    downsample = 1.0
    # Using highest resolution level (level 0)
    level = 0
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
def tile_slide(slide_path, out_dir, tile_size=512, overlap=0, num_random_tiles=None):
    """
    Legacy function for backward compatibility.
    Calls the sam_med multi-resolution tiling with single tile size.
    
    Args:
        slide_path: Path to the slide file
        out_dir: Output directory
        tile_size: Size of the square tile
        overlap: Overlap between tiles
        num_random_tiles: If provided, generate this many random tiles per level instead of tiling the whole slide
    """
    tile_slide_multi_resolution(slide_path, out_dir, [tile_size], overlap, num_random_tiles=num_random_tiles)

def main():
    """
    Main function to parse arguments and execute the tiling process.
    
    Note on performance optimization:
    - The tiling process is primarily I/O bound rather than computation-bound,
      meaning that GPU acceleration won't benefit the basic tiling.
    - However, if additional image processing is needed (like tissue detection,
      normalization, or augmentations), these could be GPU-accelerated using libraries
      like torch, CuPy, or cuCIM (CUDA Image).
    - For large slides, parallel processing of multiple slides simultaneously could
      provide more benefit than GPU acceleration of a single slide.
    - The SAM-MED2D segmentation step (in segment_sam_med.py) already uses GPU
      acceleration when available and will benefit most from GPU resources.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="sam_med data preparation for histology analysis")
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
    parser.add_argument("--num_tiles", type=int, default=None,
                        help="If provided, generate this many tiles in a uniform grid pattern across the slide instead of tiling the whole slide.")
    
    args = parser.parse_args()

    # Handle legacy single tile_size argument
    if hasattr(args, 'tile_size') and args.tile_size != 512:
        if 512 in args.tile_sizes:
            args.tile_sizes.remove(512)
        if args.tile_size not in args.tile_sizes:
            args.tile_sizes.append(args.tile_size)

    logging.info("Starting sam_med data prep with arguments: %s", args)
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
                target_mpp=args.target_mpp,
                num_tiles=args.num_tiles
            )
        except Exception as e:
            logging.error(f"Error processing {slide_path}: {e}")
            continue

    logging.info("sam_med data prep step done. Tiles saved to: %s", args.output_path)

if __name__ == "__main__":
    main()
