import argparse
import os
import glob
import openslide
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm

def tile_slide(slide_path, out_dir, tile_size=512, overlap=0):
    """
    For each level in the slide's pyramid, tiles the region at that level.
    Also creates a single 512x512 tile covering the entire slide (lowest mag).
    """
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    slide_out_dir = os.path.join(out_dir, slide_name)
    os.makedirs(slide_out_dir, exist_ok=True)

    logging.debug(f"Opening slide {slide_path} with OpenSlide...")
    slide = openslide.OpenSlide(slide_path)
    base_w, base_h = slide.dimensions
    logging.info(f"Slide {slide_path} base level dimensions: {base_w}x{base_h}")

    level_count = slide.level_count
    downsamples = slide.level_downsamples  # e.g. [1.0, 4.0, 16.0, ...]

    # Loop over each level in the pyramid
    for level in range(level_count):
        level_w, level_h = slide.level_dimensions[level]
        # Step size for tiling at this level
        step = tile_size - overlap

        steps_x = (level_w // step) + (1 if level_w % step != 0 else 0)
        steps_y = (level_h // step) + (1 if level_h % step != 0 else 0)
        total_tiles = steps_x * steps_y

        logging.info(f"Level {level}: dimensions={level_w}x{level_h}, "
                     f"downsample={downsamples[level]}, total_tiles={total_tiles}")

        with tqdm(total=total_tiles, desc=f"Tiling L{level}: {slide_name}", unit="tile") as pbar:
            x_level = 0
            while x_level < level_w:
                y_level = 0
                while y_level < level_h:
                    # Base-level coordinates (for traceability)
                    x_base = int(x_level * downsamples[level])
                    y_base = int(y_level * downsamples[level])

                    # Read the tile from the given level
                    tile_region = slide.read_region(
                        (x_base, y_base),  # base-level coordinate
                        level,
                        (tile_size, tile_size)
                    )

                    tile_rgb = tile_region.convert("RGB")
                    tile_filename = (f"{slide_name}_level{level}_"
                                     f"x{x_base}_y{y_base}.png")
                    tile_path = os.path.join(slide_out_dir, tile_filename)
                    tile_rgb.save(tile_path)

                    y_level += step
                    pbar.update(1)
                x_level += step

    # Create a single tile covering the entire slide (lowest magnification).
    # You can do this from base level and downsize to 512x512,
    # ensuring consistent color/resolution handling.
    logging.info("Creating a single 512x512 tile covering the entire slide (lowest magnification).")
    full_slide_region = slide.read_region(
        (0, 0),
        0,
        (base_w, base_h)
    ).convert("RGB")

    # Resize down to 512x512
    full_slide_tile = full_slide_region.resize((512, 512), Image.BICUBIC)
    # Name this special tile something like 'levelX_fullslide'
    single_tile_path = os.path.join(slide_out_dir,
                                    f"{slide_name}_level{level_count}_fullslide.png")
    full_slide_tile.save(single_tile_path)

    slide.close()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str,
                        help="Path to folder with NDPI (or SVS) data.")
    parser.add_argument("--output_path", type=str,
                        help="Where to store prepped data (tiles).")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Size of the square tile, default=512.")
    parser.add_argument("--overlap", type=int, default=0,
                        help="Overlap between tiles, default=0.")
    args = parser.parse_args()

    logging.info("Starting data prep with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)

    ndpi_files = glob.glob(os.path.join(args.input_data, "**/*.ndpi"), recursive=True)
    if not ndpi_files:
        logging.warning("No .ndpi files found in the input directory: %s", args.input_data)
        return

    # Show progress bar for processing slides
    for slide_path in tqdm(ndpi_files, desc="Processing slides", unit="slide"):
        logging.info(f"Tiling slide: {slide_path}")
        tile_slide(
            slide_path,
            args.output_path,
            tile_size=args.tile_size,
            overlap=args.overlap
        )

    logging.info("Data prep step done. Tiles saved to: %s", args.output_path)

if __name__ == "__main__":
    main()
