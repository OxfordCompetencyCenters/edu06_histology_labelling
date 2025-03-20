import argparse
import os
import glob
import openslide
import numpy as np
from PIL import Image
import logging

def tile_slide(slide_path, out_dir, tile_size=512, overlap=0):
    """
    Opens a whole-slide image using OpenSlide, then extracts tiles of size tile_size x tile_size.
    Saves tiles as PNG images in out_dir, with a subfolder named after the slide (minus extension).
    """
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    slide_out_dir = os.path.join(out_dir, slide_name)
    os.makedirs(slide_out_dir, exist_ok=True)

    logging.debug(f"Opening slide {slide_path} with OpenSlide...")
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions
    logging.info(f"Slide {slide_path} dimensions: {width}x{height}")

    x = 0
    while x < width:
        y = 0
        while y < height:
            tile_region = slide.read_region(
                (x, y),
                0,
                (tile_size, tile_size)
            )
            tile_rgb = tile_region.convert("RGB")

            tile_filename = f"{slide_name}_x{x}_y{y}.png"
            tile_path = os.path.join(slide_out_dir, tile_filename)
            tile_rgb.save(tile_path)
            logging.debug(f"Saved tile: {tile_path}")

            y += (tile_size - overlap)
        x += (tile_size - overlap)

    slide.close()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to folder with NDPI (or SVS) data.")
    parser.add_argument("--output_path", type=str, help="Where to store prepped data (tiles).")
    parser.add_argument("--tile_size", type=int, default=512, help="Size of the square tile, default=512.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap between tiles, default=0.")
    args = parser.parse_args()

    logging.info("Starting data prep with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)

    ndpi_files = glob.glob(os.path.join(args.input_data, "**/*.ndpi"), recursive=True)
    if not ndpi_files:
        logging.warning("No .ndpi files found in the input directory: %s", args.input_data)
        return

    for slide_path in ndpi_files:
        logging.info(f"Tiling slide: {slide_path}")
        tile_slide(slide_path, args.output_path, tile_size=args.tile_size, overlap=args.overlap)

    logging.info("Data prep step done. Tiles saved to: %s", args.output_path)

if __name__ == "__main__":
    main()
