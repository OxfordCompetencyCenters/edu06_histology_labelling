from __future__ import annotations
import argparse, logging, math, os, sys, hashlib
from pathlib import Path
from typing import List, Tuple, Generator

import openslide
from PIL import Image
from tqdm import tqdm


# ─────────────────────────── helpers ─────────────────────────── #

def parse_magnifications(mag_str: str) -> List[float]:
    factors = sorted({float(s) for s in mag_str.split(",") if s.strip()}, reverse=True)
    if not factors or any(f <= 0.0 for f in factors):
        raise argparse.ArgumentTypeError("Magnifications must be positive floats > 0.")
    return factors


def generate_slide_id(slide_name: str, replace_percent: bool = False) -> str:
    """Generate a short, filesystem-safe ID from slide name."""
    # Replace % with _percentage_ if requested
    if replace_percent:
        slide_name = slide_name.replace('%', '_percentage_')
    
    # Clean the slide name 
    clean_name = "".join(c for c in slide_name if c.isalnum() or c in " -_")
    # Take first 20 chars and add hash for uniqueness
    short_name = clean_name[:20].strip()
    hash_suffix = hashlib.md5(slide_name.encode()).hexdigest()[:8]
    return f"{short_name}_{hash_suffix}".replace(" ", "_")


def format_magnification_tag(mag: float) -> str:
    """Convert magnification to readable format: 1.0 -> 1d000, 0.7 -> 0d700."""
    int_part = int(mag)
    frac_part = int((mag - int_part) * 1000)
    return f"{int_part}d{frac_part:03d}"


def create_tile_filename(slide_id: str, mag: float, x: int, y: int, idx: int) -> str:
    """Create standardized tile filename using delimiter-based format."""
    mag_tag = format_magnification_tag(mag)
    # Format: slideID__MAG_magvalue__X_xvalue__Y_yvalue__IDX_idxvalue.png
    return f"{slide_id}__MAG_{mag_tag}__X_{x}__Y_{y}__IDX_{idx:06d}.png"


def compute_step_multiplier(total_tiles: int, requested: int | None) -> int:
    if requested is None or requested >= total_tiles:
        return 1
    return math.ceil(math.sqrt(total_tiles / requested))


def iter_grid_positions(w: int, h: int, edge: int, stride: int) -> Generator[Tuple[int, int], None, None]:
    max_x, max_y = w - edge, h - edge
    for y in range(0, max_y + 1, stride):
        for x in range(0, max_x + 1, stride):
            yield x, y


# ─────────────────────── core tiling routine ─────────────────────── #

def tile_slide(
    slide_path: Path,
    out_root: Path,
    tile_size: int = 512,
    magnifications: List[float] | None = None,
    num_tiles: int | None = None,
    replace_percent: bool = False,
) -> None:

    magnifications = magnifications or [1.0]
    slide_name = slide_path.stem
    slide_id = generate_slide_id(slide_name, replace_percent)  # Generate filesystem-safe ID
    out_dir = out_root / slide_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("→ Opening slide: %s", slide_path)
    logging.info("→ Slide ID: %s", slide_id)
    slide = openslide.OpenSlide(str(slide_path))
    base_w, base_h = slide.dimensions

    tile_idx = 0  # Global tile counter for unique IDs

    for mag in magnifications:
        if mag <= 1.0:
            # Original logic for magnifications <= 1.0 (downsampling)
            src_edge = int(round(tile_size / mag))
            if base_w < src_edge or base_h < src_edge:
                logging.info("   skipping mag %.3f: slide < %d px at this scale", mag, src_edge)
                continue
        else:
            # For magnifications > 1.0 (upsampling), we extract at tile_size and enlarge
            src_edge = tile_size
            if base_w < src_edge or base_h < src_edge:
                logging.info("   skipping mag %.3f: slide < %d px", mag, src_edge)
                continue

        tiles_x = (base_w - src_edge) // src_edge + 1
        tiles_y = (base_h - src_edge) // src_edge + 1
        stride = src_edge * compute_step_multiplier(tiles_x * tiles_y, num_tiles)

        est_tiles = ((base_w - src_edge) // stride + 1) * (
            (base_h - src_edge) // stride + 1
        )

        with tqdm(total=est_tiles, desc=f"{slide_name} | mag {mag}", unit="tile",
                  leave=False) as pbar:
            for x, y in iter_grid_positions(base_w, base_h, src_edge, stride):
                region = slide.read_region((x, y), 0, (src_edge, src_edge)).convert("RGB")
                
                if mag <= 1.0:
                    # Downsampling: resize extracted region to tile_size
                    if src_edge != tile_size:
                        region = region.resize((tile_size, tile_size), Image.LANCZOS)
                else:
                    # Upsampling: enlarge the tile_size region by the magnification factor
                    enlarged_size = int(tile_size * mag)
                    region = region.resize((enlarged_size, enlarged_size), Image.LANCZOS)
                    # Then crop back to tile_size from the center to maintain aspect ratio
                    left = (enlarged_size - tile_size) // 2
                    top = (enlarged_size - tile_size) // 2
                    region = region.crop((left, top, left + tile_size, top + tile_size))

                out_name = create_tile_filename(slide_id, mag, x, y, tile_idx)
                region.save(out_dir / out_name)
                tile_idx += 1
                pbar.update(1)

    # thumbnail
    thumb_lvl = slide.level_count - 1
    thumb = slide.read_region((0, 0), thumb_lvl, slide.level_dimensions[thumb_lvl]) \
                 .convert("RGB").resize((tile_size, tile_size), Image.BICUBIC)
    thumb.save(out_dir / f"{slide_id}__THUMBNAIL.png")

    slide.close()
    logging.info("✓ Finished %s (ID: %s)", slide_name, slide_id)


# ───────────────────────────── CLI ───────────────────────────── #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wsi_tiler",
        description="Tile NDPI/SVS images into 512×512 PNG patches."
    )
    p.add_argument("--input_data", required=True,
                   help="Directory containing .ndpi/.svs files (recursively).")
    p.add_argument("--output_path", required=True,
                   help="Directory where tiles will be written.")
    p.add_argument("--tile_size", type=int, default=512,
                   help="Output tile edge (default 512).")
    p.add_argument("--magnifications", type=parse_magnifications, default="1.0",
                   help="Comma-separated factors, e.g. '1.0,0.9,0.8' (≤1.0 for downsampling) or '1.0,1.2,1.5' (>1.0 for upsampling).")
    p.add_argument("--num_tiles", type=int, default=None,
                   help="Target #tiles per magnification (grid is thinned).")
    p.add_argument("--replace_percent_in_names", action="store_true",
                   help="Replace '%' characters in slide names with '_percentage_' for filesystem compatibility.")
    return p


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    args = build_arg_parser().parse_args(argv)
    in_dir, out_dir = Path(args.input_data), Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    wsi_paths = list(in_dir.rglob("*.ndpi")) + list(in_dir.rglob("*.svs"))
    if not wsi_paths:
        logging.error("No slides found under %s", in_dir)
        sys.exit(1)

    logging.info("Found %d slide(s).", len(wsi_paths))
    for slide_path in tqdm(wsi_paths, desc="Slides", unit="slide"):
        tile_slide(slide_path, out_dir, args.tile_size,
                   args.magnifications, args.num_tiles, args.replace_percent_in_names)

    logging.info("All done – tiles are in %s", out_dir)


if __name__ == "__main__":
    main()
