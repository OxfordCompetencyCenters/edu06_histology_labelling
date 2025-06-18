from __future__ import annotations
import argparse, logging, math, os, sys
from pathlib import Path
from typing import List, Tuple

import openslide
from PIL import Image
from tqdm import tqdm


# ─────────────────────────── helpers ─────────────────────────── #

def parse_magnifications(mag_str: str) -> List[float]:
    factors = sorted({float(s) for s in mag_str.split(",") if s.strip()}, reverse=True)
    if not factors or any(not 0.0 < f <= 1.0 for f in factors):
        raise argparse.ArgumentTypeError("Magnifications must be floats in (0,1].")
    return factors


def compute_step_multiplier(total_tiles: int, requested: int | None) -> int:
    if requested is None or requested >= total_tiles:
        return 1
    return math.ceil(math.sqrt(total_tiles / requested))


def iter_grid_positions(w: int, h: int, edge: int, stride: int) -> Tuple[int, int]:
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
) -> None:

    magnifications = magnifications or [1.0]
    slide_name = slide_path.stem
    out_dir = out_root / slide_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("→ Opening slide: %s", slide_path)
    slide = openslide.OpenSlide(str(slide_path))
    base_w, base_h = slide.dimensions

    for mag in magnifications:
        src_edge = int(round(tile_size / mag))
        if base_w < src_edge or base_h < src_edge:
            logging.info("   skipping mag %.3f: slide < 512 px at this scale", mag)
            break

        tiles_x = (base_w - src_edge) // src_edge + 1
        tiles_y = (base_h - src_edge) // src_edge + 1
        stride = src_edge * compute_step_multiplier(tiles_x * tiles_y, num_tiles)

        est_tiles = ((base_w - src_edge) // stride + 1) * (
            (base_h - src_edge) // stride + 1
        )
        mag_tag = f"{mag:.3f}".replace(".", "p")       # ← **dot → 'p'**

        with tqdm(total=est_tiles, desc=f"{slide_name} | mag {mag}", unit="tile",
                  leave=False) as pbar:
            for x, y in iter_grid_positions(base_w, base_h, src_edge, stride):
                region = slide.read_region((x, y), 0, (src_edge, src_edge)).convert("RGB")
                if src_edge != tile_size:
                    region = region.resize((tile_size, tile_size), Image.LANCZOS)

                out_name = f"{slide_name}_mag{mag_tag}_x{x}_y{y}.png"
                region.save(out_dir / out_name)
                pbar.update(1)

    # thumbnail
    thumb_lvl = slide.level_count - 1
    thumb = slide.read_region((0, 0), thumb_lvl, slide.level_dimensions[thumb_lvl]) \
                 .convert("RGB").resize((tile_size, tile_size), Image.BICUBIC)
    thumb.save(out_dir / f"{slide_name}_thumbnail.png")

    slide.close()
    logging.info("✓ Finished %s", slide_name)


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
                   help="Comma-separated factors, e.g. '1.0,0.9,0.8'.")
    p.add_argument("--num_tiles", type=int, default=None,
                   help="Target #tiles per magnification (grid is thinned).")
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
                   args.magnifications, args.num_tiles)

    logging.info("All done – tiles are in %s", out_dir)


if __name__ == "__main__":
    main()
