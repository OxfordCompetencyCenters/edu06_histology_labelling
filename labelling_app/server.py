"""
Histology Cell Labelling App (FastAPI)

Usage (local):
    pip install fastapi uvicorn
    python server.py

Then open http://localhost:8000 in your browser.
"""

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

app = FastAPI()

# Serve the HTML template directly
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _find_batches(directory):
    """Return sorted list of batch_* directory names inside *directory*."""
    p = Path(directory)
    if not p.is_dir():
        return []
    return sorted(
        d.name for d in p.iterdir() if d.is_dir() and d.name.startswith("batch_")
    )


def find_data_dir(parent_path):
    """Locate the directory containing ``batch_*`` subdirectories.

    Searches directly in *parent_path* and one level deep.

    Returns ``(data_dir, [batch_names])`` or ``(None, [])``.
    """
    parent = Path(parent_path)
    if not parent.exists() or not parent.is_dir():
        return None, []

    # Batch dirs directly in parent
    batches = _find_batches(parent)
    if batches:
        return str(parent), batches

    # One level deep (e.g. parent/qa_sample/batch_1/)
    for sub in sorted(parent.iterdir()):
        if sub.is_dir():
            batches = _find_batches(sub)
            if batches:
                return str(sub), batches

    return None, []


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------


class BrowseRequest(BaseModel):
    path: str


class ScanRequest(BaseModel):
    path: str


class SaveTileLabels(BaseModel):
    tile_index: int
    slide_id: str
    tile_name: str
    total_cells: int
    labelled_cells: list[dict]


class SaveRequest(BaseModel):
    data_dir: str
    batch: str
    tiles: list[SaveTileLabels]


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = _TEMPLATE_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/browse")
def browse(body: BrowseRequest):
    """List subdirectories at the given path."""
    raw = body.path.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="No path provided")

    target = Path(raw).resolve()
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    entries: list[dict] = []

    for child in sorted(target.iterdir()):
        if child.is_dir():
            has_batches = any(
                d.name.startswith("batch_") for d in child.iterdir() if d.is_dir()
            )
            entries.append(
                {"name": child.name, "type": "dir", "has_batches": has_batches}
            )

    return {
        "path": str(target),
        "entries": entries,
        "has_batches": any(
            d.name.startswith("batch_") for d in target.iterdir() if d.is_dir()
        ),
    }


@app.post("/api/scan")
def scan(body: ScanRequest):
    parent_path = body.path.strip()
    if not parent_path:
        raise HTTPException(status_code=400, detail="No path provided")

    data_dir, batches = find_data_dir(parent_path)
    if not data_dir or not batches:
        raise HTTPException(
            status_code=404,
            detail="No batch directories found at the given path",
        )

    return {"data_dir": data_dir, "batches": batches}


@app.get("/api/tiles")
def get_tiles(data_dir: str = Query(...), batch: str = Query(...)):
    if not data_dir or not batch:
        raise HTTPException(
            status_code=400, detail="Missing data_dir or batch parameter"
        )

    # Sanitise batch name to prevent traversal
    batch = os.path.basename(batch)
    labels_path = os.path.join(data_dir, batch, "cell_labels.json")
    if not os.path.isfile(labels_path):
        raise HTTPException(
            status_code=404,
            detail=f"cell_labels.json not found in {batch}",
        )

    with open(labels_path, "r") as f:
        cell_data = json.load(f)

    # Attach the expected image filename for each tile (matches the
    # naming convention used by sample_patches_for_qa.ipynb).
    for tile in cell_data.get("tiles", []):
        img_name = f"{tile['slide_id']}__{tile['tile_name']}"
        if not img_name.endswith(".png"):
            img_name += ".png"
        tile["image_filename"] = img_name

    return JSONResponse(content=cell_data)


@app.get("/api/image")
def serve_image(
    data_dir: str = Query(...),
    batch: str = Query(...),
    filename: str = Query(...),
):
    batch = os.path.basename(batch)
    safe_name = os.path.basename(filename)
    directory = os.path.join(data_dir, batch)
    if not os.path.isdir(directory):
        raise HTTPException(status_code=404, detail="Batch directory not found")

    file_path = os.path.join(directory, safe_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_path, media_type="image/png")


@app.post("/api/save")
def save_labels(body: SaveRequest):
    data_dir = body.data_dir
    batch = os.path.basename(body.batch)

    if not data_dir or not batch:
        raise HTTPException(status_code=400, detail="Missing data_dir or batch")

    now = datetime.now()
    save_obj = {
        "timestamp": now.isoformat(),
        "batch": batch,
        "source_dir": data_dir,
        "tiles": [t.model_dump() for t in body.tiles],
    }

    ts_str = now.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(data_dir, batch, f"human_labels_{ts_str}.json")

    with open(save_path, "w") as f:
        json.dump(save_obj, f, indent=2)

    return {"success": True, "path": save_path}


# ------------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------------


class EvaluateRequest(BaseModel):
    data_dir: str
    batch: str
    human_labels_file: str


def _normalize_labels(labels):
    """Remap arbitrary label ints to a 0-based contiguous range."""
    unique = sorted(set(labels))
    mapping = {old: new for new, old in enumerate(unique)}
    return [mapping[l] for l in labels], mapping


def _hungarian_match_accuracy(human, model, n_human, n_model):
    """Build a contingency matrix and use the Hungarian algorithm to find
    the optimal 1-to-1 mapping from model clusters to human clusters.
    Returns (accuracy, mapping_dict).
    """
    size = max(n_human, n_model)
    cost = np.zeros((size, size), dtype=np.int64)
    for h, m in zip(human, model):
        cost[m, h] += 1
    # Hungarian minimises cost, so negate the counts
    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_model:
            mapping[r] = c if c < n_human else None
    matched = sum(1 for h, m in zip(human, model) if mapping.get(m) == h)
    accuracy = matched / len(human) if human else 0.0
    return accuracy, mapping


@app.get("/api/human_labels")
def list_human_labels(data_dir: str = Query(...), batch: str = Query(...)):
    """List ``human_labels_*.json`` files available for a batch."""
    batch = os.path.basename(batch)
    directory = os.path.join(data_dir, batch)
    if not os.path.isdir(directory):
        raise HTTPException(status_code=404, detail="Batch directory not found")
    files = sorted(
        f
        for f in os.listdir(directory)
        if f.startswith("human_labels_") and f.endswith(".json")
    )
    return {"files": files}


@app.post("/api/evaluate")
def evaluate(body: EvaluateRequest):
    """Compare human labels against model cluster_id values.

    Steps
    -----
    1. Load the model ``cell_labels.json`` and the chosen human labels file.
    2. Match cells present in both files.
    3. Normalize both label sets to ``0 … n``.
    4. Use the Hungarian algorithm to find the best mapping.
    5. Compute batch-level accuracy from tile-level alignments.
    """
    data_dir = body.data_dir
    batch = os.path.basename(body.batch)
    human_file = os.path.basename(body.human_labels_file)

    # --- Load model labels ------------------------------------------------
    model_path = os.path.join(data_dir, batch, "cell_labels.json")
    if not os.path.isfile(model_path):
        raise HTTPException(status_code=404, detail="cell_labels.json not found")

    with open(model_path, "r") as f:
        model_data = json.load(f)

    # Build lookup: (slide_id, tile_name, cell_id) -> cluster_id
    model_lookup: dict[tuple, int] = {}
    model_polygons: dict[tuple, list] = {}
    for tile in model_data.get("tiles", []):
        sid = tile["slide_id"]
        tn = tile["tile_name"]
        for cell in tile.get("cells", []):
            if cell.get("cluster_id") is not None:
                key = (sid, tn, cell["cell_id"])
                model_lookup[key] = cell["cluster_id"]
                if cell.get("polygon"):
                    model_polygons[key] = cell["polygon"]

    # --- Load human labels ------------------------------------------------
    human_path = os.path.join(data_dir, batch, human_file)
    if not os.path.isfile(human_path):
        raise HTTPException(status_code=404, detail="Human labels file not found")

    with open(human_path, "r") as f:
        human_data = json.load(f)

    # --- Match cells present in both --------------------------------------
    human_raw = []
    model_raw = []
    cell_records = []  # (slide_id, tile_name, cell_id, human_label, model_label, polygon)
    for tile in human_data.get("tiles", []):
        sid = tile["slide_id"]
        tn = tile["tile_name"]
        for lc in tile.get("labelled_cells", []):
            key = (sid, tn, lc["cell_id"])
            if key in model_lookup:
                human_raw.append(lc["human_label"])
                model_raw.append(model_lookup[key])
                cell_records.append(
                    (
                        sid,
                        tn,
                        lc["cell_id"],
                        lc["human_label"],
                        model_lookup[key],
                        model_polygons.get(key),
                    )
                )

    if not human_raw:
        raise HTTPException(
            status_code=400,
            detail="No overlapping labelled cells found between human and model.",
        )

    n_human = len(set(human_raw))
    n_model = len(set(model_raw))

    # --- Per-tile cell comparison (tile-level Hungarian alignment) ---------
    # Group raw cell records by tile, then run per-tile Hungarian matching
    # so that each tile's labels are aligned independently.
    tile_raw: dict[tuple, list] = {}
    for rec in cell_records:
        sid, tn = rec[0], rec[1]
        tile_raw.setdefault((sid, tn), []).append(rec)

    tile_comparison = []
    for (sid, tn), recs in tile_raw.items():
        t_human = [r[3] for r in recs]
        t_model = [r[4] for r in recs]

        t_human_norm, t_human_map = _normalize_labels(t_human)
        t_model_norm, t_model_map = _normalize_labels(t_model)
        t_n_human = len(set(t_human_norm))
        t_n_model = len(set(t_model_norm))

        _, t_match_map = _hungarian_match_accuracy(
            t_human_norm,
            t_model_norm,
            t_n_human,
            t_n_model,
        )

        t_rev_human = {v: k for k, v in t_human_map.items()}

        cells = []
        for rec in recs:
            cid, h_raw, m_raw, poly = rec[2], rec[3], rec[4], rec[5]
            m_norm = t_model_map[m_raw]
            aligned_h_idx = t_match_map.get(m_norm)
            aligned_label = (
                t_rev_human.get(aligned_h_idx, "?")
                if aligned_h_idx is not None
                else "?"
            )
            cell_entry: dict = {
                "cell_id": cid,
                "human_label": h_raw,
                "model_label_raw": m_raw,
                "model_label_aligned": aligned_label,
                "match": (h_raw == aligned_label),
            }
            if poly:
                cell_entry["polygon"] = poly
            cells.append(cell_entry)

        # --- Build per-tile confusion matrix (human × aligned-model) ----
        all_labels = sorted(
            set(
                [c["human_label"] for c in cells]
                + [
                    c["model_label_aligned"]
                    for c in cells
                    if c["model_label_aligned"] != "?"
                ]
            )
        )
        label_to_idx = {l: i for i, l in enumerate(all_labels)}
        cm = [[0] * len(all_labels) for _ in all_labels]
        for c in cells:
            hi = label_to_idx.get(c["human_label"])
            mi = label_to_idx.get(c["model_label_aligned"])
            if hi is not None and mi is not None:
                cm[hi][mi] += 1

        img_name = f"{sid}__{tn}"
        if not img_name.endswith(".png"):
            img_name += ".png"
        tile_comparison.append(
            {
                "slide_id": sid,
                "tile_name": tn,
                "image_filename": img_name,
                "cells": cells,
                "confusion_matrix": {"labels": all_labels, "matrix": cm},
            }
        )

    # --- Batch-level accuracy from tile-level alignments ------------------
    total_cells = 0
    total_correct = 0
    for tile in tile_comparison:
        for cell in tile["cells"]:
            total_cells += 1
            if cell["match"]:
                total_correct += 1
    batch_accuracy = round(total_correct / total_cells, 4) if total_cells else 0

    return {
        "n_cells": len(human_raw),
        "n_human_clusters": n_human,
        "n_model_clusters": n_model,
        "metrics": {
            "accuracy": batch_accuracy,
        },
        "tile_comparison": tile_comparison,
    }


class SaveMetricsRequest(BaseModel):
    data_dir: str
    batch: str
    results: dict


@app.post("/api/save_metrics")
def save_metrics(body: SaveMetricsRequest):
    """Save evaluation results to a timestamped JSON file in the batch directory."""
    batch = os.path.basename(body.batch)
    batch_dir = os.path.join(body.data_dir, batch)
    if not os.path.isdir(batch_dir):
        raise HTTPException(status_code=404, detail="Batch directory not found")

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(batch_dir, f"eval_metrics_{ts_str}.json")
    with open(save_path, "w") as f:
        json.dump(body.results, f, indent=2)

    return {"success": True, "path": save_path}


# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("\n  Histology Cell Labelling App")
    print("  Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
