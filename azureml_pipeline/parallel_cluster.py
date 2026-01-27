"""
Azure ML Parallel Entry Script for Cell Clustering (Per-Slide).

This script implements the init()/run(mini_batch) interface required by
Azure ML parallel_run_function. Each mini-batch contains TRIGGER FILES
(one per slide). The actual data is read from SIDE INPUTS.

Input structure (mini-batch):
    trigger_path/slide_id_A  (empty trigger file)
    
Side inputs:
    segmentation_path/slide_id_A/tile_bboxes.json
    prepped_tiles_path/slide_id_A/tile.png

Output creates per-slide subfolders with cluster_assignments.json.

Usage in pipeline:
    parallel_run_function(
        task=RunFunction(entry_script="parallel_cluster.py", ...),
        input_data="${{inputs.trigger_path}}",  # Trigger files
        mini_batch_size="1",  # Slides per mini-batch
        ...
    )
"""
from __future__ import annotations
import argparse
import glob
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import DBSCAN as CPUDbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torchvision import models, transforms

# Conditionally import UMAP and cuML/CuPy
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import cupy as cp
    from cuml.cluster import DBSCAN as GPUDbscan
    HAS_CUML = True
except ImportError:
    HAS_CUML = False

try:
    from kneed import KneeLocator
    HAS_KNEED = True
except ImportError:
    HAS_KNEED = False


# Global state initialized in init()
_args = None
_output_base = None
_manifest_base = None
_model = None
_transform = None
_device = None
_slide_to_files = None  # Cache: slide_id -> list of bbox files


def extract_slide_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract slide_id from a tile/bbox filename.
    
    Filename format: {slide_id}__MAG_{mag}__X_{x}__Y_{y}__IDX_{idx}.png
    or: {slide_id}__MAG_{mag}__X_{x}__Y_{y}__IDX_{idx}_bboxes.json
    
    Returns slide_id or None if pattern doesn't match.
    """
    # Match everything before __MAG_
    match = re.match(r'^(.+?)__MAG_', filename)
    if match:
        return match.group(1)
    return None


def build_slide_to_files_mapping() -> Dict[str, List[str]]:
    """
    Build a mapping from slide_id to list of bbox files.
    
    Segmentation outputs are organized in slide subfolders:
        segmentation_path/
          slideA/
            slideA__MAG_1d000__X_0__Y_0__IDX_000001_bboxes.json
            ...
          slideB/
            slideB__MAG_1d000__X_0__Y_0__IDX_000001_bboxes.json
            ...
    """
    mapping = defaultdict(list)
    
    # Find all bbox files in slide subfolders
    bbox_files = glob.glob(os.path.join(_args.segmentation_path, "*", "*_bboxes.json"))
    
    for bbox_file in bbox_files:
        # Slide ID is the parent folder name
        slide_id = os.path.basename(os.path.dirname(bbox_file))
        mapping[slide_id].append(bbox_file)
    
    logging.info(f"Built slide mapping: {len(mapping)} slides, {len(bbox_files)} bbox files total")
    return dict(mapping)


def init():
    """
    Initialize model and resources before processing mini-batches.
    Called once per worker process.
    """
    global _args, _output_base, _manifest_base, _model, _transform, _device, _slide_to_files
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation_path", type=str, required=True,
                        help="Path to segmentation results (side input, per-slide subfolders)")
    parser.add_argument("--prepped_tiles_path", type=str, required=True,
                        help="Path to the tiled images (side input, per-slide subfolders)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Base output directory for cluster results")
    parser.add_argument("--output_manifest", type=str, required=True,
                        help="Output directory for trigger files (one per slide)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for embedding extraction and DBSCAN")
    
    # Embedding Options
    parser.add_argument("--normalize_embeddings", action="store_true",
                        help="L2 normalize embeddings before reduction/clustering")
    
    # UMAP Options
    parser.add_argument("--use_umap", action="store_true",
                        help="Enable UMAP dimensionality reduction before clustering")
    parser.add_argument("--umap_n_components", type=int, default=50,
                        help="Target dimensions for UMAP")
    parser.add_argument("--umap_n_neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="UMAP min_dist parameter")
    parser.add_argument("--umap_metric", type=str, default="euclidean",
                        help="Metric for UMAP ('euclidean', 'cosine', etc.)")
    
    # DBSCAN Options
    parser.add_argument("--eps", type=float, default=None,
                        help="DBSCAN eps. If None, estimates automatically")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="DBSCAN min_samples (also k for auto-eps)")
    parser.add_argument("--auto_eps_sensitivity", type=float, default=1.0,
                        help="Sensitivity for kneed elbow detection")
    parser.add_argument("--auto_eps_fallback_percentile", type=float, default=95.0,
                        help="Percentile fallback if elbow not found")
    parser.add_argument("--dbscan_metric", type=str, default="euclidean",
                        help="Metric for DBSCAN ('euclidean', 'cosine')")
    
    _args, _ = parser.parse_known_args()
    _output_base = Path(_args.output_path)
    _output_base.mkdir(parents=True, exist_ok=True)
    
    _manifest_base = Path(_args.output_manifest)
    _manifest_base.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    if _args.gpu and torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
        if _args.gpu:
            logging.warning("GPU requested but CUDA not available. Using CPU.")
    
    # Load ResNet50 model for embedding extraction
    logging.info(f"Loading ResNet50 model for embedding extraction on {_device}")
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    _model = models.resnet50(weights=weights)
    _model.fc = nn.Identity()  # Get 2048-dim embedding
    _model.to(_device)
    _model.eval()
    _transform = weights.transforms()
    
    # Build slide-to-files mapping from flat structure
    _slide_to_files = build_slide_to_files_mapping()
    
    logging.info("Parallel clustering initialized")
    logging.info(f"  Segmentation path: {_args.segmentation_path}")
    logging.info(f"  Prepped tiles path: {_args.prepped_tiles_path}")
    logging.info(f"  Output path: {_output_base}")
    logging.info(f"  Device: {_device}")
    logging.info(f"  UMAP enabled: {_args.use_umap}")
    logging.info(f"  Discovered {len(_slide_to_files)} unique slide IDs")


def extract_patch_embedding(
    tile_img: Image.Image,
    bbox: List[int]
) -> Optional[np.ndarray]:
    """Extract embedding for a bounding box patch."""
    xmin, ymin, xmax, ymax = bbox
    if xmin >= xmax or ymin >= ymax:
        return None
    
    try:
        patch = tile_img.crop((xmin, ymin, xmax, ymax))
        if patch.size[0] == 0 or patch.size[1] == 0:
            return None
        if patch.mode != 'RGB':
            patch = patch.convert('RGB')
        
        input_tensor = _transform(patch).unsqueeze(0).to(_device)
        with torch.no_grad():
            embedding = _model(input_tensor)
        return embedding.cpu().numpy().squeeze()
    except Exception as e:
        logging.debug(f"Error extracting embedding for bbox {bbox}: {e}")
        return None


def compute_cluster_confidence(embedding: np.ndarray, centroid: np.ndarray) -> float:
    """Compute confidence score based on distance to centroid."""
    dist = np.linalg.norm(embedding - centroid)
    return float(1.0 / (1.0 + max(0, dist)))


def find_optimal_eps(
    embeddings: np.ndarray,
    min_samples: int,
    output_path: str
) -> Optional[float]:
    """Estimate optimal eps using k-distance graph."""
    if len(embeddings) < min_samples:
        return None
    
    effective_k = min(min_samples, len(embeddings))
    
    try:
        nbrs = NearestNeighbors(n_neighbors=effective_k, algorithm='auto', metric='euclidean').fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        kth_distances = distances[:, effective_k - 1]
        sorted_kth_distances = np.sort(kth_distances)
        
        if HAS_KNEED and len(sorted_kth_distances) >= 3:
            try:
                x = np.arange(len(sorted_kth_distances))
                kneedle = KneeLocator(
                    x, sorted_kth_distances,
                    curve='convex', direction='increasing',
                    S=_args.auto_eps_sensitivity
                )
                if kneedle.elbow_y is not None:
                    return float(kneedle.elbow_y)
            except Exception:
                pass
        
        # Fallback to percentile
        fallback_eps = np.percentile(sorted_kth_distances, _args.auto_eps_fallback_percentile)
        if fallback_eps < 1e-6:
            fallback_eps = np.mean(sorted_kth_distances[sorted_kth_distances > 1e-6]) if np.any(sorted_kth_distances > 1e-6) else 0.1
        return float(fallback_eps)
    
    except Exception as e:
        logging.error(f"Error during k-distance calculation: {e}")
        return None


def get_bbox_file_list_for_slide(slide_id: str) -> List[Tuple[str, str]]:
    """Get list of (bbox_file, tile_file) pairs for a single slide using per-slide subfolder structure."""
    
    # Get bbox files for this slide from the cached mapping
    if slide_id not in _slide_to_files:
        logging.warning(f"Slide ID '{slide_id}' not found in mapping")
        return []
    
    bbox_files = _slide_to_files[slide_id]
    valid_pairs = []
    
    for bbox_file in bbox_files:
        # Convert bbox filename to tile filename
        # slideA__MAG_1d000__X_0__Y_0__IDX_000001_bboxes.json -> slideA__MAG_1d000__X_0__Y_0__IDX_000001.png
        bbox_basename = os.path.basename(bbox_file)
        tile_name = bbox_basename.replace("_bboxes.json", ".png")
        
        # Tiles are in per-slide subfolders: prepped_tiles_path/slide_id/tile.png
        tile_path = os.path.join(_args.prepped_tiles_path, slide_id, tile_name)
        
        if os.path.exists(tile_path):
            valid_pairs.append((bbox_file, tile_path))
        else:
            logging.debug(f"Tile not found: {tile_path}")
    
    return valid_pairs


def cluster_slide(slide_id: str) -> dict:
    """
    Perform clustering for a single slide.
    
    Returns dict with:
        - slide_id: ID of the slide
        - num_embeddings: total embeddings extracted
        - num_clusters: number of clusters found
        - num_noise: number of noise points
        - output_path: path to cluster_assignments.json
    """
    slide_output_path = _output_base / slide_id
    slide_output_path.mkdir(parents=True, exist_ok=True)
    
    # Get bbox/tile pairs for this slide (flat structure)
    bbox_tile_pairs = get_bbox_file_list_for_slide(slide_id)
    
    if not bbox_tile_pairs:
        logging.warning(f"No valid bbox/tile pairs found for slide {slide_id}")
        return {
            "slide_id": slide_id,
            "num_embeddings": 0,
            "num_clusters": 0,
            "num_noise": 0,
            "output_path": None,
            "error": "No bbox/tile pairs found"
        }
    
    logging.info(f"Processing {len(bbox_tile_pairs)} bbox/tile pairs for {slide_id}")
    
    # Extract embeddings
    embeddings = []
    record_tracker = []  # (bbox_file, label_id, bbox)
    
    for bbox_file, tile_path in bbox_tile_pairs:
        try:
            with open(bbox_file, "r") as f:
                bboxes_data = json.load(f)
            
            if not isinstance(bboxes_data, list):
                continue
            
            tile_img = Image.open(tile_path)
            
            for bb_dict in bboxes_data:
                if not isinstance(bb_dict, dict) or "label_id" not in bb_dict or "bbox" not in bb_dict:
                    continue
                
                label_id = bb_dict["label_id"]
                bbox = bb_dict["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                
                emb = extract_patch_embedding(tile_img, bbox)
                if emb is not None:
                    embeddings.append(emb)
                    record_tracker.append((bbox_file, label_id, bbox))
            
            tile_img.close()
            
        except Exception as e:
            logging.error(f"Error processing {bbox_file}: {e}")
    
    if not embeddings:
        logging.warning(f"No embeddings extracted for slide {slide_id}")
        return {
            "slide_id": slide_id,
            "num_embeddings": 0,
            "num_clusters": 0,
            "num_noise": 0,
            "output_path": None,
            "error": "No embeddings extracted"
        }
    
    embeddings_np = np.array(embeddings, dtype=np.float32)
    logging.info(f"Extracted {len(embeddings_np)} embeddings for {slide_id}")
    
    # Optional: Normalize embeddings
    if _args.normalize_embeddings:
        embeddings_np = normalize(embeddings_np, norm='l2', axis=1)
    
    # Optional: UMAP dimensionality reduction
    if _args.use_umap and HAS_UMAP:
        try:
            reducer = umap.UMAP(
                n_neighbors=_args.umap_n_neighbors,
                n_components=_args.umap_n_components,
                min_dist=_args.umap_min_dist,
                metric=_args.umap_metric,
                random_state=42,
                n_jobs=-1,
                verbose=False
            )
            embeddings_np = reducer.fit_transform(embeddings_np)
            logging.info(f"UMAP reduced to {embeddings_np.shape[1]} dimensions for {slide_id}")
        except Exception as e:
            logging.error(f"UMAP failed for {slide_id}: {e}")
    
    # Check minimum samples
    if len(embeddings_np) < _args.min_samples:
        logging.warning(f"Too few embeddings ({len(embeddings_np)}) for DBSCAN on {slide_id}")
        return {
            "slide_id": slide_id,
            "num_embeddings": len(embeddings_np),
            "num_clusters": 0,
            "num_noise": len(embeddings_np),
            "output_path": None,
            "error": f"Too few embeddings for DBSCAN (need at least {_args.min_samples})"
        }
    
    # Determine eps
    chosen_eps = _args.eps
    if chosen_eps is None:
        chosen_eps = find_optimal_eps(embeddings_np, _args.min_samples, str(slide_output_path))
        if chosen_eps is None or chosen_eps <= 0:
            chosen_eps = 0.5  # Default fallback
            logging.warning(f"Using fallback eps={chosen_eps} for {slide_id}")
    
    logging.info(f"Using eps={chosen_eps:.4f} for {slide_id}")
    
    # Run DBSCAN
    labels = None
    use_gpu_dbscan = _args.gpu and HAS_CUML
    
    if use_gpu_dbscan:
        try:
            embeddings_gpu = cp.asarray(embeddings_np, order='C')
            dbscan_gpu = GPUDbscan(eps=chosen_eps, min_samples=_args.min_samples)
            dbscan_gpu.fit(embeddings_gpu)
            labels = dbscan_gpu.labels_.get()
            del embeddings_gpu
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            logging.error(f"cuML DBSCAN failed for {slide_name}: {e}. Falling back to CPU.")
            use_gpu_dbscan = False
    
    if not use_gpu_dbscan:
        dbscan_cpu = CPUDbscan(eps=chosen_eps, min_samples=_args.min_samples, metric=_args.dbscan_metric, n_jobs=-1)
        labels = dbscan_cpu.fit_predict(embeddings_np)
    
    # Compute centroids and confidences
    cluster_to_indices = defaultdict(list)
    for i, cluster_id in enumerate(labels):
        cluster_to_indices[cluster_id].append(i)
    
    centroids: Dict[int, np.ndarray] = {}
    for cluster_id, indices in cluster_to_indices.items():
        if cluster_id == -1:
            continue
        cluster_embeddings = embeddings_np[indices]
        if len(cluster_embeddings) > 0:
            centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
    
    # Build cluster assignments
    cluster_assignments = []
    for i, emb in enumerate(embeddings_np):
        cluster_id = int(labels[i])
        bbox_file, label_id, bbox = record_tracker[i]
        
        confidence = 0.0
        if cluster_id != -1 and cluster_id in centroids:
            confidence = compute_cluster_confidence(emb, centroids[cluster_id])
        
        cluster_assignments.append({
            "bbox_file": bbox_file,
            "label_id": label_id,
            "bbox": bbox,
            "cluster_id": cluster_id,
            "confidence": confidence
        })
    
    # Save results
    out_json_path = slide_output_path / "cluster_assignments.json"
    with open(out_json_path, "w") as f:
        json.dump(cluster_assignments, f, indent=2)
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = list(labels).count(-1)
    
    logging.info(f"Clustering results for {slide_id}: {n_clusters} clusters, {n_noise} noise points")
    
    return {
        "slide_id": slide_id,
        "num_embeddings": len(embeddings_np),
        "num_clusters": n_clusters,
        "num_noise": n_noise,
        "output_path": str(out_json_path)
    }


def run(mini_batch: List[str]) -> List[str]:
    """
    Process a mini-batch of TRIGGER FILES from the segmentation step.
    
    Each item in mini_batch is a path to a trigger file named after the slide_id.
    The actual bbox files are read from segmentation_path side input.
    
    Args:
        mini_batch: List of paths to trigger files (e.g., trigger_path/slide_123)
        
    Returns:
        List of result strings (one per processed slide) - required by Azure ML
    """
    results = []
    
    for trigger_file in mini_batch:
        trigger_path = Path(str(trigger_file).strip())
        if not trigger_path.name:
            continue
        
        # Extract slide_id from trigger filename
        slide_id = trigger_path.name
        
        logging.info(f"Processing slide from trigger: {slide_id}")
        
        # Verify slide exists in our mapping (built during init)
        if slide_id not in _slide_to_files:
            logging.warning(f"Slide ID '{slide_id}' not in pre-built mapping, skipping")
            results.append(f"SKIP:{slide_id}:not_in_mapping")
            continue
        
        try:
            result_info = cluster_slide(slide_id)
            
            if result_info.get("error"):
                result = f"WARN:{slide_id}:{result_info['error']}"
            else:
                result = f"OK:{slide_id}:clusters={result_info['num_clusters']},embeddings={result_info['num_embeddings']}"
                
                # Create trigger file for downstream steps (classify)
                out_trigger = _manifest_base / slide_id
                out_trigger.touch()
                logging.info(f"Created trigger file: {out_trigger}")
            
            logging.info(f"Finished clustering {slide_id}: {result_info['num_clusters']} clusters from {result_info['num_embeddings']} embeddings")
            results.append(result)
            
        except Exception as e:
            logging.error(f"Error clustering {slide_id}: {e}")
            results.append(f"ERROR:{slide_id}:{str(e)}")
    
    return results


def shutdown():
    """Cleanup after all mini-batches processed."""
    global _model
    _model = None
    logging.info("Parallel clustering shutdown complete")
