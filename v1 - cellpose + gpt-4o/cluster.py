import argparse
import json
import os
import glob
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt # For plotting

# Clustering & Utility Imports
from sklearn.cluster import DBSCAN as CPUDbscan
from sklearn.neighbors import NearestNeighbors # For k-distance
from sklearn.preprocessing import normalize # For optional normalization
from kneed import KneeLocator # For automatic elbow finding

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


# --- Helper Functions ---

def load_pretrained_resnet50(device: torch.device) -> nn.Module:
    """Loads a pretrained ResNet50, replaces final FC with identity."""
    # Use recommended weights API
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity() # Get 2048-dim embedding
    model.to(device)
    model.eval()
    return model

def get_image_transform() -> transforms.Compose:
    """Returns the standard transforms for ResNet."""
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    return weights.transforms() # Use transforms associated with weights

def extract_patch_embedding(
    tile_img: Image.Image,
    bbox: List[int],
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device
) -> Optional[np.ndarray]:
    """Extracts embedding for a bounding box patch."""
    xmin, ymin, xmax, ymax = bbox
    if xmin >= xmax or ymin >= ymax:
        logging.warning(f"Invalid bounding box: {bbox}. Width/Height <= 0. Skipping.")
        return None

    try:
        patch = tile_img.crop((xmin, ymin, xmax, ymax))
        if patch.size[0] == 0 or patch.size[1] == 0:
            logging.warning(f"Cropped patch has zero dimension for bbox {bbox}. Skipping.")
            return None
        # Ensure patch is RGB before transform
        if patch.mode != 'RGB':
            patch = patch.convert('RGB')

        input_tensor = transform(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(input_tensor)
        return embedding.cpu().numpy().squeeze()
    except Exception as e:
        logging.error(f"Error extracting embedding for bbox {bbox}: {e}", exc_info=True)
        return None


def compute_cluster_confidence(
    embedding: np.ndarray,
    centroid: np.ndarray
) -> float:
    """Computes confidence score based on distance to centroid."""
    embedding = np.asarray(embedding)
    centroid = np.asarray(centroid)
    dist = np.linalg.norm(embedding - centroid)
    confidence = 1.0 / (1.0 + max(0, dist))
    return float(confidence)

def find_optimal_eps(
    embeddings: np.ndarray,
    min_samples: int,
    output_path: str,
    sensitivity: float = 1.0,
    fallback_percentile: float = 95.0
) -> Optional[float]:
    """Estimates optimal eps using the k-distance graph and KneeLocator."""
    if len(embeddings) < min_samples:
        logging.warning(f"Not enough samples ({len(embeddings)}) for k-distance (k={min_samples}). Cannot estimate eps.")
        return None

    logging.info(f"Calculating k-distance graph to find optimal eps (k={min_samples}).")
    effective_k = min(min_samples, len(embeddings))
    if effective_k != min_samples:
         logging.warning(f"Adjusting k for nearest neighbors from {min_samples} to {effective_k} due to limited samples.")

    try:
        nbrs = NearestNeighbors(n_neighbors=effective_k, algorithm='auto', metric='euclidean').fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        kth_distances = distances[:, effective_k - 1]
        sorted_kth_distances = np.sort(kth_distances)

        # --- Plotting ---
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_kth_distances)
        plt.title(f'k-Distance Graph (k={effective_k}) - Used for Eps Estimation')
        plt.xlabel("Points sorted by distance to k-th neighbor")
        plt.ylabel(f'{effective_k}-th Nearest Neighbor Distance (Epsilon candidates)')
        plt.grid(True)
        k_dist_plot_path = os.path.join(output_path, "k_distance_graph.png")
        plt.savefig(k_dist_plot_path)
        plt.close()
        logging.info(f"Saved k-distance graph to {k_dist_plot_path}")

        # --- Find the elbow ---
        x = np.arange(len(sorted_kth_distances))
        # Filter out potential zero distances at the beginning if they exist
        # (can sometimes happen with duplicates or very close points)
        first_positive_idx = np.argmax(sorted_kth_distances > 1e-9)
        if first_positive_idx > 0:
            logging.info(f"Ignoring first {first_positive_idx} zero/near-zero distances for knee detection.")
            x = x[first_positive_idx:]
            sorted_kth_distances_filt = sorted_kth_distances[first_positive_idx:]
            if len(sorted_kth_distances_filt) < 3: # Need at least 3 points for kneed
                 logging.warning("Not enough non-zero distances to reliably find knee.")
                 sorted_kth_distances_filt = sorted_kth_distances # Use original if filtering removes too much
                 x = np.arange(len(sorted_kth_distances)) # Reset x as well
        else:
            sorted_kth_distances_filt = sorted_kth_distances


        if len(sorted_kth_distances_filt) < 3:
             logging.warning("Too few points overall (< 3) to use kneed. Falling back to percentile.")
             elbow_point_y = None
        else:
            try:
                kneedle = KneeLocator(
                    x,
                    sorted_kth_distances_filt,
                    curve='convex',
                    direction='increasing',
                    S=sensitivity
                )
                elbow_point_y = kneedle.elbow_y
            except Exception as kneed_error:
                 logging.warning(f"kneed KneeLocator failed: {kneed_error}. Falling back to percentile.")
                 elbow_point_y = None


        if elbow_point_y is not None:
            logging.info(f"Automatically determined optimal eps: {elbow_point_y:.4f} (using kneed sensitivity S={sensitivity})")
            return float(elbow_point_y)
        else:
            logging.warning(f"Could not automatically find elbow in k-distance graph (S={sensitivity}).")
            fallback_eps = np.percentile(sorted_kth_distances, fallback_percentile)
            logging.warning(f"Falling back to {fallback_percentile}th percentile distance: {fallback_eps:.4f}")
            # Ensure fallback is not practically zero
            if fallback_eps < 1e-6:
                 alternative_fallback = np.mean(sorted_kth_distances[sorted_kth_distances > 1e-6]) if np.any(sorted_kth_distances > 1e-6) else 0.1
                 logging.warning(f"Fallback percentile distance {fallback_eps:.4f} is near zero. Using mean of non-zero distances or 0.1 as fallback: {alternative_fallback:.4f}")
                 fallback_eps = alternative_fallback

            return float(fallback_eps)

    except Exception as e:
        logging.error(f"Error during k-distance calculation or elbow finding: {e}", exc_info=True)
        return None

def get_bbox_file_list(segmentation_path: str, prepped_tiles_path: str) -> List[Tuple[str, str]]:
    """
    Get list of (bbox_file, tile_file) pairs for clustering.
    
    Returns:
        List of tuples (bbox_file_path, corresponding_tile_file_path)
    """
    bbox_files = glob.glob(os.path.join(segmentation_path, "**/*_bboxes.json"), recursive=True)
    valid_pairs = []
    
    for bbox_file in bbox_files:
        rel_dir = os.path.relpath(os.path.dirname(bbox_file), segmentation_path)
        tile_name = os.path.basename(bbox_file).replace("_bboxes.json", ".png")
        tile_path = os.path.join(prepped_tiles_path, rel_dir, tile_name)
        
        if os.path.exists(tile_path):
            valid_pairs.append((bbox_file, tile_path))
        else:
            logging.warning(f"Tile image not found: {tile_path}. Skipping {bbox_file}.")
    
    return valid_pairs

def perform_clustering(bbox_tile_pairs: List[Tuple[str, str]], output_path: str, 
                      model: nn.Module, transform: transforms.Compose, 
                      device: torch.device, args, name: str = "") -> bool:
    """
    Perform clustering on a list of bbox/tile file pairs.
    
    Args:
        bbox_tile_pairs: List of (bbox_file, tile_file) tuples
        output_path: Directory to write cluster_assignments.json
        model: ResNet model for embedding extraction
        transform: Image transforms
        device: PyTorch device
        args: Command line arguments
        name: Name for logging (e.g., slide name or "global")
    
    Returns:
        True if successful, False otherwise
    """
    if not bbox_tile_pairs:
        logging.warning(f"No valid bbox/tile pairs found for {name or 'clustering'}. Skipping.")
        return False
    
    logging.info(f"Processing {len(bbox_tile_pairs)} bbox/tile pairs for {name or 'global clustering'}...")
    os.makedirs(output_path, exist_ok=True)
    
    # --- Extract Embeddings ---
    embeddings = []
    record_tracker = []  # Stores (bbox_file, label_id, bbox)
    processed_bbox_count = 0
    skipped_bbox_count = 0
    processed_files = 0
    
    logging.info(f"Starting embedding extraction for {name or 'global clustering'}...")
    
    for file_idx, (bbox_file, tile_path) in enumerate(bbox_tile_pairs):
        try:
            with open(bbox_file, "r") as f:
                bboxes_data = json.load(f)
            if not isinstance(bboxes_data, list):
                logging.warning(f"Invalid format in {bbox_file} (expected list). Skipping file.")
                continue

            tile_img = Image.open(tile_path)
            file_bbox_count = 0

            for bb_dict in bboxes_data:
                if not isinstance(bb_dict, dict) or "label_id" not in bb_dict or "bbox" not in bb_dict:
                    logging.warning(f"Skipping invalid entry in {bbox_file}: {bb_dict}")
                    skipped_bbox_count += 1
                    continue

                label_id = bb_dict["label_id"]
                bbox = bb_dict["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    logging.warning(f"Skipping entry with invalid bbox format in {bbox_file}: {bbox}")
                    skipped_bbox_count += 1
                    continue

                emb = extract_patch_embedding(tile_img, bbox, model, transform, device)
                if emb is not None:
                    embeddings.append(emb)
                    record_tracker.append((bbox_file, label_id, bbox))
                    processed_bbox_count += 1
                    file_bbox_count += 1
                else:
                    skipped_bbox_count += 1

            tile_img.close()
            processed_files += 1

            # Progress logging every 50 files or for significant milestones
            if (file_idx + 1) % 50 == 0 or (file_idx + 1) in [1, 10, 25]:
                logging.info(f"  Progress for {name}: {file_idx + 1}/{len(bbox_tile_pairs)} files processed "
                           f"({processed_bbox_count} embeddings, {skipped_bbox_count} skipped)")
            
            # Log individual file progress for first few files or if very few files total
            if file_idx < 5 or len(bbox_tile_pairs) <= 20:
                rel_path = os.path.relpath(bbox_file, args.segmentation_path) if hasattr(args, 'segmentation_path') else os.path.basename(bbox_file)
                logging.info(f"    Processed {rel_path}: {file_bbox_count} embeddings extracted")

        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {bbox_file}. Skipping.")
        except FileNotFoundError:
            logging.error(f"Tile image error: {tile_path}. Skipping {bbox_file}.")
        except Exception as e:
            logging.error(f"Unexpected error processing {bbox_file}: {e}", exc_info=True)

    logging.info(f"Finished embedding extraction for {name}. Got {processed_bbox_count} embeddings.")
    if skipped_bbox_count > 0:
        logging.warning(f"Skipped {skipped_bbox_count} bounding boxes due to errors for {name}.")
    if not embeddings:
        logging.error(f"No embeddings extracted for {name}. Cannot proceed.")
        return False

    embeddings_np = np.array(embeddings, dtype=np.float32)
    del embeddings  # Free memory
    logging.info(f"Embeddings array shape for {name}: {embeddings_np.shape}")

    # --- Optional: Normalize Embeddings ---
    if args.normalize_embeddings:
        logging.info(f"Applying L2 normalization to embeddings for {name}...")
        embeddings_np = normalize(embeddings_np, norm='l2', axis=1)
        logging.info(f"Normalized embeddings norm (mean) for {name}: {np.mean(np.linalg.norm(embeddings_np, axis=1)):.4f}")

    # --- Optional: Dimensionality Reduction (UMAP) ---
    if args.use_umap:
        logging.info(f"Applying UMAP dimensionality reduction for {name}:")
        logging.info(f"  n_components={args.umap_n_components}, n_neighbors={args.umap_n_neighbors}, "
                     f"min_dist={args.umap_min_dist}, metric='{args.umap_metric}'")
        try:
            reducer = umap.UMAP(
                n_neighbors=args.umap_n_neighbors,
                n_components=args.umap_n_components,
                min_dist=args.umap_min_dist,
                metric=args.umap_metric,
                random_state=42,
                n_jobs=-1,
                verbose=False
            )
            embeddings_np = reducer.fit_transform(embeddings_np)
            logging.info(f"Embeddings shape after UMAP for {name}: {embeddings_np.shape}")
        except Exception as e:
            logging.error(f"UMAP failed for {name}: {e}. Proceeding with original embeddings.", exc_info=True)

    # --- Determine DBSCAN Epsilon ---
    if len(embeddings_np) < args.min_samples:
        logging.error(f"Number of embeddings ({len(embeddings_np)}) is less than min_samples ({args.min_samples}) for {name}. Cannot run DBSCAN.")
        return False

    chosen_eps = args.eps
    if chosen_eps is None:
        logging.info(f"Estimating optimal DBSCAN eps for {name}...")
        chosen_eps = find_optimal_eps(
            embeddings=embeddings_np,
            min_samples=args.min_samples,
            output_path=output_path,
            sensitivity=args.auto_eps_sensitivity,
            fallback_percentile=args.auto_eps_fallback_percentile
        )
        if chosen_eps is None or chosen_eps <= 0:
            logging.error(f"Failed to determine a valid positive eps automatically for {name}. Cannot proceed.")
            return False
        logging.info(f"Using automatically estimated eps = {chosen_eps:.4f} for {name}")
    else:
        logging.info(f"Using user-provided eps = {chosen_eps} for {name}")

    # --- Perform DBSCAN Clustering ---
    labels = None
    use_gpu_dbscan = args.gpu and HAS_CUML

    if use_gpu_dbscan:
        try:
            if args.dbscan_metric.lower() != "euclidean":
                logging.warning(f"cuML DBSCAN primarily uses Euclidean metric. Requested '{args.dbscan_metric}' ignored for GPU.")

            logging.info(f"Using GPU DBSCAN for {name}")
            embeddings_gpu = cp.asarray(embeddings_np, order='C')
            dbscan_gpu = GPUDbscan(eps=chosen_eps, min_samples=args.min_samples)
            dbscan_gpu.fit(embeddings_gpu)
            labels = dbscan_gpu.labels_.get()
            del embeddings_gpu
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            logging.error(f"cuML DBSCAN failed for {name}: {e}. Falling back to CPU.", exc_info=True)
            use_gpu_dbscan = False

    if not use_gpu_dbscan:
        metric = args.dbscan_metric.lower()
        try:
            from sklearn.metrics import pairwise_distances
            _ = pairwise_distances(embeddings_np[:2], metric=metric)
        except Exception as metric_e:
            logging.warning(f"Metric '{metric}' is invalid for sklearn DBSCAN: {metric_e}. Using 'euclidean'.")
            metric = "euclidean"

        logging.info(f"Using CPU DBSCAN for {name}")
        dbscan_cpu = CPUDbscan(eps=chosen_eps, min_samples=args.min_samples, metric=metric, n_jobs=-1)
        labels = dbscan_cpu.fit_predict(embeddings_np)

    # --- Compute Centroids and Confidences ---
    from collections import defaultdict
    cluster_to_indices = defaultdict(list)
    for i, cluster_id in enumerate(labels):
        cluster_to_indices[cluster_id].append(i)

    centroids: Dict[int, np.ndarray] = {}
    for cluster_id, indices in cluster_to_indices.items():
        if cluster_id == -1: continue
        cluster_embeddings = embeddings_np[indices]
        if len(cluster_embeddings) > 0:
            centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)

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

    out_json_path = os.path.join(output_path, "cluster_assignments.json")
    try:
        with open(out_json_path, "w") as f:
            json.dump(cluster_assignments, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to write output JSON to {out_json_path}: {e}", exc_info=True)
        return False

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = list(labels).count(-1)
    noise_percent = (n_noise / len(labels)) * 100 if len(labels) > 0 else 0
    
    # Log cluster size distribution
    cluster_sizes = {}
    for cluster_id in unique_labels:
        if cluster_id != -1:
            cluster_sizes[cluster_id] = list(labels).count(cluster_id)
    
    if cluster_sizes:
        avg_cluster_size = np.mean(list(cluster_sizes.values()))
        max_cluster_size = max(cluster_sizes.values())
        min_cluster_size = min(cluster_sizes.values())
        
        logging.info(f"Clustering results for {name}:")
        logging.info(f"  • Total embeddings processed: {len(labels)}")
        logging.info(f"  • Clusters found: {n_clusters}")
        logging.info(f"  • Noise points: {n_noise} ({noise_percent:.1f}%)")
        logging.info(f"  • Cluster size statistics:")
        logging.info(f"    - Average: {avg_cluster_size:.1f}")
        logging.info(f"    - Range: {min_cluster_size} - {max_cluster_size}")
        logging.info(f"  • Parameters used: eps={chosen_eps:.4f}, min_samples={args.min_samples}")
        
        # Show top 10 largest clusters
        if len(cluster_sizes) > 0:
            top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
            logging.info(f"  • Top {min(len(top_clusters), 10)} largest clusters:")
            for cluster_id, size in top_clusters:
                logging.info(f"    - Cluster {cluster_id}: {size} cells")
    else:
        logging.warning(f"No valid clusters found for {name}!")
    
    logging.info(f"Saved cluster assignments to: {out_json_path}")
    
    return True

# --- Main Function ---
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Cluster cell bounding box embeddings using DBSCAN, with optional UMAP reduction.")
    parser.add_argument("--segmentation_path", type=str, required=True, help="Path to bounding box JSON files.")
    parser.add_argument("--prepped_tiles_path", type=str, required=True, help="Path to tiled images.")
    parser.add_argument("--output_path", type=str, required=True, help="Where to write cluster outputs.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for embedding extraction and DBSCAN (if cuML available).")
    parser.add_argument("--per_slide", action="store_true", help="Cluster each slide separately instead of all slides together.")
    parser.add_argument("--slide_folders", type=str, nargs='*', help="Specific slide folder names to process. If not provided, processes all folders.")

    # Embedding Options
    parser.add_argument("--normalize_embeddings", action="store_true", help="L2 normalize embeddings before reduction/clustering.")

    # UMAP Options
    parser.add_argument("--use_umap", action="store_true", help="Enable UMAP dimensionality reduction before clustering.")
    parser.add_argument("--umap_n_components", type=int, default=50, help="Target dimensions for UMAP.")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP n_neighbors parameter.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist parameter.")
    parser.add_argument("--umap_metric", type=str, default="euclidean", help="Metric for UMAP ('euclidean', 'cosine', etc.).")

    # DBSCAN Options
    parser.add_argument("--eps", type=float, default=None, help="DBSCAN eps. If None, estimates automatically.")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN min_samples (also k for auto-eps).")
    parser.add_argument("--auto_eps_sensitivity", type=float, default=1.0, help="Sensitivity for kneed elbow detection.")
    parser.add_argument("--auto_eps_fallback_percentile", type=float, default=95.0, help="Percentile fallback if elbow not found.")
    parser.add_argument("--dbscan_metric", type=str, default="euclidean", help="Metric for DBSCAN ('euclidean', 'cosine'). Note: cuML favors Euclidean.")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    logging.info(f"Starting clustering process.")
    logging.info(f"Input BBoxes: {args.segmentation_path}")
    logging.info(f"Input Tiles: {args.prepped_tiles_path}")
    logging.info(f"Output Path: {args.output_path}")

    # Validate UMAP usage vs installation
    if args.use_umap and not HAS_UMAP:
        logging.error("UMAP requested (--use_umap) but 'umap-learn' package is not installed. Please install it.")
        return
    # Validate GPU usage vs installation
    if args.gpu and not torch.cuda.is_available():
         logging.warning("GPU requested (--gpu) but no CUDA device found. Proceeding with CPU.")
         args.gpu = False
    if args.gpu and not HAS_CUML:
         logging.warning("GPU requested (--gpu) but cuML/CuPy not found. Embedding on GPU, DBSCAN on CPU.")
         # Keep args.gpu=True for embedding, but DBSCAN will fallback later


    # --- Device Setup ---
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device for embedding extraction: {device}")

    # --- Load Model ---
    model = load_pretrained_resnet50(device)
    transform = get_image_transform()

    # --- Choose Processing Mode ---
    if args.per_slide:
        # Per-slide clustering mode
        logging.info("Per-slide clustering mode enabled.")
        all_slide_dirs = [d for d in os.listdir(args.segmentation_path) 
                         if os.path.isdir(os.path.join(args.segmentation_path, d))]
        
        if not all_slide_dirs:
            logging.error(f"No slide directories found in {args.segmentation_path}")
            return
        
        # Filter slides based on --slide_folders argument
        if args.slide_folders:
            # Validate that requested slide folders exist
            slide_dirs = []
            for slide_folder in args.slide_folders:
                if slide_folder in all_slide_dirs:
                    slide_dirs.append(slide_folder)
                    logging.info(f"Selected slide folder: {slide_folder}")
                else:
                    logging.warning(f"Requested slide folder '{slide_folder}' not found in {args.segmentation_path}. Available folders: {all_slide_dirs}")
            
            if not slide_dirs:
                logging.error("None of the requested slide folders were found. Exiting.")
                return
        else:
            slide_dirs = all_slide_dirs
            logging.info(f"No specific slide folders requested. Processing all {len(slide_dirs)} folders: {slide_dirs}")
            
        failed_slides = []
        successful_slides = []
        
        logging.info(f"Starting per-slide clustering for {len(slide_dirs)} slides...")
        
        for slide_idx, slide_name in enumerate(slide_dirs, 1):
            logging.info(f"Processing slide {slide_idx}/{len(slide_dirs)}: {slide_name}")
            
            slide_segmentation_path = os.path.join(args.segmentation_path, slide_name)
            slide_prepped_tiles_path = os.path.join(args.prepped_tiles_path, slide_name)
            slide_output_path = os.path.join(args.output_path, slide_name)
            
            if not os.path.exists(slide_prepped_tiles_path):
                logging.warning(f"Prepped tiles directory not found for slide {slide_name}: {slide_prepped_tiles_path}. Skipping.")
                failed_slides.append(slide_name)
                continue
            
            # Get bbox/tile pairs for this slide
            bbox_tile_pairs = get_bbox_file_list(slide_segmentation_path, slide_prepped_tiles_path)
            
            if not bbox_tile_pairs:
                logging.warning(f"No valid bbox/tile pairs found for slide {slide_name}. Skipping.")
                failed_slides.append(slide_name)
                continue
            
            logging.info(f"Found {len(bbox_tile_pairs)} bbox/tile pairs for slide {slide_name}")
            
            # Perform clustering for this slide
            success = perform_clustering(bbox_tile_pairs, slide_output_path, model, transform, device, args, slide_name)
            if success:
                successful_slides.append(slide_name)
                logging.info(f"✓ Successfully processed slide {slide_idx}/{len(slide_dirs)}: {slide_name}")
            else:
                failed_slides.append(slide_name)
                logging.error(f"✗ Failed to process slide {slide_idx}/{len(slide_dirs)}: {slide_name}")
        
        # Final summary
        logging.info(f"Clustering summary:")
        logging.info(f"  Total slides attempted: {len(slide_dirs)}")
        logging.info(f"  Successfully processed: {len(successful_slides)}")
        logging.info(f"  Failed: {len(failed_slides)}")
        
        if successful_slides:
            logging.info(f"  Successful slides: {successful_slides}")
        if failed_slides:
            logging.warning(f"  Failed slides: {failed_slides}")
        
        return
    
    else:
        # Global clustering mode
        logging.info("Running clustering across all slides together (global clustering)...")
        
        # Get all bbox/tile pairs across all slides
        bbox_tile_pairs = get_bbox_file_list(args.segmentation_path, args.prepped_tiles_path)
        
        # Perform clustering for all slides combined
        success = perform_clustering(bbox_tile_pairs, args.output_path, model, transform, device, args)
        if not success:
            logging.error("Global clustering failed.")
        else:
            logging.info("Global clustering completed successfully.")


if __name__ == "__main__":
    main()