import argparse
import os
import glob
import json
import numpy as np
import torch
import cv2
import logging
from typing import Dict, Tuple, Optional, List
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

# Optional imports with graceful fallbacks
try:
    import transformers
    from transformers import AutoModel, AutoImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("Transformers not available. Install transformers to use DINOv2/UNI features.")

try:
    import cuml
    from cuml.manifold import UMAP as cumlUMAP
    from cuml.cluster import KMeans as cumlKMeans, DBSCAN as cumlDBSCAN
    HAS_CUML = True
except ImportError:
    HAS_CUML = False
    logging.info("cuML not available. GPU acceleration disabled. Install cuml for GPU clustering.")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logging.info("UMAP not available. Install umap-learn for dimensionality reduction.")

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels
    HAS_CRF = True
except ImportError:
    HAS_CRF = False
    logging.warning("pydensecrf not available. CRF smoothing will be skipped.")

try:
    from kneed import KneeLocator
    HAS_KNEED = True
except ImportError:
    HAS_KNEED = False
    logging.info("kneed not available. Automatic eps estimation will use percentile fallback.")


class TokenClusteringPipeline:    """
    Token clustering pipeline using DINOv2 or UNI encoders for histology analysis.
    Uses DBSCAN + UMAP clustering similar to v1 approach but adapted for tokens.
    """
    
    def __init__(self, model_name: str = "facebook/dinov2-large", 
                 use_crf: bool = False, device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.clustering_method = "dbscan"  # Fixed to DBSCAN only
        self.use_crf = use_crf
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for token clustering")
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Get model configuration
        self.patch_size = getattr(self.model.config, 'patch_size', 16)
        self.hidden_size = self.model.config.hidden_size
        
        logging.info(f"Loaded {model_name} with patch_size={self.patch_size}, hidden_size={self.hidden_size}")
    
    def extract_token_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract token features from a 256x256 image.
        Returns: [N_tokens, D] feature array
        """
        # Ensure image is 256x256
        if image.size != (256, 256):
            image = image.resize((256, 256), Image.BICUBIC)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get the last hidden state (excluding CLS token for DINOv2)
            last_hidden_states = outputs.last_hidden_state  # [1, N_tokens, D]
            
            # Remove batch dimension and CLS token if present
            if self.model_name.startswith("facebook/dinov2"):
                # DINOv2 has CLS token as first token
                token_features = last_hidden_states[0, 1:, :]  # Remove CLS token
            else:
                # For other models, might need different handling
                token_features = last_hidden_states[0, :, :]
        
        return token_features.cpu().numpy()
    
    def tokens_to_spatial_map(self, token_features: np.ndarray, image_size: int = 256) -> Tuple[np.ndarray, int, int]:
        """
        Convert token features to spatial 2D map.
        Returns: (reshaped_features [H, W, D], H, W)
        """
        n_tokens, feature_dim = token_features.shape
        
        # Calculate spatial dimensions
        tokens_per_side = int(np.sqrt(n_tokens))
        
        if tokens_per_side * tokens_per_side != n_tokens:
            # Handle non-square token arrangements
            tokens_h = int(np.sqrt(n_tokens))
            tokens_w = n_tokens // tokens_h
            if tokens_h * tokens_w < n_tokens:
                tokens_w += 1
            # Pad if necessary
            if tokens_h * tokens_w > n_tokens:
                padding = tokens_h * tokens_w - n_tokens
                token_features = np.pad(token_features, ((0, padding), (0, 0)), mode='constant')
        else:
            tokens_h = tokens_w = tokens_per_side
        
        # Reshape to spatial map
        spatial_features = token_features[:tokens_h * tokens_w].reshape(tokens_h, tokens_w, feature_dim)
        
        return spatial_features, tokens_h, tokens_w
    
    def find_optimal_eps(self, embeddings: np.ndarray, min_samples: int, 
                        output_dir: str, sensitivity: float = 1.0, 
                        fallback_percentile: float = 95.0) -> Optional[float]:
        """
        Estimate optimal eps using k-distance graph and KneeLocator.
        Adapted from v1 clustering approach.
        """
        if len(embeddings) < min_samples:
            logging.warning(f"Not enough samples ({len(embeddings)}) for k-distance (k={min_samples}). Cannot estimate eps.")
            return None

        logging.info(f"Calculating k-distance graph to find optimal eps (k={min_samples}).")
        effective_k = min(min_samples, len(embeddings))
        
        try:
            nbrs = NearestNeighbors(n_neighbors=effective_k, algorithm='auto', metric='euclidean').fit(embeddings)
            distances, _ = nbrs.kneighbors(embeddings)
            kth_distances = distances[:, effective_k - 1]
            sorted_kth_distances = np.sort(kth_distances)

            # Create k-distance plot
            plt.figure(figsize=(10, 6))
            plt.plot(sorted_kth_distances)
            plt.title(f'k-Distance Graph (k={effective_k}) - Used for Eps Estimation')
            plt.xlabel("Points sorted by distance to k-th neighbor")
            plt.ylabel(f'{effective_k}-th Nearest Neighbor Distance (Epsilon candidates)')
            plt.grid(True)
            k_dist_plot_path = os.path.join(output_dir, "k_distance_graph.png")
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(k_dist_plot_path)
            plt.close()
            logging.info(f"Saved k-distance graph to {k_dist_plot_path}")

            # Find the elbow using KneeLocator if available
            if HAS_KNEED and len(sorted_kth_distances) >= 3:
                x = np.arange(len(sorted_kth_distances))
                # Filter out potential zero distances
                first_positive_idx = np.argmax(sorted_kth_distances > 1e-9)
                if first_positive_idx > 0:
                    x = x[first_positive_idx:]
                    sorted_kth_distances_filt = sorted_kth_distances[first_positive_idx:]
                else:
                    sorted_kth_distances_filt = sorted_kth_distances

                if len(sorted_kth_distances_filt) >= 3:
                    try:
                        kneedle = KneeLocator(
                            x[:len(sorted_kth_distances_filt)], 
                            sorted_kth_distances_filt,
                            curve="convex", 
                            direction="increasing",
                            S=sensitivity
                        )
                        if kneedle.elbow is not None:
                            elbow_point_y = sorted_kth_distances_filt[kneedle.elbow - x[0]] if kneedle.elbow >= x[0] else None
                        else:
                            elbow_point_y = None
                    except Exception as kneed_error:
                        logging.warning(f"KneeLocator failed: {kneed_error}")
                        elbow_point_y = None
                else:
                    elbow_point_y = None
            else:
                elbow_point_y = None

            if elbow_point_y is not None and elbow_point_y > 1e-6:
                logging.info(f"Automatically determined optimal eps: {elbow_point_y:.4f}")
                return float(elbow_point_y)
            else:
                # Fallback to percentile
                fallback_eps = np.percentile(sorted_kth_distances, fallback_percentile)
                logging.warning(f"Could not find elbow. Using {fallback_percentile}th percentile: {fallback_eps:.4f}")
                if fallback_eps < 1e-6:
                    fallback_eps = 0.1  # Reasonable default
                return float(fallback_eps)

        except Exception as e:
            logging.error(f"Error during k-distance calculation: {e}")
            return None
    
    def cluster_tokens_dbscan(self, spatial_features: np.ndarray, 
                             eps: Optional[float] = None, min_samples: int = 5,
                             use_umap: bool = True, use_gpu: bool = False,
                             umap_n_components: int = 50, umap_n_neighbors: int = 15,
                             umap_min_dist: float = 0.1, normalize_embeddings: bool = True,
                             output_dir: str = "") -> np.ndarray:
        """
        Cluster token features using DBSCAN + UMAP approach similar to v1.
        """
        h, w, d = spatial_features.shape
        flattened_features = spatial_features.reshape(-1, d)
        
        logging.info(f"Starting DBSCAN clustering on {len(flattened_features)} tokens with {d} dimensions")
        
        # Normalize features if requested
        if normalize_embeddings:
            logging.info("Applying L2 normalization to embeddings...")
            flattened_features = normalize(flattened_features, norm='l2', axis=1)
        else:
            # Standard scaling
            scaler = StandardScaler()
            flattened_features = scaler.fit_transform(flattened_features)
        
        # Apply UMAP dimensionality reduction if requested
        reduced_features = flattened_features
        if use_umap:
            logging.info(f"Applying UMAP reduction: {d} -> {umap_n_components} dimensions")
            
            if use_gpu and HAS_CUML:
                # Use cuML GPU UMAP
                try:
                    import cudf
                    features_gpu = cudf.DataFrame(flattened_features)
                    umap_reducer = cumlUMAP(
                        n_components=umap_n_components,
                        n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist,
                        random_state=42
                    )
                    reduced_features = umap_reducer.fit_transform(features_gpu).values
                    logging.info("Used GPU UMAP")
                except Exception as e:
                    logging.warning(f"GPU UMAP failed: {e}. Falling back to CPU UMAP")
                    use_gpu = False
            
            if not use_gpu and HAS_UMAP:
                # Use CPU UMAP
                umap_reducer = umap.UMAP(
                    n_components=umap_n_components,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    random_state=42
                )
                reduced_features = umap_reducer.fit_transform(flattened_features)
                logging.info("Used CPU UMAP")
            elif not HAS_UMAP:
                logging.warning("UMAP requested but not available. Skipping dimensionality reduction.")
        
        # Estimate eps if not provided
        if eps is None:
            logging.info("Estimating optimal DBSCAN eps...")
            eps = self.find_optimal_eps(
                reduced_features, min_samples, output_dir,
                sensitivity=1.0, fallback_percentile=95.0
            )
            if eps is None or eps <= 0:
                eps = 0.5  # Default fallback
                logging.warning(f"Could not estimate eps. Using default: {eps}")
        
        logging.info(f"Using DBSCAN with eps={eps:.4f}, min_samples={min_samples}")
        
        # Perform DBSCAN clustering
        if use_gpu and HAS_CUML:
            try:
                import cudf
                if isinstance(reduced_features, np.ndarray):
                    features_gpu = cudf.DataFrame(reduced_features)
                else:
                    features_gpu = reduced_features
                
                clusterer = cumlDBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = clusterer.fit_predict(features_gpu).values.ravel()
                logging.info("Used GPU DBSCAN")
            except Exception as e:
                logging.warning(f"GPU DBSCAN failed: {e}. Falling back to CPU DBSCAN")
                use_gpu = False
        
        if not use_gpu:
            # Use sklearn CPU DBSCAN
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            cluster_labels = clusterer.fit_predict(reduced_features)
            logging.info("Used CPU DBSCAN")
        
        # Handle noise points (label -1) by creating a separate "background" cluster
        if np.any(cluster_labels == -1):
            max_label = np.max(cluster_labels[cluster_labels != -1]) if np.any(cluster_labels != -1) else -1
            cluster_labels[cluster_labels == -1] = max_label + 1
            n_noise = np.sum(cluster_labels == max_label + 1)
            logging.info(f"DBSCAN found {n_noise} noise points, assigned to background cluster {max_label + 1}")
        
        # Ensure cluster_labels is numpy array
        if hasattr(cluster_labels, 'values'):
            cluster_labels = cluster_labels.values
        if hasattr(cluster_labels, 'ravel'):
            cluster_labels = cluster_labels.ravel()
        
        # Reshape back to spatial map
        cluster_map = cluster_labels.reshape(h, w)
        
        n_final_clusters = len(np.unique(cluster_map))
        logging.info(f"DBSCAN clustering complete: {n_final_clusters} clusters found")
        
        return cluster_map
    
    def apply_crf_smoothing(self, image: np.ndarray, cluster_map: np.ndarray, 
                           n_classes: int) -> np.ndarray:
        """
        Apply CRF smoothing to cluster assignments.
        """
        if not HAS_CRF:
            logging.warning("CRF not available, returning original cluster map")
            return cluster_map
        
        h, w = cluster_map.shape
        
        # Resize cluster map to match image size if necessary
        if cluster_map.shape != image.shape[:2]:
            cluster_map_resized = cv2.resize(
                cluster_map.astype(np.float32), 
                (image.shape[1], image.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            ).astype(int)
            h, w = cluster_map_resized.shape
        else:
            cluster_map_resized = cluster_map
        
        # Create CRF
        d = dcrf.DenseCRF2D(w, h, n_classes)
        
        # Convert cluster map to unary potentials
        unary = unary_from_labels(cluster_map_resized, n_classes, gt_prob=0.7)
        d.setUnaryEnergy(unary)
        
        # Add pairwise potentials
        d.addPairwiseGaussian(sxy=(3, 3), compat=3)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=image, compat=10)
        
        # Inference
        Q = d.inference(5)
        map_result = np.argmax(Q, axis=0).reshape((h, w))
        
        return map_result
    
    def majority_filter_smoothing(self, cluster_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply majority filtering for smoothing (alternative to CRF).
        """
        # Apply median filter as a simple majority filter
        smoothed = ndimage.median_filter(cluster_map, size=kernel_size)
        return smoothed
    
    def process_tile(self, image_path: str, output_dir: str, 
                    eps: Optional[float] = None, min_samples: int = 5,
                    use_umap: bool = True, use_gpu: bool = False,
                    umap_n_components: int = 50, umap_n_neighbors: int = 15,
                    umap_min_dist: float = 0.1, normalize_embeddings: bool = True,
                    apply_smoothing: bool = True) -> Dict:
        """
        Process a single tile: extract tokens, cluster with DBSCAN+UMAP, and save results.
        """
        tile_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Load image
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        
        # Extract token features
        token_features = self.extract_token_features(image_pil)
        
        # Convert to spatial map
        spatial_features, tokens_h, tokens_w = self.tokens_to_spatial_map(token_features)
        
        # Cluster tokens using DBSCAN + UMAP
        cluster_map = self.cluster_tokens_dbscan(
            spatial_features, eps=eps, min_samples=min_samples,
            use_umap=use_umap, use_gpu=use_gpu,
            umap_n_components=umap_n_components,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            normalize_embeddings=normalize_embeddings,
            output_dir=output_dir
        )
        
        # Apply smoothing
        if apply_smoothing:
            n_unique_clusters = len(np.unique(cluster_map))
            
            if HAS_CRF and image_np.shape[:2] == cluster_map.shape:
                # Use CRF if available and dimensions match
                try:
                    cluster_map = self.apply_crf_smoothing(image_np, cluster_map, n_unique_clusters)
                except Exception as e:
                    logging.warning(f"CRF smoothing failed: {e}, using majority filter")
                    cluster_map = self.majority_filter_smoothing(cluster_map)
            else:
                # Fallback to majority filtering
                cluster_map = self.majority_filter_smoothing(cluster_map)
        
        # Resize cluster map to original image size for consistency
        if cluster_map.shape != image_np.shape[:2]:
            cluster_map_full = cv2.resize(
                cluster_map.astype(np.float32),
                (image_np.shape[1], image_np.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(int)
        else:
            cluster_map_full = cluster_map
        
        # Save cluster map
        cluster_filename = f"{tile_name}_token_clusters.png"
        cluster_path = os.path.join(output_dir, cluster_filename)
        
        # Convert to uint8 for saving (scale to 0-255)
        max_cluster = np.max(cluster_map_full)
        if max_cluster > 0:
            cluster_img = (cluster_map_full * (255 // max_cluster)).astype(np.uint8)
        else:
            cluster_img = cluster_map_full.astype(np.uint8)
        Image.fromarray(cluster_img).save(cluster_path)
        
        # Save cluster assignments as JSON
        cluster_info = {
            "tile_name": tile_name,
            "n_clusters": int(np.max(cluster_map_full) + 1),
            "token_grid_size": [tokens_h, tokens_w],
            "clustering_method": "dbscan",
            "smoothing_applied": apply_smoothing,
            "cluster_stats": {}
        }
        
        # Calculate cluster statistics
        for cluster_id in np.unique(cluster_map_full):
            cluster_mask = cluster_map_full == cluster_id
            cluster_info["cluster_stats"][int(cluster_id)] = {
                "pixel_count": int(np.sum(cluster_mask)),
                "percentage": float(np.sum(cluster_mask) / cluster_mask.size * 100)
            }
        
        info_filename = f"{tile_name}_token_cluster_info.json"
        info_path = os.path.join(output_dir, info_filename)
        with open(info_path, "w") as f:
            json.dump(cluster_info, f, indent=2)
        
        # Create visualization
        self.create_cluster_visualization(image_np, cluster_map_full, 
                                        os.path.join(output_dir, f"{tile_name}_clusters_viz.png"))
        
        logging.info(f"Processed {tile_name}: {cluster_info['n_clusters']} clusters found")
        
        return cluster_info
    
    def process_slide(self, tile_paths: List[str], slide_name: str, output_dir: str,
                     **kwargs) -> Dict:
        """
        Process all tiles in a slide using DBSCAN + UMAP clustering.
        """
        logging.info(f"Processing slide {slide_name} with {len(tile_paths)} tiles using DBSCAN + UMAP")
        
        all_results = []
        for tile_path in tile_paths:
            try:
                result = self.process_tile(tile_path, output_dir, **kwargs)
                all_results.append(result)
            except Exception as e:
                logging.error(f"Error processing tile {tile_path}: {e}")
                continue
        
        # Save slide summary
        slide_summary = {
            "slide_name": slide_name,
            "total_tiles_processed": len(all_results),
            "clustering_method": "dbscan_umap",
            "tile_results": all_results
        }
        
        summary_path = os.path.join(output_dir, f"{slide_name}_clustering_summary.json")
        with open(summary_path, "w") as f:
            json.dump(slide_summary, f, indent=2)
        
        return {"cluster_assignments": all_results, "summary": slide_summary}
    
    def create_cluster_visualization(self, image: np.ndarray, cluster_map: np.ndarray, save_path: str):
        """Create and save cluster visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Cluster map
        im = axes[1].imshow(cluster_map, cmap='tab10')
        axes[1].set_title("DBSCAN Token Clusters")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        overlay = image.copy().astype(float) / 255.0
        max_cluster = np.max(cluster_map)
        if max_cluster > 0:
            cluster_colored = plt.cm.tab10(cluster_map / max_cluster)[:, :, :3]
        else:
            cluster_colored = np.zeros_like(overlay)
        blended = 0.6 * overlay + 0.4 * cluster_colored
        axes[2].imshow(blended)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser(description="Token clustering for histology tiles using DBSCAN + UMAP")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to tiled images (256x256 preferred).")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path for clustering output.")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-large",
                       help="Model name for feature extraction.")
    
    # DBSCAN parameters
    parser.add_argument("--eps", type=float, default=None,
                       help="DBSCAN eps. If None, estimates automatically.")
    parser.add_argument("--min_samples", type=int, default=5,
                       help="DBSCAN min_samples parameter.")
    
    # UMAP parameters
    parser.add_argument("--use_umap", action="store_true",
                       help="Apply UMAP dimensionality reduction before clustering.")
    parser.add_argument("--umap_n_components", type=int, default=50,
                       help="Target dimensions for UMAP reduction.")
    parser.add_argument("--umap_n_neighbors", type=int, default=15,
                       help="UMAP n_neighbors parameter.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                       help="UMAP min_dist parameter.")
    
    # Other parameters
    parser.add_argument("--normalize_embeddings", action="store_true",
                       help="L2 normalize embeddings before clustering.")
    parser.add_argument("--use_gpu", action="store_true",
                       help="Use GPU acceleration for clustering (requires cuML).")
    parser.add_argument("--no_smoothing", action="store_true",
                       help="Disable CRF/majority smoothing.")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu).")
    
    args = parser.parse_args()
    
    if not HAS_TRANSFORMERS:
        logging.error("transformers is required but not installed.")
        return
    
    logging.info("Starting DBSCAN + UMAP token clustering with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize pipeline
    try:
        pipeline = TokenClusteringPipeline(
            model_name=args.model_name,
            clustering_method="dbscan",
            device=args.device
        )
        logging.info(f"Initialized DBSCAN token clustering pipeline with {args.model_name}")
    except Exception as e:
        logging.error(f"Failed to initialize pipeline: {e}")
        return
    
    # Find all tile images
    tile_files = glob.glob(os.path.join(args.input_path, "**/*.png"), recursive=True)
    if not tile_files:
        logging.warning("No tile images found in the input path: %s", args.input_path)
        return
    
    logging.info(f"Found {len(tile_files)} tiles to process")
    
    # Process each tile
    all_results = []
    for tile_file in tile_files:
        relative_path = os.path.relpath(tile_file, args.input_path)
        tile_out_dir = os.path.join(args.output_path, os.path.dirname(relative_path))
        os.makedirs(tile_out_dir, exist_ok=True)
        
        try:
            result = pipeline.process_tile(
                tile_file, 
                tile_out_dir,
                eps=args.eps,
                min_samples=args.min_samples,
                use_umap=args.use_umap,
                use_gpu=args.use_gpu,
                umap_n_components=args.umap_n_components,
                umap_n_neighbors=args.umap_n_neighbors,
                umap_min_dist=args.umap_min_dist,
                normalize_embeddings=args.normalize_embeddings,
                apply_smoothing=not args.no_smoothing
            )
            all_results.append(result)
        except Exception as e:
            logging.error(f"Error processing {tile_file}: {e}")
            continue
    
    # Save summary
    summary_path = os.path.join(args.output_path, "clustering_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "total_tiles_processed": len(all_results),
            "model_used": args.model_name,
            "clustering_method": "dbscan_umap",
            "results": all_results
        }, f, indent=2)
    
    logging.info(f"DBSCAN + UMAP token clustering completed. Processed {len(all_results)} tiles. Output saved to: %s", args.output_path)


if __name__ == "__main__":
    main()
