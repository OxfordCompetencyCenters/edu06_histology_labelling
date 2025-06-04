import argparse
import os
import glob
import json
import numpy as np
import torch
import cv2
import logging
from typing import Dict, Tuple
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

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
    from cuml.cluster import KMeans as cumlKMeans, HDBSCAN as cumlHDBSCAN
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

class TokenClusteringPipeline:
    """
    Token clustering pipeline using DINOv2 or UNI encoders for histology analysis.
    Processes 256x256 tiles to extract token maps and perform clustering.
    """
    
    def __init__(self, model_name: str = "facebook/dinov2-large", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        
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
      def cluster_tokens(self, spatial_features: np.ndarray, n_clusters: int = 3, 
                      method: str = "kmeans", use_umap: bool = False, 
                      use_gpu: bool = False, umap_n_components: int = 50,
                      umap_n_neighbors: int = 15, umap_min_dist: float = 0.1) -> np.ndarray:
        """
        Cluster token features with optional UMAP dimensionality reduction and GPU acceleration.
        """
        h, w, d = spatial_features.shape
        flattened_features = spatial_features.reshape(-1, d)
        
        # Normalize features
        if use_gpu and HAS_CUML:
            # Use cuML GPU StandardScaler
            import cudf
            features_gpu = cudf.DataFrame(flattened_features)
            from cuml.preprocessing import StandardScaler as cumlStandardScaler
            scaler = cumlStandardScaler()
            normalized_features = scaler.fit_transform(features_gpu).values
        else:
            # Use sklearn CPU StandardScaler
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(flattened_features)
        
        # Apply UMAP dimensionality reduction if requested
        if use_umap:
            logging.info(f"Applying UMAP reduction: {d} -> {umap_n_components} dimensions")
            
            if use_gpu and HAS_CUML:
                # Use cuML GPU UMAP
                umap_reducer = cumlUMAP(
                    n_components=umap_n_components,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    random_state=42
                )
                if isinstance(normalized_features, np.ndarray):
                    import cudf
                    normalized_features = cudf.DataFrame(normalized_features)
                reduced_features = umap_reducer.fit_transform(normalized_features).values
            elif HAS_UMAP:
                # Use CPU UMAP
                umap_reducer = umap.UMAP(
                    n_components=umap_n_components,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    random_state=42
                )
                reduced_features = umap_reducer.fit_transform(normalized_features)
            else:
                logging.warning("UMAP requested but not available. Skipping dimensionality reduction.")
                reduced_features = normalized_features
        else:
            reduced_features = normalized_features
        
        # Perform clustering
        if method == "kmeans":
            if use_gpu and HAS_CUML:
                # Use cuML GPU KMeans
                if isinstance(reduced_features, np.ndarray):
                    import cudf
                    reduced_features = cudf.DataFrame(reduced_features)
                clusterer = cumlKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(reduced_features).values.ravel()
            else:
                # Use sklearn CPU KMeans
                if hasattr(reduced_features, 'values'):
                    reduced_features = reduced_features.values
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(reduced_features)
                
        elif method == "hdbscan":
            if use_gpu and HAS_CUML:
                # Use cuML GPU HDBSCAN
                if isinstance(reduced_features, np.ndarray):
                    import cudf
                    reduced_features = cudf.DataFrame(reduced_features)
                clusterer = cumlHDBSCAN(min_cluster_size=max(2, h*w//20), min_samples=1)
                cluster_labels = clusterer.fit_predict(reduced_features).values.ravel()
            else:
                # Use sklearn CPU HDBSCAN
                if hasattr(reduced_features, 'values'):
                    reduced_features = reduced_features.values
                clusterer = HDBSCAN(min_cluster_size=max(2, h*w//20), min_samples=1)
                cluster_labels = clusterer.fit_predict(reduced_features)
                
            # Handle noise points (label -1)
            cluster_labels[cluster_labels == -1] = np.max(cluster_labels) + 1
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Ensure cluster_labels is numpy array
        if hasattr(cluster_labels, 'values'):
            cluster_labels = cluster_labels.values
        if hasattr(cluster_labels, 'ravel'):
            cluster_labels = cluster_labels.ravel()
        
        # Reshape back to spatial map
        cluster_map = cluster_labels.reshape(h, w)
        
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
                    n_clusters: int = 3, clustering_method: str = "kmeans",
                    apply_smoothing: bool = True, use_umap: bool = False,
                    use_gpu: bool = False, umap_n_components: int = 50,
                    umap_n_neighbors: int = 15, umap_min_dist: float = 0.1) -> Dict:
        """
        Process a single tile: extract tokens, cluster, and save results.
        """
        tile_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Load image
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        
        # Extract token features
        token_features = self.extract_token_features(image_pil)
        
        # Convert to spatial map
        spatial_features, tokens_h, tokens_w = self.tokens_to_spatial_map(token_features)
          # Cluster tokens
        cluster_map = self.cluster_tokens(
            spatial_features, n_clusters, clustering_method, 
            use_umap=use_umap, use_gpu=use_gpu, 
            umap_n_components=umap_n_components,
            umap_n_neighbors=umap_n_neighbors, 
            umap_min_dist=umap_min_dist
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
        cluster_img = (cluster_map_full * (255 // np.max(cluster_map_full))).astype(np.uint8)
        Image.fromarray(cluster_img).save(cluster_path)
        
        # Save cluster assignments as JSON
        cluster_info = {
            "tile_name": tile_name,
            "n_clusters": int(np.max(cluster_map_full) + 1),
            "token_grid_size": [tokens_h, tokens_w],
            "clustering_method": clustering_method,
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
    
    def create_cluster_visualization(self, image: np.ndarray, cluster_map: np.ndarray, save_path: str):
        """Create and save cluster visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Cluster map
        im = axes[1].imshow(cluster_map, cmap='tab10')
        axes[1].set_title("Token Clusters")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        overlay = image.copy().astype(float) / 255.0
        cluster_colored = plt.cm.tab10(cluster_map / np.max(cluster_map))[:, :, :3]
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
      parser = argparse.ArgumentParser(description="Token clustering for histology tiles")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to tiled images (256x256 preferred).")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path for clustering output.")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-large",
                       help="Model name for feature extraction.")
    parser.add_argument("--n_clusters", type=int, default=3,
                       help="Number of clusters (for k-means).")
    parser.add_argument("--clustering_method", type=str, default="kmeans",
                       choices=["kmeans", "hdbscan"],
                       help="Clustering method.")
    parser.add_argument("--no_smoothing", action="store_true",
                       help="Disable CRF/majority smoothing.")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu).")
    
    # GPU and UMAP arguments
    parser.add_argument("--use_gpu", action="store_true",
                       help="Use GPU acceleration for clustering (requires cuML).")
    parser.add_argument("--use_umap", action="store_true",
                       help="Apply UMAP dimensionality reduction before clustering.")
    parser.add_argument("--umap_n_components", type=int, default=50,
                       help="Target dimensions for UMAP reduction.")
    parser.add_argument("--umap_n_neighbors", type=int, default=15,
                       help="UMAP n_neighbors parameter.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                       help="UMAP min_dist parameter.")
    
    args = parser.parse_args()
    
    if not HAS_TRANSFORMERS:
        logging.error("transformers is required but not installed.")
        return
    
    logging.info("Starting token clustering with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize pipeline
    try:
        pipeline = TokenClusteringPipeline(
            model_name=args.model_name,
            device=args.device
        )
        logging.info(f"Initialized token clustering pipeline with {args.model_name}")
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
                n_clusters=args.n_clusters,
                clustering_method=args.clustering_method,
                apply_smoothing=not args.no_smoothing,
                use_umap=args.use_umap,
                use_gpu=args.use_gpu,
                umap_n_components=args.umap_n_components,
                umap_n_neighbors=args.umap_n_neighbors,
                umap_min_dist=args.umap_min_dist
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
            "clustering_method": args.clustering_method,
            "n_clusters": args.n_clusters,
            "results": all_results
        }, f, indent=2)
    
    logging.info(f"Token clustering completed. Processed {len(all_results)} tiles. Output saved to: %s", args.output_path)

if __name__ == "__main__":
    main()
