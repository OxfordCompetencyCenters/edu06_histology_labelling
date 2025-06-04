import argparse
import os
import glob
import logging

import torch

from token_clustering import TokenClusteringPipeline


def main():
    """Main clustering function using enhanced token clustering."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Advanced histology clustering using vision transformers")
    parser.add_argument("--segmentation_path", type=str, required=True, help="Path to segmentation results")
    parser.add_argument("--prepped_tiles_path", type=str, required=True, help="Path to preprocessed tiles")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for cluster results")
    
    # Token clustering parameters
    parser.add_argument("--token_model", type=str, default="dinov2", choices=["dinov2", "uni"],
                       help="Vision transformer model for feature extraction")
    parser.add_argument("--clustering_method", type=str, default="kmeans", choices=["kmeans", "hdbscan"],
                       help="Clustering algorithm for tokens")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for k-means")
    parser.add_argument("--use_crf", action="store_true", help="Apply CRF smoothing for spatial consistency")

    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize token clustering pipeline
    pipeline = TokenClusteringPipeline(
        model_name=args.token_model,
        clustering_method=args.clustering_method,
        n_clusters=args.n_clusters,
        use_crf=args.use_crf,
        device=device
    )

    # Get all slides
    slide_dirs = [d for d in os.listdir(args.prepped_tiles_path) 
                  if os.path.isdir(os.path.join(args.prepped_tiles_path, d))]
    
    for slide_name in slide_dirs:
        logging.info(f"Processing slide: {slide_name}")
        
        slide_tiles_dir = os.path.join(args.prepped_tiles_path, slide_name)
        slide_output_dir = os.path.join(args.output_path, slide_name)
        os.makedirs(slide_output_dir, exist_ok=True)
        
        # Get all tile paths
        tile_paths = glob.glob(os.path.join(slide_tiles_dir, "*.png"))
        if not tile_paths:
            logging.warning(f"No tiles found for slide {slide_name}")
            continue
        
        try:
            # Process slide with token clustering
            logging.info("Using Token Clustering Pipeline")
            results = pipeline.process_slide(
                tile_paths=tile_paths,
                slide_name=slide_name,
                output_dir=slide_output_dir
            )
            
            logging.info(f"Token clustering results: {len(results.get('cluster_assignments', []))} tiles processed")
                
        except Exception as e:
            logging.error(f"Failed to process slide {slide_name}: {e}")
            continue

    logging.info("Clustering complete!")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
