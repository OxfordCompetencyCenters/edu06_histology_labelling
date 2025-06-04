#!/usr/bin/env python3
"""
Setup script for downloading required model checkpoints and verifying the environment.
"""

import sys
import urllib.request
import hashlib
import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ModelSetup:
    """Helper class for setting up model checkpoints and dependencies."""
    
    def __init__(self, download_dir: str = "./models"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Model checkpoint information
        self.model_info = {
            "sam_vit_h": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "filename": "sam_vit_h_4b8939.pth",
                "sha256": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
                "size_mb": 2564
            },
            "sam_vit_l": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "filename": "sam_vit_l_0b3195.pth", 
                "sha256": "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622",
                "size_mb": 1249
            },
            "sam_vit_b": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "filename": "sam_vit_b_01ec64.pth",
                "sha256": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
                "size_mb": 375
            }
        }
    
    def check_dependencies(self) -> List[str]:
        """Check if required Python packages are installed."""
        required_packages = [
            "torch", "torchvision", "transformers", "PIL", "cv2", 
            "numpy", "sklearn", "matplotlib", "openslide"
        ]
        
        optional_packages = [
            ("segment_anything", "SAM-Med segmentation"),
            ("pydensecrf", "CRF smoothing"),
            ("hdbscan", "HDBSCAN clustering"),
            ("cuml", "GPU-accelerated clustering")
        ]
        
        missing_required = []
        missing_optional = []
        
        for package in required_packages:
            try:
                if package == "cv2":
                    import cv2
                elif package == "PIL":
                    from PIL import Image
                elif package == "sklearn":
                    import sklearn
                else:
                    __import__(package)
                logging.info(f"✓ {package} is available")
            except ImportError:
                missing_required.append(package)
                logging.error(f"✗ {package} is missing (required)")
        
        for package, description in optional_packages:
            try:
                __import__(package)
                logging.info(f"✓ {package} is available")
            except ImportError:
                missing_optional.append((package, description))
                logging.warning(f"✗ {package} is missing (optional - {description})")
        
        if missing_optional:
            logging.info("\nOptional packages can be installed with:")
            for package, description in missing_optional:
                if package == "segment_anything":
                    logging.info(f"  pip install {package}")
                elif package == "pydensecrf":
                    logging.info(f"  pip install {package}")
                elif package == "hdbscan":
                    logging.info(f"  pip install {package}")
                elif package == "cuml":
                    logging.info(f"  conda install -c rapidsai cuml")
        
        return missing_required
    
    def download_file(self, url: str, filepath: Path, expected_size_mb: int = None) -> bool:
        """Download a file with progress tracking."""
        try:
            logging.info(f"Downloading {filepath.name}...")
            
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, (downloaded / total_size) * 100)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
            
            urllib.request.urlretrieve(url, filepath, progress_hook)
            print()  # New line after progress
            
            # Verify file size
            actual_size_mb = filepath.stat().st_size / (1024 * 1024)
            if expected_size_mb and abs(actual_size_mb - expected_size_mb) > 10:
                logging.warning(f"File size mismatch: expected ~{expected_size_mb}MB, got {actual_size_mb:.1f}MB")
            
            logging.info(f"✓ Downloaded {filepath.name} ({actual_size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            logging.error(f"✗ Failed to download {filepath.name}: {e}")
            return False
    
    def verify_checksum(self, filepath: Path, expected_sha256: str) -> bool:
        """Verify file integrity using SHA256."""
        try:
            logging.info(f"Verifying {filepath.name}...")
            
            sha256_hash = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            actual_sha256 = sha256_hash.hexdigest()
            
            if actual_sha256 == expected_sha256:
                logging.info(f"✓ Checksum verified for {filepath.name}")
                return True
            else:
                logging.error(f"✗ Checksum mismatch for {filepath.name}")
                logging.error(f"  Expected: {expected_sha256}")
                logging.error(f"  Actual:   {actual_sha256}")
                return False
                
        except Exception as e:
            logging.error(f"✗ Failed to verify {filepath.name}: {e}")
            return False
    
    def setup_sam_models(self, models: List[str] = ["sam_vit_h"]) -> bool:
        """Download and verify SAM model checkpoints."""
        success = True
        
        for model_name in models:
            if model_name not in self.model_info:
                logging.error(f"Unknown model: {model_name}")
                success = False
                continue
            
            model_config = self.model_info[model_name]
            filepath = self.download_dir / model_config["filename"]
            
            # Skip if already exists and verified
            if filepath.exists():
                if self.verify_checksum(filepath, model_config["sha256"]):
                    logging.info(f"✓ {model_name} already available and verified")
                    continue
                else:
                    logging.warning(f"Removing corrupted file: {filepath}")
                    filepath.unlink()
            
            # Download the model
            if self.download_file(
                model_config["url"], 
                filepath, 
                model_config["size_mb"]
            ):
                # Verify checksum
                if not self.verify_checksum(filepath, model_config["sha256"]):
                    success = False
            else:
                success = False
        
        return success
    
    def create_config_file(self):
        """Create a configuration file with model paths."""
        config = {
            "model_paths": {},
            "recommended_settings": {
                "sam_med": {
                    "model_type": "vit_h",
                    "checkpoint": str(self.download_dir / "sam_vit_h_4b8939.pth")
                },
                "token_clustering": {
                    "model_name": "facebook/dinov2-large",
                    "n_clusters": 3,
                    "clustering_method": "kmeans"
                },
                "data_preparation": {
                    "tile_sizes": [256, 512],
                    "target_mpp": 0.5,
                    "create_patches": True
                }
            }
        }
        
        # Add available model paths
        for model_name, model_config in self.model_info.items():
            filepath = self.download_dir / model_config["filename"]
            if filepath.exists():
                config["model_paths"][model_name] = str(filepath)
        
        config_path = Path("pipeline_config.json")
        import json
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"✓ Configuration saved to {config_path}")
    
    def run_setup(self, sam_models: List[str] = ["sam_vit_h"]):
        """Run the complete setup process."""
        logging.info("Starting setup for Enhanced Histology Analysis Pipeline...")
        
        # Check dependencies
        logging.info("\n1. Checking dependencies...")
        missing = self.check_dependencies()
        
        if missing:
            logging.error(f"\nMissing required packages: {missing}")
            logging.error("Please install missing packages and run setup again.")
            return False
        
        # Setup SAM models
        logging.info(f"\n2. Setting up SAM models: {sam_models}")
        if not self.setup_sam_models(sam_models):
            logging.error("Failed to setup SAM models")
            return False
        
        # Create configuration
        logging.info("\n3. Creating configuration file...")
        self.create_config_file()
        
        # Test imports
        logging.info("\n4. Testing pipeline components...")
        test_success = self.test_pipeline_components()
        
        if test_success:
            logging.info("\n✓ Setup completed successfully!")
            logging.info("\nYou can now run the pipeline with:")
            logging.info("  python integrated_pipeline.py --input_data /path/to/slides --output_path /path/to/output")
        else:
            logging.error("\n✗ Setup completed with warnings. Some components may not work.")
        
        return test_success
    
    def test_pipeline_components(self) -> bool:
        """Test if pipeline components can be imported and initialized."""
        tests = [
            ("Data preparation", self.test_data_prep),
            ("SAM-Med segmentation", self.test_sam_med),
            ("Token clustering", self.test_token_clustering),
            ("Traditional clustering", self.test_traditional_clustering)
        ]
        
        success = True
        for test_name, test_func in tests:
            try:
                if test_func():
                    logging.info(f"✓ {test_name}: OK")
                else:
                    logging.warning(f"⚠ {test_name}: Failed")
                    success = False
            except Exception as e:
                logging.warning(f"⚠ {test_name}: Error - {e}")
                success = False
        
        return success
    
    def test_data_prep(self) -> bool:
        """Test data preparation components."""
        try:
            import openslide
            from PIL import Image
            import numpy as np
            return True
        except ImportError:
            return False
    
    def test_sam_med(self) -> bool:
        """Test SAM-Med components."""
        try:
            # Check if segment-anything is available
            import segment_anything
            
            # Check if at least one SAM checkpoint exists
            for model_config in self.model_info.values():
                filepath = self.download_dir / model_config["filename"]
                if filepath.exists():
                    return True
            return False
        except ImportError:
            return False
    
    def test_token_clustering(self) -> bool:
        """Test token clustering components."""
        try:
            import transformers
            from transformers import AutoModel, AutoImageProcessor
            import torch
            return True
        except ImportError:
            return False
    
    def test_traditional_clustering(self) -> bool:
        """Test traditional clustering components."""
        try:
            import torch
            import torchvision
            from sklearn.cluster import DBSCAN, KMeans
            return True
        except ImportError:
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Enhanced Histology Analysis Pipeline")
    parser.add_argument("--models", nargs="+", 
                       choices=["sam_vit_h", "sam_vit_l", "sam_vit_b"],
                       default=["sam_vit_h"],
                       help="SAM models to download (default: sam_vit_h)")
    parser.add_argument("--download_dir", type=str, default="./models",
                       help="Directory to store model checkpoints")
    parser.add_argument("--skip_download", action="store_true",
                       help="Skip model download, only check dependencies")
    
    args = parser.parse_args()
    
    setup = ModelSetup(download_dir=args.download_dir)
    
    if args.skip_download:
        logging.info("Checking dependencies only...")
        missing = setup.check_dependencies()
        if missing:
            logging.error(f"Missing required packages: {missing}")
            sys.exit(1)
        else:
            logging.info("✓ All dependencies are available")
            sys.exit(0)
    else:
        success = setup.run_setup(sam_models=args.models)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
