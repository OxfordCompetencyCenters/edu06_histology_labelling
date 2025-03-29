import argparse
import json
import os
import glob
import logging
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# CPU fallback for DBSCAN
from sklearn.cluster import DBSCAN as CPUDbscan

def load_pretrained_resnet50(device: torch.device) -> nn.Module:
    """
    Loads a pretrained ResNet50, replaces final FC with identity,
    returns the model in eval mode on the specified device.
    """
    model = models.resnet50(pretrained=True)
    # Replace the final FC layer with an identity so we get a 2048-dim embedding
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model

def get_image_transform() -> transforms.Compose:
    """
    Returns the standard transforms for a ResNet pretrained on ImageNet.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def extract_patch_embedding(
    tile_img: Image.Image,
    bbox: List[int],
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device
) -> np.ndarray:
    """
    Crop the bounding box from the tile, apply transforms,
    pass through the model to get a 2048-d embedding.
    """
    xmin, ymin, xmax, ymax = bbox
    patch = tile_img.crop((xmin, ymin, xmax, ymax))
    input_tensor = transform(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(input_tensor)
    return embedding.cpu().numpy().squeeze()

def compute_cluster_confidence(
    embedding: np.ndarray,
    centroid: np.ndarray
) -> float:
    """
    A simple measure of "confidence" in cluster assignment:
    confidence = 1 / (1 + distance_to_centroid).
    """
    dist = np.linalg.norm(embedding - centroid)
    confidence = 1.0 / (1.0 + dist)
    return float(confidence)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation_path", type=str, required=True,
                        help="Path where segment.py wrote bounding box JSON and tile PNGs.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to write cluster outputs.")
    parser.add_argument("--gpu", action="store_true",
                        help="If set and GPU is available, try GPU-based DBSCAN.")
    # DBSCAN parameters
    parser.add_argument("--eps", type=float, default=0.5,
                        help="DBSCAN eps parameter.")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="DBSCAN min_samples parameter.")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    logging.info(f"DBSCAN clustering bounding boxes from {args.segmentation_path}")
    logging.info(f"Output path: {args.output_path}")

    # Find bounding box JSON files
    bbox_json_files = glob.glob(os.path.join(args.segmentation_path, "**/*_bboxes.json"), recursive=True)
    logging.info(f"Found {len(bbox_json_files)} bounding-box JSON files.")
    if not bbox_json_files:
        logging.warning("No bounding-box JSON files found. Exiting.")
        return

    # Decide on device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info(f"Using device: {device}")

    model = load_pretrained_resnet50(device)
    transform = get_image_transform()

    embeddings = []
    record_tracker = []

    for bbox_file in bbox_json_files:
        base_dir = os.path.dirname(bbox_file)
        tile_name = os.path.basename(bbox_file).replace("_bboxes.json", ".png")
        tile_path = os.path.join(base_dir, tile_name)

        if not os.path.exists(tile_path):
            logging.warning(f"Tile image not found for {bbox_file}, skipping.")
            continue

        with open(bbox_file, "r") as f:
            bboxes = json.load(f)

        tile_img = Image.open(tile_path).convert("RGB")

        for bb_dict in bboxes:
            label_id = bb_dict["label_id"]
            bbox = bb_dict["bbox"]  # [xmin, ymin, xmax, ymax]
            emb = extract_patch_embedding(tile_img, bbox, model, transform, device)
            embeddings.append(emb)
            record_tracker.append((bbox_file, label_id, bbox))

    embeddings = np.array(embeddings, dtype=np.float32)
    logging.info(f"Extracted embeddings for {len(embeddings)} bounding boxes.")
    if len(embeddings) == 0:
        logging.warning("No embeddings extracted. Exiting.")
        return

    # Attempt GPU-based DBSCAN via cuML if available
    if device.type == "cuda":
        try:
            import cupy as cp
            from cuml.cluster import DBSCAN as GPUDbscan
            logging.info("Using GPU-based DBSCAN with cuML.")
            embeddings_gpu = cp.asarray(embeddings)
            dbscan = GPUDbscan(eps=args.eps, min_samples=args.min_samples)
            dbscan.fit(embeddings_gpu)
            labels = dbscan.labels_.get()
        except ImportError:
            logging.warning("cuml not installed. Falling back to CPU-based DBSCAN.")
            dbscan = CPUDbscan(eps=args.eps, min_samples=args.min_samples)
            dbscan.fit(embeddings)
            labels = dbscan.labels_
    else:
        logging.info("Using CPU-based DBSCAN.")
        dbscan = CPUDbscan(eps=args.eps, min_samples=args.min_samples)
        dbscan.fit(embeddings)
        labels = dbscan.labels_

    # Build cluster -> list of embeddings
    cluster_to_embs: Dict[int, list] = {}
    for idx, lbl in enumerate(labels):
        if lbl not in cluster_to_embs:
            cluster_to_embs[lbl] = []
        cluster_to_embs[lbl].append(embeddings[idx])

    # Compute centroid for each cluster (excluding noise)
    centroids = {}
    for cluster_id, emb_list in cluster_to_embs.items():
        if cluster_id == -1:
            continue  # -1 is noise
        emb_array = np.stack(emb_list)
        centroids[cluster_id] = np.mean(emb_array, axis=0)

    # Prepare output JSON
    cluster_assignments = []
    for i, emb in enumerate(embeddings):
        cluster_id = labels[i]
        bbox_file, label_id, bbox = record_tracker[i]

        if cluster_id == -1:
            # Noise => confidence = 0 or None
            confidence = 0.0
        else:
            centroid = centroids[cluster_id]
            confidence = compute_cluster_confidence(emb, centroid)

        cluster_assignments.append({
            "bbox_file": bbox_file,
            "label_id": label_id,
            "bbox": bbox,
            "cluster_id": int(cluster_id),
            "confidence": confidence
        })

    out_json_path = os.path.join(args.output_path, "cluster_assignments.json")
    with open(out_json_path, "w") as f:
        json.dump(cluster_assignments, f, indent=2)

    # Summarize number of clusters found
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = list(labels).count(-1)
    logging.info(
        f"DBSCAN found {n_clusters} clusters and {n_noise} noise points. "
        f"Wrote cluster_assignments.json to {out_json_path}"
    )

if __name__ == "__main__":
    main()
