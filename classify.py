import argparse
import os
import glob
import json
import logging

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

def load_classifier(num_classes=4):
    """
    Loads a pretrained ResNet18 and replaces the final layer with a smaller head.
    Example for 4 classes: [Lymphocyte, Epithelial, Fibroblast, Other].
    Replace with your real classes and your own checkpoint if you have one.
    """
    logging.info("Loading classifier (ResNet18) with num_classes=%d", num_classes)
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def classify_bbox(tile_path, bbox, model, transform, device):
    """
    Crop the region from tile_path using bbox, transform it, run model, return predicted label.
    bbox format: [xmin, ymin, xmax, ymax]
    """
    tile_img = Image.open(tile_path).convert("RGB")
    xmin, ymin, xmax, ymax = bbox
    cell_crop = tile_img.crop((xmin, ymin, xmax, ymax))

    input_tensor = transform(cell_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        preds = torch.argmax(logits, dim=1).item()
    return preds

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--segmented_path", type=str,
                        help="Path to segmentation results (masks + *_bboxes.json).")
    parser.add_argument("--prepped_tiles_path", type=str,
                        help="Path to the original prepped tile images.")
    parser.add_argument("--output_path", type=str,
                        help="Path for classification results.")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of classes for classification.")
    args = parser.parse_args()

    logging.info("Starting classification with arguments: %s", args)
    os.makedirs(args.output_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_classifier(num_classes=args.num_classes)
    model = model.to(device)
    model.eval()

    # Search for all bounding box files under segmented_path
    bbox_files = glob.glob(os.path.join(args.segmented_path, "**/*_bboxes.json"), recursive=True)
    if not bbox_files:
        logging.warning("No bounding box JSON files found under: %s", args.segmented_path)
        return

    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    classification_results = []

    for bbox_file in bbox_files:
        with open(bbox_file, "r") as f:
            bboxes = json.load(f)

        tile_name = os.path.basename(bbox_file).replace("_bboxes.json", ".png")
        parent_dir = os.path.basename(os.path.dirname(bbox_file))
        tile_path = os.path.join(args.prepped_tiles_path, parent_dir, tile_name)

        if not os.path.exists(tile_path):
            logging.warning("Tile image not found at %s. Skipping.", tile_path)
            continue
        logging.info("Classifying bboxes for tile: %s", tile_path)

        tile_results = []
        for bbox_entry in bboxes:
            label_id = bbox_entry["label_id"]
            bbox = bbox_entry["bbox"]
            pred_class = classify_bbox(tile_path, bbox, model, transform, device)
            tile_results.append({
                "label_id": label_id,
                "bbox": bbox,
                "pred_class": pred_class
            })

        classification_results.append({
            "tile_path": tile_path,
            "classified_cells": tile_results
        })

    # Write out a single JSON with classification results
    out_file = os.path.join(args.output_path, "classification_results.json")
    with open(out_file, "w") as f:
        json.dump(classification_results, f, indent=2)

    logging.info("Classification step done. Output at: %s", out_file)

if __name__ == "__main__":
    main()