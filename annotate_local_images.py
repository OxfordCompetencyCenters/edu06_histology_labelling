#!/usr/bin/env python

import os
import json
import argparse
import random
import cv2
import numpy as np
import hashlib # Used for generating consistent colors

# --- Color Generation ---
# Cache for generated colors to ensure consistency within a run
_color_cache = {}

def generate_color_from_value(value):
    """
    Generates a visually distinct BGR color based on any value (string, int, etc.).
    Uses hashing to create somewhat predictable but varied colors.
    Avoids very dark or very light colors for better visibility.
    Handles None or "N/A" gracefully with a default color (e.g., gray).
    """
    if value is None or value == "N/A":
        # Assign a default neutral color like gray for missing/invalid values
        return (128, 128, 128)
        
    if value not in _color_cache:
        # Use SHA256 hash for better distribution than simple hash()
        hasher = hashlib.sha256(str(value).encode('utf-8'))
        hash_bytes = hasher.digest()
        # Use parts of the hash to generate B, G, R values
        # Ensure values are within a visible range (e.g., 50-255)
        b = 50 + hash_bytes[0] % 206 # Modulo 206 + 50 gives range [50, 255]
        g = 50 + hash_bytes[1] % 206
        r = 50 + hash_bytes[2] % 206
        _color_cache[value] = (b, g, r)
    return _color_cache[value]

# --- Drawing Functions ---
def draw_polygon(img, polygon_points, color=(0, 255, 0), thickness=2):
    """
    Draws a polygon on the image using the provided list of (x, y) points.
    polygon_points is a list of [x, y] coordinates.
    """
    if not polygon_points: return # Avoid error if polygon list is empty
    pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def draw_bbox(img, bbox, color=(255, 0, 0), thickness=2):
    """
    Draws a bounding box on the image, where bbox = [x_min, y_min, x_max, y_max].
    """
    if not bbox or len(bbox) != 4: return # Avoid error if bbox is invalid
    x_min, y_min, x_max, y_max = map(int, bbox) # Ensure integer coordinates
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

# --- Annotation Logic ---
def annotate_images(json_file, images_dir, output_dir,
                    max_labels, random_labels,
                    draw_bbox_flag, draw_polygon_flag,
                    text_scale,
                    text_use_pred_class, text_use_cluster_id, text_use_cluster_confidence, no_text,
                    color_by,
                    filter_unclassified): # Added filter_unclassified parameter
    """
    Reads the label JSON file, draws bounding boxes and/or polygons for each cell,
    then writes annotated images to the output directory.

    Text content and shape colors are controlled by flags.
    Optionally filters out cells with pred_class == 'Unclassified'.
    Optionally disables all text output.
    If random_labels is True, shuffles cells first, then truncates to max_labels.
    """
    global _color_cache # Use the global cache
    _color_cache.clear() # Clear cache for each run

    with open(json_file, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for item in data:
        tile_path = item.get("tile_path")
        if not tile_path:
            print(f"Warning: Missing 'tile_path' in item. Skipping.")
            continue

        # Extract the subpath (logic retained from previous version)
        base_path_marker = "INPUT_prepped_tiles_path/"
        subpath = os.path.basename(tile_path) # Default to basename
        if base_path_marker in tile_path:
             parts = tile_path.split(base_path_marker, 1)
             if len(parts) > 1:
                 subpath = parts[1].lstrip("/")
        else:
            print(f"Warning: '{base_path_marker}' not found in tile_path '{tile_path}'. Using filename '{subpath}'.")

        local_image_path = os.path.join(images_dir, subpath)

        if not os.path.exists(local_image_path):
            print(f"Warning: {local_image_path} does not exist. Skipping.")
            continue

        img = cv2.imread(local_image_path)
        if img is None:
            print(f"Warning: Failed to load image {local_image_path}. Skipping.")
            continue

        # Extract cells for this tile
        cells = item.get("cells", [])

        # --- Filtering Step ---
        if filter_unclassified:
            original_count = len(cells)
            # Filter out cells where pred_class is 'Unclassified'
            # This filter applies if EITHER text uses pred_class OR color is by pred_class
            cells = [
                cell for cell in cells
                if not ((text_use_pred_class or color_by == 'pred_class') and cell.get("pred_class") == 'Unclassified')
            ]
            filtered_count = len(cells)
            if original_count != filtered_count:
                 print(f"Filtered {original_count - filtered_count} 'Unclassified' cells for {subpath}")


        # Possibly shuffle the (potentially filtered) cells if random_labels is True
        if random_labels:
            random.shuffle(cells)

        # Draw bounding boxes/polygons for the selected cells
        label_count = 0
        for cell in cells:
            # Safely get cell data with defaults
            pred_class = cell.get("pred_class", "N/A")
            cluster_id = cell.get("cluster_id", "N/A") # Use string N/A for consistency
            cluster_confidence = cell.get("cluster_confidence") # Can be None
            bbox = cell.get("bbox")       # [x_min, y_min, x_max, y_max]
            polygon = cell.get("polygon") # list of [x, y]

            # Skip if essential geometry is missing for drawing requested shapes
            if draw_bbox_flag and not bbox:
                print(f"Warning: Skipping cell in {subpath} - Bbox drawing requested but bbox missing.")
                continue
            if draw_polygon_flag and not polygon:
                print(f"Warning: Skipping cell in {subpath} - Polygon drawing requested but polygon missing.")
                continue
            # Also skip if no shape is requested to be drawn for this cell
            if not draw_bbox_flag and not draw_polygon_flag:
                continue # Nothing to draw for this cell

            # If text is enabled, we need a bbox for positioning.
            # If text is desired but only polygon exists, we might skip or use polygon centroid (more complex)
            # For simplicity, require bbox if text is enabled (unless no_text is True)
            if not no_text and not bbox and \
               (text_use_pred_class or text_use_cluster_id or text_use_cluster_confidence):
                print(f"Warning: Skipping cell in {subpath} - Text requested but bbox missing for positioning.")
                continue


            # Determine the color based on the chosen attribute
            color_value = "default" # Default if coloring is off or attribute missing
            if color_by == "pred_class":
                color_value = pred_class
            elif color_by == "cluster_id":
                # Ensure cluster_id is treated as a string for consistent hashing/coloring
                color_value = str(cluster_id) if cluster_id is not None else "N/A"


            # Use white as default if no color_by is specified, otherwise generate color
            shape_color = (255, 255, 255) # Default: White
            if color_by != "none":
                 shape_color = generate_color_from_value(color_value)

            # Draw shapes if requested
            drawn_shape = False
            if draw_bbox_flag and bbox:
                draw_bbox(img, bbox, color=shape_color)
                drawn_shape = True

            if draw_polygon_flag and polygon:
                draw_polygon(img, polygon, color=shape_color)
                drawn_shape = True

            # --- Text Labeling (conditional) ---
            if not no_text and drawn_shape and bbox: # Only add text if enabled, a shape was drawn, and bbox exists
                label_parts = []
                if text_use_pred_class:
                    label_parts.append(f"c:{pred_class}")
                if text_use_cluster_id:
                    label_parts.append(f"cl:{cluster_id}")
                if text_use_cluster_confidence and cluster_confidence is not None:
                    try:
                        label_parts.append(f"p:{float(cluster_confidence):.2f}")
                    except (ValueError, TypeError):
                         label_parts.append(f"p:?") # Handle non-numeric confidence


                if label_parts: # Only proceed if there's something to write
                    label_str = ", ".join(label_parts)
                    x_min, y_min, _, _ = map(int, bbox)
                    text_pos = (max(0, x_min), max(0, y_min - 10)) # Position above top-left

                    # Add background rectangle for better readability
                    (text_width, text_height), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
                    # Adjust background rectangle calculations
                    bg_y1 = text_pos[1] + baseline
                    bg_y2 = text_pos[1] - text_height - baseline // 2 # Adjust for better fit
                    bg_x1 = text_pos[0]
                    bg_x2 = text_pos[0] + text_width

                    # Draw rectangle on a copy and blend
                    overlay = img.copy()
                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1) # Black background
                    alpha = 0.6 # Transparency factor
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


                    # Draw the actual text
                    cv2.putText(img,
                                label_str,
                                text_pos,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                text_scale,
                                (255, 255, 255), # White text for contrast
                                1,
                                cv2.LINE_AA)

            # Increment count only if a shape was actually drawn
            if drawn_shape:
                label_count += 1
                if label_count >= max_labels:
                    break

        # Save the annotated image
        out_filename = os.path.basename(subpath)
        # Ensure output filename doesn't have problematic characters if subpath had them
        out_filename_safe = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in out_filename)
        output_path = os.path.join(output_dir, out_filename_safe)

        try:
            success = cv2.imwrite(output_path, img)
            if success:
                print(f"Annotated image saved to {output_path} ({label_count} labels drawn)")
            else:
                print(f"Error: Failed to write image {output_path}")
        except Exception as e:
            print(f"Error saving image {output_path}: {e}")


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes and/or polygons on images with customizable text and colors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
        )

    # Input/Output Arguments
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to the label JSON file.")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing the source images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the annotated images.")

    # Annotation Content Control
    parser.add_argument("--max_labels", type=int, default=999999,
                        help="Maximum number of labels (shapes) to draw per image.")
    parser.add_argument("--random_labels", action="store_true",
                        help="Pick labels randomly up to max_labels from the (potentially filtered) list.")

    # Shape Drawing Control
    parser.add_argument("--draw_bbox", action="store_true",
                        help="If set, draw bounding boxes.")
    parser.add_argument("--draw_polygon", action="store_true",
                        help="If set, draw polygons.")

    # Text Content Control
    parser.add_argument("--no_text", action="store_true",
                        help="If set, do NOT draw any text labels on the image.")
    parser.add_argument("--text_use_pred_class", action="store_true",
                        help="Include predicted class in the text label (ignored if --no_text).")
    parser.add_argument("--text_use_cluster_id", action="store_true",
                        help="Include cluster ID in the text label (ignored if --no_text).")
    parser.add_argument("--text_use_cluster_confidence", action="store_true",
                        help="Include cluster confidence in the text label (ignored if --no_text).")
    parser.add_argument("--text_scale", type=float, default=0.5,
                        help="Scale factor for label text size (ignored if --no_text).")

    # Color Coding Control
    parser.add_argument("--color_by", type=str, default="pred_class",
                        choices=["pred_class", "cluster_id", "none"],
                        help="Attribute to use for color-coding shapes. 'none' uses default white.")

    # Filtering Control
    parser.add_argument("--filter_unclassified", action="store_true",
                        help="Filter out cells with pred_class='Unclassified' IF --text_use_pred_class or --color_by=pred_class is set.")


    args = parser.parse_args()

    # Basic validation
    if not (args.draw_bbox or args.draw_polygon):
         print("Warning: Neither --draw_bbox nor --draw_polygon specified. No shapes will be drawn.")

    if not args.no_text and not (args.text_use_pred_class or args.text_use_cluster_id or args.text_use_cluster_confidence) and (args.draw_bbox or args.draw_polygon):
        print("Info: No text components (--text_use_*) selected, but --no_text is not set. Only shapes will be drawn.")

    if args.no_text and (args.text_use_pred_class or args.text_use_cluster_id or args.text_use_cluster_confidence):
        print("Info: --no_text is set. Flags --text_use_* will be ignored.")


    annotate_images(
        json_file=args.json_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        max_labels=args.max_labels,
        random_labels=args.random_labels,
        draw_bbox_flag=args.draw_bbox,
        draw_polygon_flag=args.draw_polygon,
        text_scale=args.text_scale,
        text_use_pred_class=args.text_use_pred_class,
        text_use_cluster_id=args.text_use_cluster_id,
        text_use_cluster_confidence=args.text_use_cluster_confidence,
        no_text=args.no_text,
        color_by=args.color_by,
        filter_unclassified=args.filter_unclassified
    )

if __name__ == "__main__":
    main()