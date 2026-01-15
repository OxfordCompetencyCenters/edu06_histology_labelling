#!/usr/bin/env python

import os
import json
import argparse
import random
import cv2
import numpy as np
import hashlib # Used for generating consistent colors
import glob

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

def resolve_local_image_path(tile_path, images_dir):
    """
    Resolve the tile_path to find the corresponding local image file.
    
    Args:
        tile_path: The tile path from the JSON (could be absolute or relative)
        images_dir: Base directory containing the local images
    
    Returns:
        The resolved local image path
    """
    # Try different path resolution strategies
    
    # Strategy 1: Extract filename and search in images_dir structure
    filename = os.path.basename(tile_path)
    
    # Strategy 2: Look for common path markers and extract relative path
    path_markers = [
        "INPUT_prepped_tiles_path/",
        "prepped_tiles_path/", 
        "/prepped_tiles/",
        "/tiles/"
    ]
    
    subpath = filename  # Default fallback
    
    for marker in path_markers:
        if marker in tile_path:
            parts = tile_path.split(marker, 1)
            if len(parts) > 1:
                subpath = parts[1].lstrip("/\\")
                break
    
    # Strategy 3: Try to find the file by searching in the images directory
    potential_paths = [
        os.path.join(images_dir, subpath),
        os.path.join(images_dir, filename),
        # Try with slide name subdirectory structure
        os.path.join(images_dir, os.path.dirname(subpath), filename) if '/' in subpath or '\\' in subpath else None
    ]
    
    # Filter out None values
    potential_paths = [p for p in potential_paths if p is not None]
    
    # Return the first existing path
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    # If nothing found, return the most likely path for error reporting
    return os.path.join(images_dir, subpath)

def find_annotation_files(json_path):
    """
    Find annotation JSON files from either a direct file path or a directory.
    Returns a dictionary mapping slide names to file paths for per-slide processing.
    
    Args:
        json_path: Path to a JSON file or directory containing JSON files
        
    Returns:
        Dictionary mapping slide names to annotation JSON file paths
    """
    annotation_files = {}
    
    if os.path.isfile(json_path):
        # Single file - extract slide name from parent directory or use 'global'
        parent_dir = os.path.basename(os.path.dirname(json_path))
        slide_name = parent_dir if parent_dir and parent_dir != '.' else 'global'
        annotation_files[slide_name] = json_path
        return annotation_files
    
    if os.path.isdir(json_path):
        # Look for per-slide structure: slide_name/annotation_file.json
        for slide_dir in os.listdir(json_path):
            slide_path = os.path.join(json_path, slide_dir)
            if os.path.isdir(slide_path):
                # Look for common annotation file patterns in this slide directory
                patterns = [
                    "*.json",
                    "*annotations.json",
                    "*_annotations.json",
                    "final_annotations.json",
                    "v1_*_annotations.json",
                    "filtered_annotations.json"
                ]
                
                for pattern in patterns:
                    matches = glob.glob(os.path.join(slide_path, pattern))
                    if matches:
                        # Prefer files with "annotation" in the name
                        annotation_files_in_dir = [f for f in matches if "annotation" in os.path.basename(f).lower()]
                        if annotation_files_in_dir:
                            annotation_files[slide_dir] = annotation_files_in_dir[0]
                            break
                        else:
                            annotation_files[slide_dir] = matches[0]
                            break
        
        # Fallback: look for a single global file in the root directory
        if not annotation_files:
            patterns = [
                "*.json",
                "*annotations.json", 
                "*_annotations.json",
                "final_annotations.json",
                "v1_*_annotations.json",
                "filtered_annotations.json"
            ]
            
            for pattern in patterns:
                matches = glob.glob(os.path.join(json_path, pattern))
                if matches:
                    annotation_files_in_root = [f for f in matches if "annotation" in os.path.basename(f).lower()]
                    if annotation_files_in_root:
                        annotation_files['global'] = annotation_files_in_root[0]
                        break
                    else:
                        annotation_files['global'] = matches[0]
                        break
    
    if not annotation_files:
        raise FileNotFoundError(f"No JSON annotation files found at {json_path}")
    
    return annotation_files

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

        # Use improved path resolution
        local_image_path = resolve_local_image_path(tile_path, images_dir)
        subpath = os.path.relpath(local_image_path, images_dir)

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
            if not no_text and drawn_shape: # Only add text if enabled and a shape was drawn
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
                    
                    # Determine text position - prefer bbox, fallback to polygon centroid
                    if bbox:
                        x_min, y_min, _, _ = map(int, bbox)
                        text_pos = (max(0, x_min), max(0, y_min - 10)) # Position above top-left
                    elif polygon:
                        # Calculate polygon centroid for text positioning
                        polygon_array = np.array(polygon)
                        centroid_x = int(np.mean(polygon_array[:, 0]))
                        centroid_y = int(np.mean(polygon_array[:, 1]))
                        text_pos = (max(0, centroid_x), max(0, centroid_y - 10))
                    else:
                        continue # Skip if no positioning reference available

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
        )    # Input/Output Arguments
    parser.add_argument("--json_file", type=str, 
                        help="Path to the label JSON file.")
    parser.add_argument("--json_dir", type=str,
                        help="Directory containing the label JSON file.")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing the source images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the annotated images.")    # Annotation Content Control
    parser.add_argument("--max_labels", type=int, default=100,
                        help="Maximum number of labels (shapes) to draw per image.")
    parser.add_argument("--random_labels", action="store_true",
                        help="Pick labels randomly up to max_labels from the (potentially filtered) list.")

    # Shape Drawing Control
    parser.add_argument("--draw_bbox", action="store_true",
                        help="If set, draw bounding boxes.")
    parser.add_argument("--draw_polygon", action="store_true", default=True,
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
    parser.add_argument("--color_by", type=str, default="cluster_id",
                        choices=["pred_class", "cluster_id", "none"],
                        help="Attribute to use for color-coding shapes. 'none' uses default white.")

    # Filtering Control
    parser.add_argument("--filter_unclassified", action="store_true", default=True,
                        help="Filter out cells with pred_class='Unclassified' IF --text_use_pred_class or --color_by=pred_class is set.")


    args = parser.parse_args()

    # Determine JSON file paths (per-slide structure)
    if args.json_file and args.json_dir:
        print("Error: Specify either --json_file or --json_dir, not both.")
        return
    elif args.json_file:
        # Single file - treat as one slide
        slide_name = os.path.basename(os.path.dirname(args.json_file))
        if not slide_name or slide_name == '.':
            slide_name = 'global'
        annotation_files = {slide_name: args.json_file}
    elif args.json_dir:
        try:
            annotation_files = find_annotation_files(args.json_dir)
            print(f"Found annotation files for slides: {list(annotation_files.keys())}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    else:
        print("Error: Must specify either --json_file or --json_dir.")
        return

    # Basic validation
    if not (args.draw_bbox or args.draw_polygon):
         print("Warning: Neither --draw_bbox nor --draw_polygon specified. No shapes will be drawn.")

    # Check if text flags are specified without --no_text
    text_flags_specified = args.text_use_pred_class or args.text_use_cluster_id or args.text_use_cluster_confidence
    
    if not args.no_text and not text_flags_specified and (args.draw_bbox or args.draw_polygon):
        print("Info: No text components (--text_use_*) selected. Only shapes will be drawn.")

    if args.no_text and text_flags_specified:
        print("Warning: --no_text is set but text flags (--text_use_*) are also specified. Text flags will be ignored.")
    elif text_flags_specified and not args.no_text:
        print(f"Info: Text will be displayed with the following components: " +
              f"{'pred_class ' if args.text_use_pred_class else ''}" +
              f"{'cluster_id ' if args.text_use_cluster_id else ''}" +
              f"{'cluster_confidence ' if args.text_use_cluster_confidence else ''}".strip())

    # Auto-enable common defaults based on provided example command
    if not any([args.draw_bbox, args.draw_polygon]):
        # Default to drawing polygons if available
        args.draw_polygon = True
        print("Info: Auto-enabled --draw_polygon since no drawing mode was specified.")

    # Process each slide separately
    print(f"Processing {len(annotation_files)} slide(s)...")
    for slide_name, json_path in annotation_files.items():
        print(f"\nProcessing slide: {slide_name}")
        
        # Create slide-specific output directory
        slide_output_dir = os.path.join(args.output_dir, slide_name)
        
        annotate_images(
            json_file=json_path,
            images_dir=args.images_dir,
            output_dir=slide_output_dir,
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
    
    print(f"\nCompleted processing all slides. Annotated images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()