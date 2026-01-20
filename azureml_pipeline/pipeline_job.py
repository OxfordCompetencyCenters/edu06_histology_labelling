from __future__ import annotations
import argparse, logging, os, sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline


env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

SUBSCRIPTION_ID  = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP   = os.getenv("AZURE_RESOURCE_GROUP")
WORKSPACE_NAME   = os.getenv("AZURE_ML_WORKSPACE_NAME")
COMPUTE_CLUSTER  = os.getenv("AZURE_ML_COMPUTE_CLUSTER", "gpu-cluster")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def format_param_for_name(value):
    """Return a string safe for filenames ('.'→'pt', ','→'-')."""
    if value is None:
        return "auto"
    if isinstance(value, float):
        return f"{value}".replace(".", "pt")
    if isinstance(value, bool):
        return "T" if value else "F"
    return str(value).replace(".", "pt").replace(",", "-")

def build_param_string(args):
    """Compact signature of key hyper-params for folder/exp names."""
    mode = getattr(args, 'mode', 'full')
    parts = []
    
    # For cluster tiles and filtered annotations only modes
    if mode in ["extract_cluster_tiles_only", "cluster_tiles_and_filtered_annotations"]:
        if hasattr(args, 'enable_cluster_tiles') and args.enable_cluster_tiles:
            # reptiles = representative tiles
            parts.append(f"reptiles_{format_param_for_name(args.cluster_analyzer_confidence_threshold)}")
            if args.cluster_analyzer_max_items is not None:
                parts.append(f"maxper_{args.cluster_analyzer_max_items}")
        if hasattr(args, 'enable_filtered_annotations') and args.enable_filtered_annotations:
            parts.append("filteredAnnotated")
            parts.append(f"maxlabels_{args.filtered_annotation_max_labels}")
            if args.filtered_annotation_random_labels:
                parts.append("random")
            if args.filtered_annotation_draw_bbox:
                parts.append("bbox")
            if args.filtered_annotation_draw_polygon:
                parts.append("poly")
            if args.filtered_annotation_no_text:
                parts.append("notext")
            parts.append(f"colorby_{args.filtered_annotation_color_by}")
            if args.filtered_annotation_filter_unclassified:
                parts.append("filterunclass")
        return "_".join(parts) if parts else "default"
    
    # For annotation only mode
    elif mode == "annotate_only":
        if hasattr(args, 'enable_annotations') and args.enable_annotations:
            parts.append("annotated")
            parts.append(f"maxlabels_{args.annotation_max_labels}")
            if args.annotation_random_labels:
                parts.append("random")
            if args.annotation_draw_bbox:
                parts.append("bbox")
            if args.annotation_draw_polygon:
                parts.append("poly")
            if args.annotation_no_text:
                parts.append("notext")
            parts.append(f"colorby_{args.annotation_color_by}")
            if args.annotation_filter_unclassified:
                parts.append("filterunclass")
        return "_".join(parts) if parts else "default"
    
    # For all other modes (full pipeline parameters)
    else:
        parts = [
            f"model_{format_param_for_name(args.segment_model_type)}",
            f"prob_{format_param_for_name(args.segment_cellprob_threshold)}",
            f"flow_{format_param_for_name(args.segment_flow_threshold)}",
            f"eps_{format_param_for_name(args.cluster_eps)}",
            f"mins_{args.cluster_min_samples}",
        ]
        if args.segment_use_cellpose_sam:
            parts.append("cellpose_sam")
        if args.segment_use_gpu:
            parts.append("segGPU")
        if args.segment_diameter is not None:
            parts.append(f"diam_{format_param_for_name(args.segment_diameter)}")
        if args.segment_resample:
            parts.append("resample")
        if not args.segment_normalize:
            parts.append("nonorm")
        if args.segment_do_3D:
            parts.append("3D")
        if args.cluster_normalize:
            parts.append("norm")
        if args.cluster_use_gpu:
            parts.append("cluGPU")
        if args.cluster_use_umap:
            parts.append(f"umap_{args.cluster_umap_components}")
        if args.cluster_per_slide:
            parts.append("perslide")
        parts.append(f"mag_{format_param_for_name(args.magnifications)}")
        if args.num_tiles is not None:
            parts.append(f"ntiles_{args.num_tiles}")
        if args.filter_tiles:
            parts.append("filtered")
            parts.append(f"edge_{format_param_for_name(args.filter_min_edge_density)}")
        if hasattr(args, 'enable_annotations') and args.enable_annotations:
            parts.append("annotated")
        if hasattr(args, 'enable_cluster_tiles') and args.enable_cluster_tiles:
            parts.append(f"reptiles_{format_param_for_name(args.cluster_analyzer_confidence_threshold)}")
            if args.cluster_analyzer_max_items is not None:
                parts.append(f"maxper_{args.cluster_analyzer_max_items}")
        if hasattr(args, 'enable_filtered_annotations') and args.enable_filtered_annotations:
            parts.append("filteredAnnotated")
        return "_".join(parts)

def build_cluster_command(**kwargs) -> str:
    """Return the shell command for cluster.py with only the flags we need."""
    cmd  = (
        "python cluster.py "
        "--segmentation_path ${{inputs.segmentation_path}} "
        "--prepped_tiles_path ${{inputs.prepped_tiles_path}} "
        "--output_path ${{outputs.cluster_output}} "    )
    if kwargs["cluster_eps"] is not None:
        cmd += f"--eps {kwargs['cluster_eps']} "
    cmd += f"--min_samples {kwargs['cluster_min_samples']} "
    if kwargs["cluster_use_gpu"]:
        cmd += "--gpu "
    if kwargs["cluster_normalize"]:
        cmd += "--normalize_embeddings "
    if kwargs.get("cluster_per_slide", False):
        cmd += "--per_slide "
    if kwargs["cluster_use_umap"]:
        cmd += (
            "--use_umap "
            f"--umap_n_components {kwargs['cluster_umap_components']} "
            f"--umap_n_neighbors {kwargs['cluster_umap_neighbors']} "
            f"--umap_min_dist {kwargs['cluster_umap_min_dist']} "
            f"--umap_metric {kwargs['cluster_umap_metric']} "
        )
    # Add slide_folders support
    if kwargs.get("cluster_slide_folders"):
        slide_folders_str = " ".join(f'"{folder}"' for folder in kwargs["cluster_slide_folders"])
        cmd += f"--slide_folders {slide_folders_str} "
    logging.info("Cluster cmd: %s", cmd)
    return cmd.strip()

# --------------------------------------------------------------------------- #
# Component factory
# --------------------------------------------------------------------------- #
def build_components(
    *,
    env: Environment,
    data_prep_output_uri: str,
    tile_filter_output_uri: str,
    segment_output_uri: str,
    cluster_output_uri: str,
    classify_output_uri: str,
    postprocess_output_uri: str,
    annotation_output_uri: str,
    cluster_tiles_output_uri: str,
    filtered_annotation_output_uri: str,
    prepped_tiles_uri: str,
    classify_per_cluster: int,
    param_string: str,
    magnifications: str,
    num_tiles: int | None,
    filter_tiles: bool,
    filter_min_edge_density: float,
    filter_max_bright_ratio: float,
    filter_max_dark_ratio: float,
    filter_min_std_intensity: float,
    filter_min_laplacian_var: float,
    filter_min_color_variance: float,
    segment_model_type: str,
    segment_flow_threshold: float,
    segment_cellprob_threshold: float,
    segment_use_gpu: bool,
    segment_diameter: float | None,
    segment_resample: bool,
    segment_normalize: bool,
    segment_do_3D: bool,
    segment_stitch_threshold: float,
    segment_channels: str,
    segment_use_cellpose_sam: bool,
    cluster_eps: float | None,
    cluster_min_samples: int,
    cluster_use_gpu: bool,
    cluster_normalize: bool,
    cluster_use_umap: bool,
    cluster_umap_components: int,
    cluster_umap_neighbors: int,
    cluster_umap_min_dist: float,
    cluster_umap_metric: str,
    cluster_per_slide: bool,
    cluster_slide_folders: list | None,
    # Annotation parameters
    enable_annotations: bool,
    annotation_max_labels: int,
    annotation_random_labels: bool,
    annotation_draw_bbox: bool,
    annotation_draw_polygon: bool,
    annotation_no_text: bool,
    annotation_text_use_pred_class: bool,
    annotation_text_use_cluster_id: bool,
    annotation_text_use_cluster_confidence: bool,
    annotation_text_scale: float,
    annotation_color_by: str,
    annotation_filter_unclassified: bool,
    # Cluster analyzer parameters
    enable_cluster_tiles: bool,
    cluster_analyzer_confidence_threshold: float,
    cluster_analyzer_max_items: int,
    # Filtered annotation parameters
    enable_filtered_annotations: bool,
    filtered_annotation_max_labels: int,
    filtered_annotation_random_labels: bool,
    filtered_annotation_draw_bbox: bool,
    filtered_annotation_draw_polygon: bool,
    filtered_annotation_no_text: bool,
    filtered_annotation_text_use_pred_class: bool,
    filtered_annotation_text_use_cluster_id: bool,
    filtered_annotation_text_use_cluster_confidence: bool,
    filtered_annotation_text_scale: float,
    filtered_annotation_color_by: str,
    filtered_annotation_filter_unclassified: bool,
):
    """Create the Azure ML command components used in the pipeline."""
    logging.info("Building component objects …")

    dp_cmd = (
        "python data_prep.py "
        "--input_data ${{inputs.input_data}} "
        "--output_path ${{outputs.output_path}} "
        f"--tile_size 512 "
        f"--magnifications \"{magnifications}\" "
        f"{'--num_tiles ' + str(num_tiles) + ' ' if num_tiles is not None else ''}"
        "--replace_percent_in_names"
    )
    data_prep_component = command(
        name="DataPrep",
        display_name="Data Preparation (tiling)",
        inputs={"input_data": Input(type=AssetTypes.URI_FOLDER)},
        outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=data_prep_output_uri)},
        code="./",
        command=dp_cmd,
        environment=env,
    )

    tile_filter_component = None
    if filter_tiles:
        filter_cmd = (
            "python tile_filter.py "
            "--input_path ${{inputs.input_path}} "
            "--output_path ${{outputs.output_path}} "
            f"--min_edge_density {filter_min_edge_density} "
            f"--max_bright_ratio {filter_max_bright_ratio} "
            f"--max_dark_ratio {filter_max_dark_ratio} "
            f"--min_std_intensity {filter_min_std_intensity} "
            f"--min_laplacian_var {filter_min_laplacian_var} "
            f"--min_color_variance {filter_min_color_variance} "
            "--save_stats"
        )
        tile_filter_component = command(
            name="TileFilter",
            display_name="Tile Quality Filtering",
            inputs={"input_path": Input(type=AssetTypes.URI_FOLDER)},
            outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=tile_filter_output_uri)},
            code="./",
            command=filter_cmd,
            environment=env,
        )

    # Determine model type (cellpose-sam takes precedence)
    model_type = "cellpose_sam" if segment_use_cellpose_sam else segment_model_type
    
    seg_cmd = (
        "python segment.py "
        "--input_path ${{inputs.prepped_tiles_path}} "
        "--output_path ${{outputs.output_path}} "
        f"--model_type {model_type} "
        f"--channels {segment_channels} "
        f"--flow_threshold {segment_flow_threshold} "
        f"--cellprob_threshold {segment_cellprob_threshold} "
        f"{'--segment_use_gpu ' if segment_use_gpu else ''}"
        f"{'--diameter ' + str(segment_diameter) + ' ' if segment_diameter is not None else ''}"
        f"{'--resample ' if segment_resample else ''}"
        f"{'--normalize ' if segment_normalize else '--no_normalize '}"
        f"{'--do_3D ' if segment_do_3D else ''}"
        f"--stitch_threshold {segment_stitch_threshold} "
    )
    segment_component = command(
        name="Segmentation",
        display_name="Cell segmentation",
        inputs={"prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER)},
        outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=segment_output_uri)},
        code="./",
        command=seg_cmd,
        environment=env,
    )

    cluster_component = command(
        name="Clustering",
        display_name="DBSCAN clustering of cells",
        inputs={
            "segmentation_path": Input(type=AssetTypes.URI_FOLDER),
            "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER),
        },
        outputs={"cluster_output": Output(type=AssetTypes.URI_FOLDER, path=cluster_output_uri)},
        code="./",
        command=build_cluster_command(
            cluster_eps=cluster_eps,
            cluster_min_samples=cluster_min_samples,
            cluster_use_gpu=cluster_use_gpu,
            cluster_normalize=cluster_normalize,
            cluster_use_umap=cluster_use_umap,
            cluster_umap_components=cluster_umap_components,
            cluster_umap_neighbors=cluster_umap_neighbors,
            cluster_umap_min_dist=cluster_umap_min_dist,
            cluster_umap_metric=cluster_umap_metric,
            cluster_per_slide=cluster_per_slide,
            cluster_slide_folders=cluster_slide_folders,
        ),
        environment=env,
    )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    classify_env = {"OPENAI_API_KEY": openai_api_key} if openai_api_key else {}
    cls_cmd = (
        "python classify.py "
        "--segmented_path ${{inputs.segmented_path}} "
        "--prepped_tiles_path ${{inputs.prepped_tiles_path}} "
        "--clustered_cells_path ${{inputs.cluster_path}} "
        "--output_path ${{outputs.output_path}} "
        "--num_classes 4 "
        f"--classify_per_cluster {classify_per_cluster}"
    )
    classify_component = command(
        name="Classification",
        display_name="GPT-4o cell-type labelling",
        inputs={
            "segmented_path"   : Input(type=AssetTypes.URI_FOLDER),
            "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER),
            "cluster_path"     : Input(type=AssetTypes.URI_FOLDER),
        },
        outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=classify_output_uri)},
        code="./",
        command=cls_cmd,
        environment=env,
        environment_variables=classify_env,
    )

    post_cmd = (
        "python post_process.py "
        "--segmentation_path ${{inputs.segmentation_path}} "
        "--classification_path ${{inputs.classification_path}} "
        "--output_path ${{outputs.output_path}} "
        f"--param_string {param_string}"
    )
    post_process_component = command(
        name="PostProcess",
        display_name="Post-processing",
        inputs={
            "segmentation_path": Input(type=AssetTypes.URI_FOLDER),
            "classification_path": Input(type=AssetTypes.URI_FOLDER),
        },
        outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=postprocess_output_uri)},
        code="./",
        command=post_cmd,
        environment=env,
    )

    annotation_component = None
    if enable_annotations:
        # Don't use --no_text if we want to show cluster IDs or other text
        use_text = annotation_text_use_pred_class or annotation_text_use_cluster_id or annotation_text_use_cluster_confidence
        annotation_cmd = (
            "python annotate_images.py "
            "--json_dir ${{inputs.annotations_json}} "
            "--images_dir ${{inputs.prepped_tiles_path}} "
            "--output_dir ${{outputs.output_path}} "
            f"--max_labels {annotation_max_labels} "
            f"{'--random_labels ' if annotation_random_labels else ''}"
            f"{'--draw_bbox ' if annotation_draw_bbox else ''}"
            f"{'--draw_polygon ' if annotation_draw_polygon else ''}"
            f"{'--no_text ' if annotation_no_text and not use_text else ''}"
            f"{'--text_use_pred_class ' if annotation_text_use_pred_class else ''}"
            f"{'--text_use_cluster_id ' if annotation_text_use_cluster_id else ''}"
            f"{'--text_use_cluster_confidence ' if annotation_text_use_cluster_confidence else ''}"
            f"--text_scale {annotation_text_scale} "
            f"--color_by {annotation_color_by} "
            f"{'--filter_unclassified ' if annotation_filter_unclassified else ''}"
        )
        annotation_component = command(
            name="Annotation",
            display_name="Image Annotation",
            inputs={
                "annotations_json": Input(type=AssetTypes.URI_FOLDER),
                "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER),
            },
            outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=annotation_output_uri)},
            code="./",
            command=annotation_cmd.strip(),
            environment=env,
        )

    cluster_tiles_component = None
    if enable_cluster_tiles:
        cluster_tiles_cmd = (
            "python cluster_analyzer.py "
            "--json_dir ${{inputs.annotations_json}} "
            "--tiles_dir ${{inputs.prepped_tiles_path}} "
            "--output_dir ${{outputs.output_path}} "
            f"--confidence_threshold {cluster_analyzer_confidence_threshold} "
            f"{'--max_items ' + str(cluster_analyzer_max_items) + ' ' if cluster_analyzer_max_items is not None else ''}"
        )
        cluster_tiles_component = command(
            name="ClusterTileExtraction",
            display_name="Extract Representative Tiles by Cluster",
            inputs={
                "annotations_json": Input(type=AssetTypes.URI_FOLDER),
                "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER),
            },
            outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=cluster_tiles_output_uri)},
            code="./",
            command=cluster_tiles_cmd.strip(),
            environment=env,
        )

    filtered_annotation_component = None
    if enable_filtered_annotations and enable_cluster_tiles:
        # Don't use --no_text if we want to show cluster IDs or other text
        filtered_use_text = filtered_annotation_text_use_pred_class or filtered_annotation_text_use_cluster_id or filtered_annotation_text_use_cluster_confidence
        filtered_annotation_cmd = (
            "python annotate_images.py "
            "--json_dir ${{inputs.cluster_tiles_path}} "
            "--images_dir ${{inputs.prepped_tiles_path}} "
            "--output_dir ${{outputs.output_path}} "
            f"--max_labels {filtered_annotation_max_labels} "
            f"{'--random_labels ' if filtered_annotation_random_labels else ''}"
            f"{'--draw_bbox ' if filtered_annotation_draw_bbox else ''}"
            f"{'--draw_polygon ' if filtered_annotation_draw_polygon else ''}"
            f"{'--no_text ' if filtered_annotation_no_text and not filtered_use_text else ''}"
            f"{'--text_use_pred_class ' if filtered_annotation_text_use_pred_class else ''}"
            f"{'--text_use_cluster_id ' if filtered_annotation_text_use_cluster_id else ''}"
            f"{'--text_use_cluster_confidence ' if filtered_annotation_text_use_cluster_confidence else ''}"
            f"--text_scale {filtered_annotation_text_scale} "
            f"--color_by {filtered_annotation_color_by} "
            f"{'--filter_unclassified ' if filtered_annotation_filter_unclassified else ''}"
        )
        filtered_annotation_component = command(
            name="FilteredAnnotation",
            display_name="Filtered Cluster Tile Annotation",
            inputs={
                "cluster_tiles_path": Input(type=AssetTypes.URI_FOLDER),
                "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER),
            },
            outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=filtered_annotation_output_uri)},
            code="./",
            command=filtered_annotation_cmd.strip(),
            environment=env,
        )

    components = {
        "data_prep"   : data_prep_component,
        "segment"     : segment_component,
        "cluster"     : cluster_component,
        "classify"    : classify_component,
        "post_process": post_process_component,
    }
    
    if tile_filter_component:
        components["tile_filter"] = tile_filter_component
    
    if annotation_component:
        components["annotation"] = annotation_component
    
    if cluster_tiles_component:
        components["cluster_tiles"] = cluster_tiles_component
    
    if filtered_annotation_component:
        components["filtered_annotation"] = filtered_annotation_component
    
    return components

# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def run_pipeline():
    # ---------------- CLI ---------------- #
    p = argparse.ArgumentParser("Launch Azure ML histology pipeline")
    p.add_argument("--mode", choices=[
        "prep_only", "full", "seg_cluster_cls", "cluster_cls", "classify_only", "annotate_only", "extract_cluster_tiles_only", "cluster_tiles_and_filtered_annotations"
    ], default="full")

    # Input URIs
    p.add_argument("--raw_slides_uri", default="azureml://datastores/workspaceblobstore/paths/your_slides/")
    p.add_argument("--prepped_data_uri", default="azureml://datastores/workspaceblobstore/paths/my_prepped_data/")
    p.add_argument("--segmented_data_uri", default="azureml://datastores/workspaceblobstore/paths/my_segmented_data/")
    p.add_argument("--clustered_data_uri", default="azureml://datastores/workspaceblobstore/paths/my_clustered_data/")
    p.add_argument("--postprocess_data_uri", default="azureml://datastores/workspaceblobstore/paths/my_postprocess_data/")

    # Data-prep params
    p.add_argument("--magnifications", type=str, default="1.0",
                   help="Comma-separated list of magnifications, e.g. '1.0,0.9,0.8' (≤1.0 for downsampling) or '1.0,1.2,1.5' (>1.0 for upsampling)")
    p.add_argument("--num_tiles", type=int, default=None,
                   help="Approx. number of tiles per magnification (uniform thinning)")
    p.add_argument("--filter_tiles", action="store_true",
                   help="Enable tile filtering to remove background noise")
    p.add_argument("--filter_min_edge_density", type=float, default=0.02,
                   help="Minimum edge density (structure content) [0.02]")
    p.add_argument("--filter_max_bright_ratio", type=float, default=0.8,
                   help="Maximum ratio of bright pixels (background) [0.8]")
    p.add_argument("--filter_max_dark_ratio", type=float, default=0.8,
                   help="Maximum ratio of dark pixels (empty space) [0.8]")
    p.add_argument("--filter_min_std_intensity", type=float, default=10.0,
                   help="Minimum intensity standard deviation [10.0]")
    p.add_argument("--filter_min_laplacian_var", type=float, default=50.0,
                   help="Minimum Laplacian variance (sharpness) [50.0]")
    p.add_argument("--filter_min_color_variance", type=float, default=5.0,
                   help="Minimum color variance across channels [5.0]")

    # Classification
    p.add_argument("--classify_per_cluster", type=int, default=10)

    # Segmentation
    p.add_argument("--segment_model_type", type=str, default="cellpose_sam",
                   choices=["cyto", "cyto2", "cyto3", "nuclei", "tissuenet", "livecell", "yeast_PhC", "yeast_BF", 
                           "bact_phase", "bact_fluor", "deepbact", "cyto2_cp3", "cyto2_omni", "cellpose_sam"],
                   help="Cellpose model type [cellpose_sam]. Uses latest SAM-based model with superhuman generalization by default")
    p.add_argument("--segment_flow_threshold", type=float, default=0.4)
    p.add_argument("--segment_cellprob_threshold", type=float, default=0.0)
    p.add_argument("--segment_use_gpu", action="store_true", default=True,
                   help="Use GPU for segmentation (default: True for V100)")
    p.add_argument("--segment_diameter", type=float, default=None,
                   help="Expected cell diameter in pixels. If None, cellpose will estimate automatically")
    p.add_argument("--segment_resample", action="store_true",
                   help="Enable resampling for better segmentation of variable-sized objects")
    p.add_argument("--segment_normalize", action="store_true", default=True,
                   help="Normalize images before segmentation (recommended for most cases)")
    p.add_argument("--segment_do_3D", action="store_true",
                   help="Enable 3D segmentation (for Z-stacks)")
    p.add_argument("--segment_stitch_threshold", type=float, default=0.0,
                   help="Threshold for stitching masks across tiles (0.0 = no stitching)")
    p.add_argument("--segment_channels", type=str, default="2,1",
                   help="Comma-separated channel specification: 'cytoplasm_channel,nucleus_channel' (e.g., '2,1' or '0,0' for grayscale)")
    p.add_argument("--segment_use_cellpose_sam", action="store_true",
                   help="Use Cellpose-SAM for enhanced generalization (automatically sets model to cellpose_sam)")

    # Clustering
    p.add_argument("--cluster_eps", type=float, default=None)
    p.add_argument("--cluster_min_samples", type=int, default=5)
    p.add_argument("--cluster_use_gpu", action="store_true", default=True,
                   help="Use GPU for clustering (default: True for V100)")
    p.add_argument("--cluster_normalize", action="store_true")
    p.add_argument("--cluster_use_umap", action="store_true")
    p.add_argument("--cluster_umap_components", type=int, default=50)
    p.add_argument("--cluster_umap_neighbors", type=int, default=15)
    p.add_argument("--cluster_umap_min_dist", type=float, default=0.1)
    p.add_argument("--cluster_umap_metric", default="euclidean")
    p.add_argument("--cluster_per_slide", action="store_true",
                   help="Perform clustering separately for each slide instead of globally")
    p.add_argument("--cluster_slide_folders", type=str, nargs='*',
                   help="Specific slide folder names to process when using --cluster_per_slide. If not provided, processes all folders.")
    
    p.add_argument("--enable_annotations", action="store_true",
                   help="Enable image annotation generation")
    p.add_argument("--annotation_max_labels", type=int, default=100,
                   help="Maximum number of labels to draw per image")
    p.add_argument("--annotation_random_labels", action="store_true",
                   help="Pick labels randomly up to max_labels")
    p.add_argument("--annotation_draw_bbox", action="store_true",
                   help="Draw bounding boxes")
    p.add_argument("--annotation_draw_polygon", action="store_true", default=True,
                   help="Draw polygons")
    p.add_argument("--annotation_no_text", action="store_true", default=True,
                   help="Do not draw text labels")
    p.add_argument("--annotation_text_use_pred_class", action="store_true",
                   help="Include predicted class in text labels")
    p.add_argument("--annotation_text_use_cluster_id", action="store_true",
                   help="Include cluster ID in text labels")
    p.add_argument("--annotation_text_use_cluster_confidence", action="store_true",
                   help="Include cluster confidence in text labels")
    p.add_argument("--annotation_text_scale", type=float, default=0.5,
                   help="Scale factor for text size")
    p.add_argument("--annotation_color_by", default="cluster_id",
                   choices=["pred_class", "cluster_id", "none"],
                   help="Attribute to use for color-coding shapes")
    p.add_argument("--annotation_filter_unclassified", action="store_true", default=True,
                   help="Filter out unclassified cells")

    # Cluster analyzer parameters
    p.add_argument("--enable_cluster_tiles", action="store_true",
                   help="Enable extraction of representative tiles for each cluster")
    p.add_argument("--cluster_analyzer_confidence_threshold", type=float, default=0.75,
                   help="Minimum cluster confidence for tile extraction [0.75]")
    p.add_argument("--cluster_analyzer_max_items", type=int, default=20,
                   help="Maximum number of representative tiles per cluster [20]")
    
    # Filtered annotation parameters
    p.add_argument("--enable_filtered_annotations", action="store_true",
                   help="Enable annotation of filtered cluster tiles")
    p.add_argument("--filtered_annotation_max_labels", type=int, default=100,
                   help="Maximum number of labels to draw per filtered image")
    p.add_argument("--filtered_annotation_random_labels", action="store_true",
                   help="Pick labels randomly up to max_labels for filtered annotations")
    p.add_argument("--filtered_annotation_draw_bbox", action="store_true",
                   help="Draw bounding boxes for filtered annotations")
    p.add_argument("--filtered_annotation_draw_polygon", action="store_true", default=True,
                   help="Draw polygons for filtered annotations")
    p.add_argument("--filtered_annotation_no_text", action="store_true",
                   help="Do not draw text labels for filtered annotations")
    p.add_argument("--filtered_annotation_text_use_pred_class", action="store_true",
                   help="Include predicted class in text labels for filtered annotations")
    p.add_argument("--filtered_annotation_text_use_cluster_id", action="store_true",
                   help="Include cluster ID in text labels for filtered annotations")
    p.add_argument("--filtered_annotation_text_use_cluster_confidence", action="store_true",
                   help="Include cluster confidence in text labels for filtered annotations")
    p.add_argument("--filtered_annotation_text_scale", type=float, default=0.5,
                   help="Scale factor for text size for filtered annotations")
    p.add_argument("--filtered_annotation_color_by", default="cluster_id",
                   choices=["pred_class", "cluster_id", "none"],
                   help="Attribute to use for color-coding shapes for filtered annotations")
    p.add_argument("--filtered_annotation_filter_unclassified", action="store_true", default=True,
                   help="Filter out unclassified cells for filtered annotations")

    args = p.parse_args()
    validate_environment(args)
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info("Args: %s", vars(args))
    # ------------- Paths & names ------------- #
    timestamp     = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    param_string  = build_param_string(args)
    base_uri      = f"azureml://datastores/workspaceblobstore/paths/{timestamp}_v4_{param_string}"
    dp_uri        = f"{base_uri}/data_prep/"
    filter_uri    = f"{base_uri}/tile_filter/" if args.filter_tiles else None
    seg_uri       = f"{base_uri}/segment/"
    clu_uri       = f"{base_uri}/cluster/"
    cls_uri       = f"{base_uri}/classify/"
    post_uri      = f"{base_uri}/postprocess/"
    annotation_uri = f"{base_uri}/annotations/" if args.enable_annotations else None
    cluster_tiles_uri = f"{base_uri}/cluster_tiles/" if args.enable_cluster_tiles else None
    filtered_annotation_uri = f"{base_uri}/filtered_annotations/" if args.enable_filtered_annotations else None

    # ------------- Azure ML client / env ------------- #
    try:
        ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME,
        )
    except Exception as e:
        logging.error("Azure ML connection failed: %s", e)
        sys.exit(1)

    env = Environment(
        name="edu06_env_cellpose4",
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest",
    )
    ml_client.environments.create_or_update(env)
    # ------------- Build components ------------- #
    components = build_components(
        env=env,
        data_prep_output_uri=dp_uri,
        tile_filter_output_uri=filter_uri,
        segment_output_uri=seg_uri,
        cluster_output_uri=clu_uri,
        classify_output_uri=cls_uri,
        postprocess_output_uri=post_uri,
        annotation_output_uri=annotation_uri,
        cluster_tiles_output_uri=cluster_tiles_uri,
        filtered_annotation_output_uri=filtered_annotation_uri,
        prepped_tiles_uri=filter_uri if args.filter_tiles else dp_uri,
        classify_per_cluster=args.classify_per_cluster,
        param_string=param_string,
        # Data prep params
        magnifications=args.magnifications,
        num_tiles=args.num_tiles,
        # Tile filtering params
        filter_tiles=args.filter_tiles,
        filter_min_edge_density=args.filter_min_edge_density,
        filter_max_bright_ratio=args.filter_max_bright_ratio,
        filter_max_dark_ratio=args.filter_max_dark_ratio,
        filter_min_std_intensity=args.filter_min_std_intensity,
        filter_min_laplacian_var=args.filter_min_laplacian_var,
        filter_min_color_variance=args.filter_min_color_variance,
        # segmentation
        segment_model_type=args.segment_model_type,
        segment_flow_threshold=args.segment_flow_threshold,
        segment_cellprob_threshold=args.segment_cellprob_threshold,
        segment_use_gpu=args.segment_use_gpu,
        segment_diameter=args.segment_diameter,
        segment_resample=args.segment_resample,
        segment_normalize=args.segment_normalize,
        segment_do_3D=args.segment_do_3D,
        segment_stitch_threshold=args.segment_stitch_threshold,
        segment_channels=args.segment_channels,
        segment_use_cellpose_sam=args.segment_use_cellpose_sam,
        # clustering
        cluster_eps=args.cluster_eps,
        cluster_min_samples=args.cluster_min_samples,
        cluster_use_gpu=args.cluster_use_gpu,
        cluster_normalize=args.cluster_normalize,
        cluster_use_umap=args.cluster_use_umap,
        cluster_umap_components=args.cluster_umap_components,
        cluster_umap_neighbors=args.cluster_umap_neighbors,
        cluster_umap_min_dist=args.cluster_umap_min_dist,
        cluster_umap_metric=args.cluster_umap_metric,
        cluster_per_slide=args.cluster_per_slide,
        cluster_slide_folders=args.cluster_slide_folders,
        # Annotation parameters
        enable_annotations=args.enable_annotations,
        annotation_max_labels=args.annotation_max_labels,
        annotation_random_labels=args.annotation_random_labels,
        annotation_draw_bbox=args.annotation_draw_bbox,
        annotation_draw_polygon=args.annotation_draw_polygon,
        annotation_no_text=args.annotation_no_text,
        annotation_text_use_pred_class=args.annotation_text_use_pred_class,
        annotation_text_use_cluster_id=args.annotation_text_use_cluster_id,
        annotation_text_use_cluster_confidence=args.annotation_text_use_cluster_confidence,
        annotation_text_scale=args.annotation_text_scale,
        annotation_color_by=args.annotation_color_by,
        annotation_filter_unclassified=args.annotation_filter_unclassified,
        # Cluster analyzer parameters
        enable_cluster_tiles=args.enable_cluster_tiles,
        cluster_analyzer_confidence_threshold=args.cluster_analyzer_confidence_threshold,
        cluster_analyzer_max_items=args.cluster_analyzer_max_items,
        # Filtered annotation parameters
        enable_filtered_annotations=args.enable_filtered_annotations,
        filtered_annotation_max_labels=args.filtered_annotation_max_labels,
        filtered_annotation_random_labels=args.filtered_annotation_random_labels,
        filtered_annotation_draw_bbox=args.filtered_annotation_draw_bbox,
        filtered_annotation_draw_polygon=args.filtered_annotation_draw_polygon,
        filtered_annotation_no_text=args.filtered_annotation_no_text,
        filtered_annotation_text_use_pred_class=args.filtered_annotation_text_use_pred_class,
        filtered_annotation_text_use_cluster_id=args.filtered_annotation_text_use_cluster_id,
        filtered_annotation_text_use_cluster_confidence=args.filtered_annotation_text_use_cluster_confidence,
        filtered_annotation_text_scale=args.filtered_annotation_text_scale,
        filtered_annotation_color_by=args.filtered_annotation_color_by,
        filtered_annotation_filter_unclassified=args.filtered_annotation_filter_unclassified,
    )
    
    @pipeline(compute=COMPUTE_CLUSTER, description="Full pipeline")
    def full_pipeline(raw_slides_input):
        prep = components["data_prep"](input_data=raw_slides_input)
        
        if args.filter_tiles and "tile_filter" in components:
            filtered = components["tile_filter"](input_path=prep.outputs.output_path)
            tiles_for_segmentation = filtered.outputs.output_path
            tiles_for_clustering = filtered.outputs.output_path
        else:
            tiles_for_segmentation = prep.outputs.output_path
            tiles_for_clustering = prep.outputs.output_path
        
        seg = components["segment"](prepped_tiles_path=tiles_for_segmentation)
        clu = components["cluster"](segmentation_path=seg.outputs.output_path,
                                   prepped_tiles_path=tiles_for_clustering)
        cls = components["classify"](segmented_path=seg.outputs.output_path,
                                    prepped_tiles_path=tiles_for_clustering,
                                    cluster_path=clu.outputs.cluster_output)
        post = components["post_process"](segmentation_path=seg.outputs.output_path,
                                         classification_path=cls.outputs.output_path)
        
        outputs = {"final_output": post.outputs.output_path}
        
        if args.enable_annotations and "annotation" in components:
            annotate = components["annotation"](annotations_json=post.outputs.output_path,
                                              prepped_tiles_path=tiles_for_clustering)
            outputs["annotations"] = annotate.outputs.output_path
        
        if args.enable_cluster_tiles and "cluster_tiles" in components:
            cluster_tiles = components["cluster_tiles"](annotations_json=post.outputs.output_path,
                                                       prepped_tiles_path=tiles_for_clustering)
            outputs["cluster_tiles"] = cluster_tiles.outputs.output_path
            
            if args.enable_filtered_annotations and "filtered_annotation" in components:
                filtered_annotate = components["filtered_annotation"](cluster_tiles_path=cluster_tiles.outputs.output_path,
                                                                     prepped_tiles_path=tiles_for_clustering)
                outputs["filtered_annotations"] = filtered_annotate.outputs.output_path
        
        return outputs

    @pipeline(compute=COMPUTE_CLUSTER, description="Data prep only")
    def data_prep_pipeline(slides_in):
        prep = components["data_prep"](input_data=slides_in)
        if args.filter_tiles and "tile_filter" in components:
            filtered = components["tile_filter"](input_path=prep.outputs.output_path)
            return {"prepped": prep.outputs.output_path, "filtered": filtered.outputs.output_path}
        return {"prepped": prep.outputs.output_path}
    @pipeline(compute=COMPUTE_CLUSTER, description="Seg→Clu→Cls→Post")
    def seg_cluster_cls_pipeline(prepped_in):
        seg = components["segment"](prepped_tiles_path=prepped_in)
        clu = components["cluster"](segmentation_path=seg.outputs.output_path,
                                   prepped_tiles_path=prepped_in)
        cls = components["classify"](segmented_path=seg.outputs.output_path,
                                    prepped_tiles_path=prepped_in,
                                    cluster_path=clu.outputs.cluster_output)
        post = components["post_process"](segmentation_path=seg.outputs.output_path,
                                         classification_path=cls.outputs.output_path)
        
        outputs = {"final_output": post.outputs.output_path}
        
        if args.enable_annotations and "annotation" in components:
            annotate = components["annotation"](annotations_json=post.outputs.output_path,
                                              prepped_tiles_path=prepped_in)
            outputs["annotations"] = annotate.outputs.output_path
        
        if args.enable_cluster_tiles and "cluster_tiles" in components:
            cluster_tiles = components["cluster_tiles"](annotations_json=post.outputs.output_path,
                                                       prepped_tiles_path=prepped_in)
            outputs["cluster_tiles"] = cluster_tiles.outputs.output_path
            
            if args.enable_filtered_annotations and "filtered_annotation" in components:
                filtered_annotate = components["filtered_annotation"](cluster_tiles_path=cluster_tiles.outputs.output_path,
                                                                     prepped_tiles_path=prepped_in)
                outputs["filtered_annotations"] = filtered_annotate.outputs.output_path
        
        return outputs
    @pipeline(compute=COMPUTE_CLUSTER, description="Clu→Cls→Post")
    def cluster_cls_pipeline(prepped_in, segmented_in):
        clu = components["cluster"](segmentation_path=segmented_in,
                                   prepped_tiles_path=prepped_in)
        cls = components["classify"](segmented_path=segmented_in,
                                    prepped_tiles_path=prepped_in,
                                    cluster_path=clu.outputs.cluster_output)
        post = components["post_process"](segmentation_path=segmented_in,
                                         classification_path=cls.outputs.output_path)
        
        outputs = {"final_output": post.outputs.output_path}
        
        if args.enable_annotations and "annotation" in components:
            annotate = components["annotation"](annotations_json=post.outputs.output_path,
                                              prepped_tiles_path=prepped_in)
            outputs["annotations"] = annotate.outputs.output_path
        
        if args.enable_cluster_tiles and "cluster_tiles" in components:
            cluster_tiles = components["cluster_tiles"](annotations_json=post.outputs.output_path,
                                                       prepped_tiles_path=prepped_in)
            outputs["cluster_tiles"] = cluster_tiles.outputs.output_path
            
            if args.enable_filtered_annotations and "filtered_annotation" in components:
                filtered_annotate = components["filtered_annotation"](cluster_tiles_path=cluster_tiles.outputs.output_path,
                                                                     prepped_tiles_path=prepped_in)
                outputs["filtered_annotations"] = filtered_annotate.outputs.output_path
        
        return outputs
    @pipeline(compute=COMPUTE_CLUSTER, description="Cls→Post")
    def classify_only_pipeline(prepped_in, segmented_in, cluster_in):
        cls = components["classify"](segmented_path=segmented_in,
                                    prepped_tiles_path=prepped_in,
                                    cluster_path=cluster_in)
        post = components["post_process"](segmentation_path=segmented_in,
                                         classification_path=cls.outputs.output_path)
        
        outputs = {"final_output": post.outputs.output_path}
        
        if args.enable_annotations and "annotation" in components:
            annotate = components["annotation"](annotations_json=post.outputs.output_path,
                                              prepped_tiles_path=prepped_in)
            outputs["annotations"] = annotate.outputs.output_path
        
        if args.enable_cluster_tiles and "cluster_tiles" in components:
            cluster_tiles = components["cluster_tiles"](annotations_json=post.outputs.output_path,
                                                       prepped_tiles_path=prepped_in)
            outputs["cluster_tiles"] = cluster_tiles.outputs.output_path
            
            if args.enable_filtered_annotations and "filtered_annotation" in components:
                filtered_annotate = components["filtered_annotation"](cluster_tiles_path=cluster_tiles.outputs.output_path,
                                                                     prepped_tiles_path=prepped_in)
                outputs["filtered_annotations"] = filtered_annotate.outputs.output_path
        
        return outputs
    @pipeline(compute=COMPUTE_CLUSTER, description="Annotate existing results")
    def annotate_only_pipeline(postprocess_in, prepped_in):
        if "annotation" in components:
            annotate = components["annotation"](annotations_json=postprocess_in,
                                              prepped_tiles_path=prepped_in)
            return {"annotations": annotate.outputs.output_path}
        else:
            return {"message": "Annotations not enabled"}
    @pipeline(compute=COMPUTE_CLUSTER, description="Extract representative tiles from existing results")
    def cluster_tiles_only_pipeline(postprocess_in, prepped_in):
        if "cluster_tiles" in components:
            cluster_tiles = components["cluster_tiles"](annotations_json=postprocess_in,
                                                       prepped_tiles_path=prepped_in)
            return {"cluster_tiles": cluster_tiles.outputs.output_path}
        else:
            return {"message": "Representative tile extraction not enabled"}

    @pipeline(compute=COMPUTE_CLUSTER, description="Extract representative tiles and create filtered annotations")
    def cluster_tiles_and_filtered_annotations_pipeline(postprocess_in, prepped_in):
        outputs = {}
        
        if "cluster_tiles" in components:
            cluster_tiles = components["cluster_tiles"](annotations_json=postprocess_in,
                                                       prepped_tiles_path=prepped_in)
            outputs["cluster_tiles"] = cluster_tiles.outputs.output_path
            
            if args.enable_filtered_annotations and "filtered_annotation" in components:
                filtered_annotate = components["filtered_annotation"](cluster_tiles_path=cluster_tiles.outputs.output_path,
                                                                     prepped_tiles_path=prepped_in)
                outputs["filtered_annotations"] = filtered_annotate.outputs.output_path
            else:
                outputs["message"] = "Filtered annotations not enabled"
        else:
            outputs["message"] = "Representative tile extraction not enabled"
        
        return outputs

    # ------------- Pick + submit ------------- #
    mode = args.mode
    logging.info("Submitting mode: %s", mode)
    exp_name = f"v1_{mode}_{param_string}_{timestamp}"

    job = None
    if mode == "prep_only":
        job = data_prep_pipeline(slides_in=Input(type=AssetTypes.URI_FOLDER, path=args.raw_slides_uri))
    elif mode == "seg_cluster_cls":
        job = seg_cluster_cls_pipeline(prepped_in=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri))
    elif mode == "cluster_cls":
        job = cluster_cls_pipeline(
            prepped_in=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri),
            segmented_in=Input(type=AssetTypes.URI_FOLDER, path=args.segmented_data_uri),
        )
    elif mode == "classify_only":
        job = classify_only_pipeline(
            prepped_in=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri),
            segmented_in=Input(type=AssetTypes.URI_FOLDER, path=args.segmented_data_uri),
            cluster_in=Input(type=AssetTypes.URI_FOLDER, path=args.clustered_data_uri),
        )
    elif mode == "annotate_only":
        if not args.enable_annotations:
            logging.error("annotate_only mode requires --enable_annotations flag")
            return
        job = annotate_only_pipeline(
            postprocess_in=Input(type=AssetTypes.URI_FOLDER, path=args.postprocess_data_uri),
            prepped_in=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri),
        )
    elif mode == "extract_cluster_tiles_only":
        if not args.enable_cluster_tiles:
            logging.error("extract_cluster_tiles_only mode requires --enable_cluster_tiles flag")
            return
        job = cluster_tiles_only_pipeline(
            postprocess_in=Input(type=AssetTypes.URI_FOLDER, path=args.postprocess_data_uri),
            prepped_in=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri),
        )
    elif mode == "cluster_tiles_and_filtered_annotations":
        if not args.enable_cluster_tiles:
            logging.error("cluster_tiles_and_filtered_annotations mode requires --enable_cluster_tiles flag")
            return
        if not args.enable_filtered_annotations:
            logging.error("cluster_tiles_and_filtered_annotations mode requires --enable_filtered_annotations flag")
            return
        job = cluster_tiles_and_filtered_annotations_pipeline(
            postprocess_in=Input(type=AssetTypes.URI_FOLDER, path=args.postprocess_data_uri),
            prepped_in=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri),
        )
    else:  # full
        job = full_pipeline(raw_slides_input=Input(type=AssetTypes.URI_FOLDER, path=args.raw_slides_uri))

    submitted = ml_client.jobs.create_or_update(job, experiment_name=exp_name)
    logging.info("Job submitted: %s", submitted.studio_url)
    print("View in studio:", submitted.studio_url)

# --------------------------------------------------------------------------- #
def validate_environment(args: argparse.Namespace):
    """Validate that all required environment variables are set."""
    missing = []
    
    if not SUBSCRIPTION_ID:
        missing.append("AZURE_SUBSCRIPTION_ID")
    if not RESOURCE_GROUP:
        missing.append("AZURE_RESOURCE_GROUP")
    if not WORKSPACE_NAME:
        missing.append("AZURE_ML_WORKSPACE_NAME")

    mode_requires_openai = args.mode in {"full", "seg_cluster_cls", "cluster_cls", "classify_only"}
    if mode_requires_openai and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please create a .env file based on .env.example and fill in your values."
        )

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    run_pipeline()
