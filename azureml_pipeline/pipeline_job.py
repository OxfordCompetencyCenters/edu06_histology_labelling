from __future__ import annotations
import argparse, logging, os, sys, warnings
from datetime import datetime, timezone
from pathlib import Path

# Suppress Azure ML SDK internal deprecation warning for pathOnCompute
warnings.filterwarnings("ignore", message=".*pathOnCompute is not a known attribute.*")

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.parallel import parallel_run_function, RunFunction


env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

SUBSCRIPTION_ID  = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP   = os.getenv("AZURE_RESOURCE_GROUP")
WORKSPACE_NAME   = os.getenv("AZURE_ML_WORKSPACE_NAME")
COMPUTE_CLUSTER  = os.getenv("AZURE_ML_COMPUTE_CLUSTER", "gpu-cluster")
# Secondary cluster for memory-intensive operations (clustering)
# Falls back to main cluster if not specified
COMPUTE_CLUSTER_CLUSTERING = os.getenv("AZURE_ML_CLUSTERING_CLUSTER", COMPUTE_CLUSTER)

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
    start_from = getattr(args, 'start_from', 'data_prep')
    parts = []
    
    # For cluster tiles and filtered annotations only modes
    if start_from in ["cluster_tiles"]:
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
    elif start_from == "annotate":
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
            f"model_{format_param_for_name(args.segment_pretrained_model)}",
            f"prob_{format_param_for_name(args.segment_cellprob_threshold)}",
            f"flow_{format_param_for_name(args.segment_flow_threshold)}",
            f"eps_{format_param_for_name(args.cluster_eps)}",
            f"mins_{args.cluster_min_samples}",
        ]
        if args.segment_use_gpu:
            parts.append("segGPU")
        if args.segment_diameter is not None:
            parts.append(f"diam_{format_param_for_name(args.segment_diameter)}")
        if args.segment_resample:
            parts.append("resample")
        if not args.segment_normalize:
            parts.append("nonorm")
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

# --------------------------------------------------------------------------- #
# Component factory
# --------------------------------------------------------------------------- #
def build_components(
    *,
    env: Environment,
    data_prep_output_uri: str,
    data_prep_manifest_uri: str,
    tile_filter_output_uri: str,
    tile_filter_manifest_uri: str,
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
    segment_pretrained_model: str,
    segment_flow_threshold: float,
    segment_cellprob_threshold: float,
    segment_use_gpu: bool,
    segment_diameter: float | None,
    segment_resample: bool,
    segment_normalize: bool,
    segmentation_tile_batch_size: int,
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
    # Parallelization parameters
    max_nodes: int = 1,
    processes_per_node: int = 1,
    mini_batch_error_threshold: int = 5,
    mini_batch_size: str = "1",
    max_retries: int = 3,
    retry_timeout: int = 300,
    use_separate_clustering_cluster: bool = False,
    clustering_use_gpu: bool = True,
    slide_filter: str | None = None,
):
    """Create the Azure ML parallel/command components used in the pipeline.
    
    All stages use slide-level parallelism with parallel_run_function.
    Single-node stages (post_process, annotation, cluster_tiles) use standard command() jobs.
    """
    logging.info("Building component objects …")
    logging.info(f"Parallelization settings: max_nodes={max_nodes}, processes_per_node={processes_per_node}")
    if slide_filter:
        logging.info(f"  Slide filter: {slide_filter}")
    
    if use_separate_clustering_cluster:
        logging.info(f"  Clustering cluster: {COMPUTE_CLUSTER_CLUSTERING}")
        logging.info(f"  Clustering GPU: {clustering_use_gpu}")

    # Build reusable slide_filter argument snippet for program_arguments
    _sf_arg = f'--slide_filter "{slide_filter}" ' if slide_filter else ''

    # ==================== DATA PREP ====================
    logging.info("Building data_prep component")
    data_prep_component = parallel_run_function(
        name="DataPrep",
        display_name="Data Preparation - Tiling",
        inputs=dict(
            input_data=Input(type=AssetTypes.URI_FOLDER, description="Input folder with WSI files"),
        ),
        outputs=dict(
            output_path=Output(type=AssetTypes.URI_FOLDER, path=data_prep_output_uri),
            manifest_path=Output(type=AssetTypes.URI_FOLDER, path=data_prep_manifest_uri),
        ),
        input_data="${{inputs.input_data}}",
        instance_count=max_nodes,
        max_concurrency_per_instance=processes_per_node,
        mini_batch_size=mini_batch_size,
        mini_batch_error_threshold=mini_batch_error_threshold,
        retry_settings=dict(max_retries=max_retries, timeout=retry_timeout),
        logging_level="INFO",
        task=RunFunction(
            code="./",
            entry_script="parallel_data_prep.py",
            environment=env,
            program_arguments=(
                f"--output_path ${{{{outputs.output_path}}}} "
                f"--manifest_path ${{{{outputs.manifest_path}}}} "
                f"--tile_size 512 "
                f"--magnifications \"{magnifications}\" "
                f"{'--num_tiles ' + str(num_tiles) + ' ' if num_tiles is not None else ''}"
                f"{_sf_arg}"
                "--replace_percent_in_names"
            ),
        ),
    )

    # ==================== TILE FILTER ====================
    tile_filter_component = None
    if filter_tiles:
        logging.info("Building tile_filter component")
        tile_filter_component = parallel_run_function(
                name="TileFilter",
                display_name="Tile Quality Filtering (Parallel)",
                inputs=dict(
                    trigger_path=Input(type=AssetTypes.URI_FOLDER, description="Manifest triggers", mode="ro_mount"),
                    input_path=Input(type=AssetTypes.URI_FOLDER, description="Input tiles", mode="ro_mount"),
                ),
                outputs=dict(
                    output_path=Output(type=AssetTypes.URI_FOLDER, path=tile_filter_output_uri),
                    output_manifest=Output(type=AssetTypes.URI_FOLDER, path=tile_filter_manifest_uri),
                ),
                input_data="${{inputs.trigger_path}}",
                instance_count=max_nodes,
                max_concurrency_per_instance=processes_per_node,
                mini_batch_size=mini_batch_size,
                mini_batch_error_threshold=mini_batch_error_threshold,
                retry_settings=dict(max_retries=max_retries, timeout=retry_timeout),
                logging_level="INFO",
                task=RunFunction(
                    code="./",
                    entry_script="parallel_tile_filter.py",
                    environment=env,
                    program_arguments=(
                        f"--output_path ${{{{outputs.output_path}}}} "
                        f"--input_path ${{{{inputs.input_path}}}} "
                        f"--output_manifest_path ${{{{outputs.output_manifest}}}} "
                        f"--min_edge_density {filter_min_edge_density} "
                        f"--max_bright_ratio {filter_max_bright_ratio} "
                        f"--max_dark_ratio {filter_max_dark_ratio} "
                        f"--min_std_intensity {filter_min_std_intensity} "
                        f"--min_laplacian_var {filter_min_laplacian_var} "
                        f"--min_color_variance {filter_min_color_variance} "
                        f"{_sf_arg}"
                        "--save_stats"
                    ),
                ),
            )

    # ==================== SEGMENTATION ====================
    logging.info("Building segment component")
    segment_component = parallel_run_function(
            name="Segmentation",
            display_name="Cell Segmentation (Parallel)",
            inputs=dict(
                trigger_path=Input(type=AssetTypes.URI_FOLDER, description="Manifest triggers", mode="ro_mount"),
                prepped_tiles_path=Input(type=AssetTypes.URI_FOLDER, description="Input tiles", mode="ro_mount"),
            ),
            outputs=dict(
                output_path=Output(type=AssetTypes.URI_FOLDER, path=segment_output_uri),
                output_manifest=Output(type=AssetTypes.URI_FOLDER, description="Manifest for downstream steps"),
            ),
            input_data="${{inputs.trigger_path}}",
            instance_count=max_nodes,
            max_concurrency_per_instance=processes_per_node,
            mini_batch_size=mini_batch_size,
            mini_batch_error_threshold=mini_batch_error_threshold,
            retry_settings=dict(max_retries=max_retries, timeout=retry_timeout),
            logging_level="INFO",
            task=RunFunction(
                code="./",
                entry_script="parallel_segment.py",
                environment=env,
                program_arguments=(
                    f"--output_path ${{{{outputs.output_path}}}} "
                    f"--output_manifest ${{{{outputs.output_manifest}}}} "
                    f"--input_path ${{{{inputs.prepped_tiles_path}}}} "
                    f"--pretrained_model {segment_pretrained_model} "
                    f"--flow_threshold {segment_flow_threshold} "
                    f"--cellprob_threshold {segment_cellprob_threshold} "
                    f"{'--segment_use_gpu ' if segment_use_gpu else ''}"
                    f"{'--diameter ' + str(segment_diameter) + ' ' if segment_diameter is not None else ''}"
                    f"{'--resample ' if segment_resample else ''}"
                    f"{'--normalize ' if segment_normalize else '--no_normalize '}"
                    f"{_sf_arg}"
                    f"--tile_batch_size {segmentation_tile_batch_size}"
                ),
            ),
        )

    # ==================== CLUSTERING ====================
    logging.info("Building cluster component")
    cluster_component = parallel_run_function(
            name="Clustering",
            display_name="DBSCAN Clustering (Per-Slide)",
            inputs=dict(
                trigger_path=Input(type=AssetTypes.URI_FOLDER, description="Manifest triggers", mode="ro_mount"),
                segmentation_path=Input(type=AssetTypes.URI_FOLDER, description="Segmentation results folder", mode="ro_mount"),
                prepped_tiles_path=Input(type=AssetTypes.URI_FOLDER, description="Prepped tiles folder", mode="ro_mount"),
            ),
            outputs=dict(
                cluster_output=Output(type=AssetTypes.URI_FOLDER, path=cluster_output_uri),
                output_manifest=Output(type=AssetTypes.URI_FOLDER, description="Manifest for downstream steps"),
            ),
            input_data="${{inputs.trigger_path}}",
            instance_count=max_nodes,
            max_concurrency_per_instance=processes_per_node,
            mini_batch_size=mini_batch_size,  # 1 slide per mini-batch
            mini_batch_error_threshold=mini_batch_error_threshold,
            retry_settings=dict(max_retries=max_retries, timeout=retry_timeout),
            logging_level="INFO",
            task=RunFunction(
                code="./",
                entry_script="parallel_cluster.py",
                environment=env,
                program_arguments=(
                    f"--segmentation_path ${{{{inputs.segmentation_path}}}} "
                    f"--prepped_tiles_path ${{{{inputs.prepped_tiles_path}}}} "
                    f"--output_path ${{{{outputs.cluster_output}}}} "
                    f"--output_manifest ${{{{outputs.output_manifest}}}} "
                    f"{'--gpu ' if cluster_use_gpu else ''}"
                    f"{'--normalize_embeddings ' if cluster_normalize else ''}"
                    f"--min_samples {cluster_min_samples} "
                    f"{'--eps ' + str(cluster_eps) + ' ' if cluster_eps is not None else ''}"
                    f"{_sf_arg}"
                    + (f"--use_umap "
                       f"--umap_n_components {cluster_umap_components} "
                       f"--umap_n_neighbors {cluster_umap_neighbors} "
                       f"--umap_min_dist {cluster_umap_min_dist} "
                       f"--umap_metric {cluster_umap_metric} " if cluster_use_umap else "")
                ),
            ),
        )

    # ==================== CLASSIFICATION ====================
    openai_api_key = os.getenv("OPENAI_API_KEY")
    classify_env = {"OPENAI_API_KEY": openai_api_key} if openai_api_key else {}
    
    logging.info("Building classify component")
    classify_component = parallel_run_function(
        name="Classification",
        display_name="LLM Classification (Per-Slide)",
        inputs=dict(
            trigger_path=Input(type=AssetTypes.URI_FOLDER, description="Manifest triggers", mode="ro_mount"),
            segmented_path=Input(type=AssetTypes.URI_FOLDER, description="Segmentation results folder", mode="ro_mount"),
            prepped_tiles_path=Input(type=AssetTypes.URI_FOLDER, description="Prepped tiles folder", mode="ro_mount"),
            cluster_path=Input(type=AssetTypes.URI_FOLDER, description="Clustering results folder", mode="ro_mount"),
        ),
        outputs=dict(
            output_path=Output(type=AssetTypes.URI_FOLDER, path=classify_output_uri),
            job_output_file=Output(type=AssetTypes.URI_FILE),
        ),
        input_data="${{inputs.trigger_path}}",
        instance_count=max_nodes,
        max_concurrency_per_instance=processes_per_node,
        mini_batch_size=mini_batch_size,  # 1 slide per mini-batch (API rate limits)
        mini_batch_error_threshold=mini_batch_error_threshold,
        retry_settings=dict(max_retries=max_retries, timeout=retry_timeout),
        logging_level="INFO",
        task=RunFunction(
            code="./",
            entry_script="parallel_classify.py",
            environment=env,
            append_row_to="${{outputs.job_output_file}}",
            program_arguments=(
                f"--segmented_path ${{{{inputs.segmented_path}}}} "
                f"--prepped_tiles_path ${{{{inputs.prepped_tiles_path}}}} "
                f"--clustered_cells_path ${{{{inputs.cluster_path}}}} "
                f"--output_path ${{{{outputs.output_path}}}} "
                f"--num_classes 4 "
                f"{_sf_arg}"
                f"--classify_per_cluster {classify_per_cluster}"
            ),
        ),
        environment_variables=classify_env,
    )

    post_cmd = (
        "python post_process.py "
        "--segmentation_path ${{inputs.segmentation_path}} "
        "--classification_path ${{inputs.classification_path}} "
        "--output_path ${{outputs.output_path}} "
        f"--param_string {param_string} "
        f"{_sf_arg}"
    ).rstrip()
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
            f"{_sf_arg}"
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
            f"{_sf_arg}"
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
            f"{_sf_arg}"
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
# Sequential (single-node) component factory – uses command() instead of
# parallel_run_function() so logs are simple and easy to debug.
# --------------------------------------------------------------------------- #
def build_sequential_components(
    *,
    env: Environment,
    data_prep_output_uri: str,
    data_prep_manifest_uri: str,
    tile_filter_output_uri: str,
    tile_filter_manifest_uri: str,
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
    segment_pretrained_model: str,
    segment_flow_threshold: float,
    segment_cellprob_threshold: float,
    segment_use_gpu: bool,
    segment_diameter: float | None,
    segment_resample: bool,
    segment_normalize: bool,
    segmentation_tile_batch_size: int,
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
    enable_cluster_tiles: bool,
    cluster_analyzer_confidence_threshold: float,
    cluster_analyzer_max_items: int,
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
    # Parallelization parameters (unused but kept for signature compat)
    max_nodes: int = 1,
    processes_per_node: int = 1,
    mini_batch_error_threshold: int = 5,
    mini_batch_size: str = "1",
    max_retries: int = 3,
    retry_timeout: int = 300,
    use_separate_clustering_cluster: bool = False,
    clustering_use_gpu: bool = True,
    slide_filter: str | None = None,
):
    """Build command() components for sequential (single-node) execution.
    """
    logging.info("Building SEQUENTIAL (command) components for single-node debugging …")
    if slide_filter:
        logging.info(f"  Slide filter: {slide_filter}")

    # Build reusable slide_filter argument snippet for command strings
    _sf_arg = f'--slide_filter "{slide_filter}" ' if slide_filter else ''

    # ==================== DATA PREP ====================
    dp_cmd = (
        "python parallel_data_prep.py "
        "--input_data_path ${{inputs.input_data}} "
        "--output_path ${{outputs.output_path}} "
        "--manifest_path ${{outputs.manifest_path}} "
        "--tile_size 512 "
        f"--magnifications \"{magnifications}\" "
        f"{'--num_tiles ' + str(num_tiles) + ' ' if num_tiles is not None else ''}"
        f"{_sf_arg}"
        "--replace_percent_in_names"
    )
    data_prep_component = command(
        name="DataPrep_Seq",
        display_name="Data Preparation - Tiling (Sequential)",
        inputs={"input_data": Input(type=AssetTypes.URI_FOLDER)},
        outputs={
            "output_path": Output(type=AssetTypes.URI_FOLDER, path=data_prep_output_uri, mode="rw_mount"),
            "manifest_path": Output(type=AssetTypes.URI_FOLDER, path=data_prep_manifest_uri, mode="rw_mount"),
        },
        code="./",
        command=dp_cmd.strip(),
        environment=env,
    )

    # ==================== TILE FILTER ====================
    tile_filter_component = None
    if filter_tiles:
        tf_cmd = (
            "python parallel_tile_filter.py "
            "--output_path ${{outputs.output_path}} "
            "--input_path ${{inputs.input_path}} "
            "--output_manifest_path ${{outputs.output_manifest}} "
            f"--min_edge_density {filter_min_edge_density} "
            f"--max_bright_ratio {filter_max_bright_ratio} "
            f"--max_dark_ratio {filter_max_dark_ratio} "
            f"--min_std_intensity {filter_min_std_intensity} "
            f"--min_laplacian_var {filter_min_laplacian_var} "
            f"--min_color_variance {filter_min_color_variance} "
            f"{_sf_arg}"
            "--save_stats"
        )
        tile_filter_component = command(
            name="TileFilter_Seq",
            display_name="Tile Quality Filtering (Sequential)",
            inputs={
                "input_path": Input(type=AssetTypes.URI_FOLDER),
            },
            outputs={
                "output_path": Output(type=AssetTypes.URI_FOLDER, path=tile_filter_output_uri, mode="rw_mount"),
                "output_manifest": Output(type=AssetTypes.URI_FOLDER, path=tile_filter_manifest_uri, mode="rw_mount"),
            },
            code="./",
            command=tf_cmd.strip(),
            environment=env,
        )

    # ==================== SEGMENTATION ====================
    seg_cmd = (
        "python parallel_segment.py "
        "--output_path ${{outputs.output_path}} "
        "--input_path ${{inputs.prepped_tiles_path}} "
        "--output_manifest ${{outputs.output_manifest}} "
        f"--pretrained_model {segment_pretrained_model} "
        f"--flow_threshold {segment_flow_threshold} "
        f"--cellprob_threshold {segment_cellprob_threshold} "
        f"{'--segment_use_gpu ' if segment_use_gpu else ''}"
        f"{'--diameter ' + str(segment_diameter) + ' ' if segment_diameter is not None else ''}"
        f"{'--resample ' if segment_resample else ''}"
        f"{'--normalize ' if segment_normalize else '--no_normalize '}"
        f"{_sf_arg}"
        f"--tile_batch_size {segmentation_tile_batch_size}"
    )
    segment_component = command(
        name="Segmentation_Seq",
        display_name="Cell Segmentation (Sequential)",
        inputs={
            "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER),
        },
        outputs={
            "output_path": Output(type=AssetTypes.URI_FOLDER, path=segment_output_uri, mode="rw_mount"),
            "output_manifest": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
        },
        code="./",
        command=seg_cmd.strip(),
        environment=env,
    )

    # ==================== CLUSTERING ====================
    clu_cmd = (
        "python parallel_cluster.py "
        "--segmentation_path ${{inputs.segmentation_path}} "
        "--prepped_tiles_path ${{inputs.prepped_tiles_path}} "
        "--output_path ${{outputs.cluster_output}} "
        "--output_manifest ${{outputs.output_manifest}} "
        f"{'--gpu ' if cluster_use_gpu else ''}"
        f"{'--normalize_embeddings ' if cluster_normalize else ''}"
        f"--min_samples {cluster_min_samples} "
        f"{'--eps ' + str(cluster_eps) + ' ' if cluster_eps is not None else ''}"
        f"{_sf_arg}"
        + (f"--use_umap "
           f"--umap_n_components {cluster_umap_components} "
           f"--umap_n_neighbors {cluster_umap_neighbors} "
           f"--umap_min_dist {cluster_umap_min_dist} "
           f"--umap_metric {cluster_umap_metric} " if cluster_use_umap else "")
    )
    cluster_component = command(
        name="Clustering_Seq",
        display_name="DBSCAN Clustering (Sequential)",
        inputs={
            "segmentation_path": Input(type=AssetTypes.URI_FOLDER),
            "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER),
        },
        outputs={
            "cluster_output": Output(type=AssetTypes.URI_FOLDER, path=cluster_output_uri, mode="rw_mount"),
            "output_manifest": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
        },
        code="./",
        command=clu_cmd.strip(),
        environment=env,
    )

    # ==================== CLASSIFICATION ====================
    openai_api_key = os.getenv("OPENAI_API_KEY")
    classify_env = {"OPENAI_API_KEY": openai_api_key} if openai_api_key else {}

    cls_cmd = (
        "python parallel_classify.py "
        "--segmented_path ${{inputs.segmented_path}} "
        "--prepped_tiles_path ${{inputs.prepped_tiles_path}} "
        "--clustered_cells_path ${{inputs.cluster_path}} "
        "--output_path ${{outputs.output_path}} "
        f"--num_classes 4 "
        f"{_sf_arg}"
        f"--classify_per_cluster {classify_per_cluster}"
    )
    classify_component = command(
        name="Classification_Seq",
        display_name="LLM Classification (Sequential)",
        inputs={
            "segmented_path": Input(type=AssetTypes.URI_FOLDER),
            "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER),
            "cluster_path": Input(type=AssetTypes.URI_FOLDER),
        },
        outputs={
            "output_path": Output(type=AssetTypes.URI_FOLDER, path=classify_output_uri, mode="rw_mount"),
        },
        code="./",
        command=cls_cmd.strip(),
        environment=env,
        environment_variables=classify_env,
    )

    # ==================== POST-PROCESS ====================
    post_cmd = (
        "python post_process.py "
        "--segmentation_path ${{inputs.segmentation_path}} "
        "--classification_path ${{inputs.classification_path}} "
        "--output_path ${{outputs.output_path}} "
        f"--param_string {param_string} "
        f"{_sf_arg}"
    ).rstrip()
    post_process_component = command(
        name="PostProcess_Seq",
        display_name="Post-processing (Sequential)",
        inputs={
            "segmentation_path": Input(type=AssetTypes.URI_FOLDER),
            "classification_path": Input(type=AssetTypes.URI_FOLDER),
        },
        outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=postprocess_output_uri, mode="rw_mount")},
        code="./",
        command=post_cmd,
        environment=env,
    )

    # ==================== ANNOTATION ====================
    annotation_component = None
    if enable_annotations:
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
            f"{_sf_arg}"
        )
        annotation_component = command(
            name="Annotation_Seq",
            display_name="Image Annotation (Sequential)",
            inputs={
                "annotations_json": Input(type=AssetTypes.URI_FOLDER),
                "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER),
            },
            outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=annotation_output_uri, mode="rw_mount")},
            code="./",
            command=annotation_cmd.strip(),
            environment=env,
        )

    # ==================== CLUSTER TILES ====================
    cluster_tiles_component = None
    if enable_cluster_tiles:
        cluster_tiles_cmd = (
            "python cluster_analyzer.py "
            "--json_dir ${{inputs.annotations_json}} "
            "--tiles_dir ${{inputs.prepped_tiles_path}} "
            "--output_dir ${{outputs.output_path}} "
            f"--confidence_threshold {cluster_analyzer_confidence_threshold} "
            f"{'--max_items ' + str(cluster_analyzer_max_items) + ' ' if cluster_analyzer_max_items is not None else ''}"
            f"{_sf_arg}"
        )
        cluster_tiles_component = command(
            name="ClusterTileExtraction_Seq",
            display_name="Extract Representative Tiles (Sequential)",
            inputs={
                "annotations_json": Input(type=AssetTypes.URI_FOLDER),
                "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER),
            },
            outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=cluster_tiles_output_uri, mode="rw_mount")},
            code="./",
            command=cluster_tiles_cmd.strip(),
            environment=env,
        )

    # ==================== FILTERED ANNOTATION ====================
    filtered_annotation_component = None
    if enable_filtered_annotations and enable_cluster_tiles:
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
            f"{_sf_arg}"
        )
        filtered_annotation_component = command(
            name="FilteredAnnotation_Seq",
            display_name="Filtered Cluster Tile Annotation (Sequential)",
            inputs={
                "cluster_tiles_path": Input(type=AssetTypes.URI_FOLDER),
                "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER),
            },
            outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=filtered_annotation_output_uri, mode="rw_mount")},
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

    # ============ STEP SELECTION ============
    # Pick a step and the pipeline runs from that step onward.
    # Steps in order: data_prep → tile_filter → segment → cluster →
    #                 classify → post_process → annotate / cluster_tiles
    ORDERED_STEPS = [
        "data_prep", "tile_filter", "segment", "cluster",
        "classify", "post_process", "annotate", "cluster_tiles",
    ]
    p.add_argument(
        "--start_from",
        choices=ORDERED_STEPS,
        default="data_prep",
        help=(
            "Run the pipeline starting from this step onward. "
            "Steps in order: data_prep → tile_filter → segment → cluster → "
            "classify → post_process → annotate → cluster_tiles. "
            "Provide the required --*_uri args for whichever step you start from."
        ),
    )

    # Input / output URIs — supply what the chosen start step needs.
    # data_prep needs:  --raw_slides_uri
    # tile_filter needs: --prepped_data_uri
    # segment needs:     --prepped_data_uri  (+ --prepped_manifest_uri for parallel)
    # cluster needs:     --prepped_data_uri  --segmented_data_uri  (+ --segmented_manifest_uri for parallel)
    # classify needs:    --prepped_data_uri  --segmented_data_uri  --clustered_data_uri (+ --clustered_manifest_uri for parallel)
    # post_process needs:--prepped_data_uri  --segmented_data_uri  --classified_data_uri
    # annotate needs:    --prepped_data_uri  --postprocess_data_uri
    # cluster_tiles:     --prepped_data_uri  --postprocess_data_uri
    p.add_argument("--raw_slides_uri", default="azureml://datastores/workspaceblobstore/paths/your_slides/")
    p.add_argument("--prepped_data_uri", default=None,
                   help="URI to prepped tiles folder (required for start_from >= tile_filter)")
    p.add_argument("--prepped_manifest_uri", default=None,
                   help="URI to manifest/trigger folder for prepped tiles (parallel mode, start_from=segment)")
    p.add_argument("--segmented_data_uri", default=None,
                   help="URI to segmentation results folder (required for start_from >= cluster)")
    p.add_argument("--segmented_manifest_uri", default=None,
                   help="URI to segmentation manifest/trigger folder (parallel mode, start_from=cluster)")
    p.add_argument("--clustered_data_uri", default=None,
                   help="URI to clustering results folder (required for start_from >= classify)")
    p.add_argument("--clustered_manifest_uri", default=None,
                   help="URI to clustering manifest/trigger folder (parallel mode, start_from=classify)")
    p.add_argument("--classified_data_uri", default=None,
                   help="URI to classification results folder (required for start_from=post_process)")
    p.add_argument("--postprocess_data_uri", default=None,
                   help="URI to post-process results folder (required for start_from >= annotate)")

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
    p.add_argument("--segment_pretrained_model", type=str, default="cpsam",
                   choices=["cpsam", "cyto", "cyto2", "cyto3", "nuclei", "tissuenet", "livecell", "yeast_PhC", "yeast_BF", 
                           "bact_phase", "bact_fluor", "deepbact", "cyto2_cp3", "cyto2_omni"])
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
    p.add_argument("--segmentation_tile_batch_size", type=int, default=1,
                   help="Number of tiles to segment in a single batch (higher = faster on GPU, more VRAM)")

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

    # ============ RESUME / CHECKPOINT CONTROL ============
    p.add_argument("--base_uri", type=str, default=None,
                   help="Resume a preempted run by reusing the same blob storage base URI. "
                        "Copy the 'base_uri' value printed by the previous run and pass it here. "
                        "When omitted a new timestamped path is generated automatically.")

    # ============ PARALLELIZATION CONTROL ============
    # Global parallelization settings
    p.add_argument("--max_nodes", type=int, default=1,
                   help="Maximum number of nodes for parallel stages. "
                        "When set to 1 (default), uses sequential command()-based jobs "
                        "with simple linear logs for easy debugging. "
                        "Set >1 to use parallel_run_function for multi-node execution.")
    p.add_argument("--processes_per_node", type=int, default=1,
                   help="Number of processes per node (set >1 for multi-GPU nodes)")
    p.add_argument("--mini_batch_size", type=str, default="1",
                   help="Number of files (slides) per mini-batch for parallel stages (default: 1)")
    p.add_argument("--mini_batch_error_threshold", type=int, default=5,
                   help="Number of failed mini-batches allowed before failing the job")
    p.add_argument("--max_retries", type=int, default=3,
                   help="Max retries per mini-batch on failure/timeout (useful for low-priority VMs)")
    p.add_argument("--retry_timeout", type=int, default=300,
                   help="Timeout in seconds for each mini-batch retry (default: 300)")
    
    # Clustering-specific settings
    p.add_argument("--use_separate_clustering_cluster", action="store_true",
                   help="Use a separate compute cluster for clustering (set AZURE_ML_CLUSTERING_CLUSTER in .env)")
    p.add_argument("--clustering_use_gpu", action="store_true",
                   help="Use GPU for clustering on the clustering cluster (default: auto-detect from cluster name)")

    # Slide subset selection
    p.add_argument("--slide_filter", type=str, default=None,
                   help="Comma-separated list of slide names to process (others are skipped). "
                        "Slide names may contain spaces but not commas. "
                        "Matching works against both original filenames and generated slide IDs. "
                        "Example: --slide_filter \"Slide A,Slide B,Slide C\"")

    args = p.parse_args()
    validate_environment(args)
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    
    # Suppress verbose Azure SDK logging
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("azure.identity").setLevel(logging.WARNING)

    # Suppress "pathOnCompute is not a known attribute" warning from azure.ai.ml REST models.
    # This is emitted via logging (not warnings.warn), so we filter the specific logger.
    class _PathOnComputeFilter(logging.Filter):
        def filter(self, record):
            return "pathOnCompute is not a known attribute" not in record.getMessage()
    logging.getLogger("azure.ai.ml._restclient").addFilter(_PathOnComputeFilter())
    logging.getLogger("azure.ai.ml").addFilter(_PathOnComputeFilter())
    
    logging.info("Args: %s", vars(args))
    # ------------- Paths & names ------------- #
    timestamp     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    param_string  = build_param_string(args)
    if args.base_uri:
        base_uri = args.base_uri.rstrip("/")
        logging.info("RESUMING with existing base_uri: %s", base_uri)
    else:
        base_uri = f"azureml://datastores/workspaceblobstore/paths/{timestamp}_v4_{param_string}"
    # Print prominently so the user can copy-paste for --base_uri on resume
    logging.info("========================================")
    logging.info("base_uri = %s", base_uri)
    logging.info("To resume after preemption, re-run with:")
    logging.info("  --base_uri \"%s\"", base_uri)
    logging.info("========================================")
    dp_uri        = f"{base_uri}/data_prep/"
    dp_manifest   = f"{base_uri}/manifest_dp/"
    filter_uri    = f"{base_uri}/tile_filter/" if args.filter_tiles else None
    filter_manifest = f"{base_uri}/manifest_tf/" if args.filter_tiles else None
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
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest",
    )
    ml_client.environments.create_or_update(env)
    # ------------- Build components ------------- #
    use_sequential = (args.max_nodes <= 1)
    if use_sequential:
        logging.info("max_nodes=1 → using SEQUENTIAL (command-based) pipeline for easier debugging")
    
    component_builder = build_sequential_components if use_sequential else build_components
    components = component_builder(
        env=env,
        data_prep_output_uri=dp_uri,
        data_prep_manifest_uri=dp_manifest,
        tile_filter_output_uri=filter_uri,
        tile_filter_manifest_uri=filter_manifest,
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
        segment_pretrained_model=args.segment_pretrained_model,
        segment_flow_threshold=args.segment_flow_threshold,
        segment_cellprob_threshold=args.segment_cellprob_threshold,
        segment_use_gpu=args.segment_use_gpu,
        segment_diameter=args.segment_diameter,
        segment_resample=args.segment_resample,
        segment_normalize=args.segment_normalize,
        segmentation_tile_batch_size=args.segmentation_tile_batch_size,
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
        # Parallelization parameters
        max_nodes=args.max_nodes,
        processes_per_node=args.processes_per_node,
        mini_batch_size=args.mini_batch_size,
        mini_batch_error_threshold=args.mini_batch_error_threshold,
        max_retries=args.max_retries,
        retry_timeout=args.retry_timeout,
        use_separate_clustering_cluster=args.use_separate_clustering_cluster,
        clustering_use_gpu=args.clustering_use_gpu if args.clustering_use_gpu else args.cluster_use_gpu,
        slide_filter=args.slide_filter,
    )
    
    # ------------- Determine clustering compute target ------------- #
    clustering_cluster_target = COMPUTE_CLUSTER
    effective_cluster_use_gpu = args.cluster_use_gpu
    
    if args.use_separate_clustering_cluster:
        clustering_cluster_target = COMPUTE_CLUSTER_CLUSTERING
        if not args.clustering_use_gpu and 'cpu' in COMPUTE_CLUSTER_CLUSTERING.lower():
            effective_cluster_use_gpu = False
            logging.info(f"Auto-detected CPU-only clustering cluster: {COMPUTE_CLUSTER_CLUSTERING}")
        elif args.clustering_use_gpu:
            effective_cluster_use_gpu = True
        logging.info(f"Using separate clustering cluster: {clustering_cluster_target}")
        logging.info(f"Clustering GPU enabled: {effective_cluster_use_gpu}")

    # ================================================================ #
    # Helper: append optional annotation/cluster-tiles tail steps
    # to any pipeline that produces a post_process output.
    # ================================================================ #
    def _append_tail_steps(outputs, post_output, prepped_tiles):
        """Wire up annotation / cluster_tiles / filtered_annotation from post_process output."""
        if args.enable_annotations and "annotation" in components:
            annotate = components["annotation"](
                annotations_json=post_output,
                prepped_tiles_path=prepped_tiles,
            )
            outputs["annotations"] = annotate.outputs.output_path

        if args.enable_cluster_tiles and "cluster_tiles" in components:
            ct = components["cluster_tiles"](
                annotations_json=post_output,
                prepped_tiles_path=prepped_tiles,
            )
            outputs["cluster_tiles"] = ct.outputs.output_path

            if args.enable_filtered_annotations and "filtered_annotation" in components:
                fa = components["filtered_annotation"](
                    cluster_tiles_path=ct.outputs.output_path,
                    prepped_tiles_path=prepped_tiles,
                )
                outputs["filtered_annotations"] = fa.outputs.output_path

    # ================================================================ #
    # Pipeline definitions — one per start_from step, ×2 (seq/parallel)
    # ================================================================ #

    # ---- data_prep (full) ----
    @pipeline(compute=COMPUTE_CLUSTER, description="Full pipeline (parallel)")
    def pipe_data_prep(raw_slides_input):
        prep = components["data_prep"](input_data=raw_slides_input)
        if args.filter_tiles and "tile_filter" in components:
            tf = components["tile_filter"](trigger_path=prep.outputs.manifest_path, input_path=prep.outputs.output_path)
            tiles = tf.outputs.output_path
            trigger = tf.outputs.output_manifest
        else:
            tiles = prep.outputs.output_path
            trigger = prep.outputs.manifest_path
        seg = components["segment"](trigger_path=trigger, prepped_tiles_path=tiles)
        clu = components["cluster"](trigger_path=seg.outputs.output_manifest, segmentation_path=seg.outputs.output_path, prepped_tiles_path=tiles)
        if args.use_separate_clustering_cluster: clu.compute = clustering_cluster_target
        cls = components["classify"](trigger_path=clu.outputs.output_manifest, segmented_path=seg.outputs.output_path, prepped_tiles_path=tiles, cluster_path=clu.outputs.cluster_output)
        post = components["post_process"](segmentation_path=seg.outputs.output_path, classification_path=cls.outputs.output_path)
        outputs = {"final_output": post.outputs.output_path}
        _append_tail_steps(outputs, post.outputs.output_path, tiles)
        return outputs

    @pipeline(compute=COMPUTE_CLUSTER, description="Full pipeline (sequential)")
    def pipe_data_prep_seq(raw_slides_input):
        prep = components["data_prep"](input_data=raw_slides_input)
        if args.filter_tiles and "tile_filter" in components:
            tf = components["tile_filter"](input_path=prep.outputs.output_path)
            tiles = tf.outputs.output_path
        else:
            tiles = prep.outputs.output_path
        seg = components["segment"](prepped_tiles_path=tiles)
        clu = components["cluster"](segmentation_path=seg.outputs.output_path, prepped_tiles_path=tiles)
        if args.use_separate_clustering_cluster: clu.compute = clustering_cluster_target
        cls = components["classify"](segmented_path=seg.outputs.output_path, prepped_tiles_path=tiles, cluster_path=clu.outputs.cluster_output)
        post = components["post_process"](segmentation_path=seg.outputs.output_path, classification_path=cls.outputs.output_path)
        outputs = {"final_output": post.outputs.output_path}
        _append_tail_steps(outputs, post.outputs.output_path, tiles)
        return outputs

    # ---- segment ----
    @pipeline(compute=COMPUTE_CLUSTER, description="From segmentation onward (parallel)")
    def pipe_segment(prepped_in, manifest_in):
        seg = components["segment"](trigger_path=manifest_in, prepped_tiles_path=prepped_in)
        clu = components["cluster"](trigger_path=seg.outputs.output_manifest, segmentation_path=seg.outputs.output_path, prepped_tiles_path=prepped_in)
        if args.use_separate_clustering_cluster: clu.compute = clustering_cluster_target
        cls = components["classify"](trigger_path=clu.outputs.output_manifest, segmented_path=seg.outputs.output_path, prepped_tiles_path=prepped_in, cluster_path=clu.outputs.cluster_output)
        post = components["post_process"](segmentation_path=seg.outputs.output_path, classification_path=cls.outputs.output_path)
        outputs = {"final_output": post.outputs.output_path}
        _append_tail_steps(outputs, post.outputs.output_path, prepped_in)
        return outputs

    @pipeline(compute=COMPUTE_CLUSTER, description="From segmentation onward (sequential)")
    def pipe_segment_seq(prepped_in):
        seg = components["segment"](prepped_tiles_path=prepped_in)
        clu = components["cluster"](segmentation_path=seg.outputs.output_path, prepped_tiles_path=prepped_in)
        if args.use_separate_clustering_cluster: clu.compute = clustering_cluster_target
        cls = components["classify"](segmented_path=seg.outputs.output_path, prepped_tiles_path=prepped_in, cluster_path=clu.outputs.cluster_output)
        post = components["post_process"](segmentation_path=seg.outputs.output_path, classification_path=cls.outputs.output_path)
        outputs = {"final_output": post.outputs.output_path}
        _append_tail_steps(outputs, post.outputs.output_path, prepped_in)
        return outputs

    # ---- cluster ----
    @pipeline(compute=COMPUTE_CLUSTER, description="From clustering onward (parallel)")
    def pipe_cluster(prepped_in, segmented_in, seg_manifest_in):
        clu = components["cluster"](trigger_path=seg_manifest_in, segmentation_path=segmented_in, prepped_tiles_path=prepped_in)
        if args.use_separate_clustering_cluster: clu.compute = clustering_cluster_target
        cls = components["classify"](trigger_path=clu.outputs.output_manifest, segmented_path=segmented_in, prepped_tiles_path=prepped_in, cluster_path=clu.outputs.cluster_output)
        post = components["post_process"](segmentation_path=segmented_in, classification_path=cls.outputs.output_path)
        outputs = {"final_output": post.outputs.output_path}
        _append_tail_steps(outputs, post.outputs.output_path, prepped_in)
        return outputs

    @pipeline(compute=COMPUTE_CLUSTER, description="From clustering onward (sequential)")
    def pipe_cluster_seq(prepped_in, segmented_in):
        clu = components["cluster"](segmentation_path=segmented_in, prepped_tiles_path=prepped_in)
        if args.use_separate_clustering_cluster: clu.compute = clustering_cluster_target
        cls = components["classify"](segmented_path=segmented_in, prepped_tiles_path=prepped_in, cluster_path=clu.outputs.cluster_output)
        post = components["post_process"](segmentation_path=segmented_in, classification_path=cls.outputs.output_path)
        outputs = {"final_output": post.outputs.output_path}
        _append_tail_steps(outputs, post.outputs.output_path, prepped_in)
        return outputs

    # ---- classify ----
    @pipeline(compute=COMPUTE_CLUSTER, description="From classification onward (parallel)")
    def pipe_classify(prepped_in, segmented_in, clustered_in, clu_manifest_in):
        cls = components["classify"](trigger_path=clu_manifest_in, segmented_path=segmented_in, prepped_tiles_path=prepped_in, cluster_path=clustered_in)
        post = components["post_process"](segmentation_path=segmented_in, classification_path=cls.outputs.output_path)
        outputs = {"final_output": post.outputs.output_path}
        _append_tail_steps(outputs, post.outputs.output_path, prepped_in)
        return outputs

    @pipeline(compute=COMPUTE_CLUSTER, description="From classification onward (sequential)")
    def pipe_classify_seq(prepped_in, segmented_in, clustered_in):
        cls = components["classify"](segmented_path=segmented_in, prepped_tiles_path=prepped_in, cluster_path=clustered_in)
        post = components["post_process"](segmentation_path=segmented_in, classification_path=cls.outputs.output_path)
        outputs = {"final_output": post.outputs.output_path}
        _append_tail_steps(outputs, post.outputs.output_path, prepped_in)
        return outputs

    # ---- post_process ----
    @pipeline(compute=COMPUTE_CLUSTER, description="From post-processing onward")
    def pipe_post_process(prepped_in, segmented_in, classified_in):
        post = components["post_process"](segmentation_path=segmented_in, classification_path=classified_in)
        outputs = {"final_output": post.outputs.output_path}
        _append_tail_steps(outputs, post.outputs.output_path, prepped_in)
        return outputs

    # ---- annotate ----
    @pipeline(compute=COMPUTE_CLUSTER, description="Annotate existing post-process results")
    def pipe_annotate(postprocess_in, prepped_in):
        outputs = {}
        if "annotation" in components:
            annotate = components["annotation"](annotations_json=postprocess_in, prepped_tiles_path=prepped_in)
            outputs["annotations"] = annotate.outputs.output_path
        return outputs

    # ---- cluster_tiles (+ filtered annotations) ----
    @pipeline(compute=COMPUTE_CLUSTER, description="Extract cluster tiles (and filtered annotations)")
    def pipe_cluster_tiles(postprocess_in, prepped_in):
        outputs = {}
        if "cluster_tiles" in components:
            ct = components["cluster_tiles"](annotations_json=postprocess_in, prepped_tiles_path=prepped_in)
            outputs["cluster_tiles"] = ct.outputs.output_path
            if args.enable_filtered_annotations and "filtered_annotation" in components:
                fa = components["filtered_annotation"](cluster_tiles_path=ct.outputs.output_path, prepped_tiles_path=prepped_in)
                outputs["filtered_annotations"] = fa.outputs.output_path
        return outputs

    # ================================================================ #
    # Pick + submit
    # ================================================================ #
    step = args.start_from
    logging.info("Submitting start_from=%s  (sequential=%s)", step, use_sequential)
    seq_tag = "seq_" if use_sequential else ""
    exp_name = f"v1_{seq_tag}{step}_{param_string}_{timestamp}"

    def _inp(uri):
        return Input(type=AssetTypes.URI_FOLDER, path=uri)

    job = None

    if step == "data_prep":
        if use_sequential:
            job = pipe_data_prep_seq(raw_slides_input=_inp(args.raw_slides_uri))
        else:
            job = pipe_data_prep(raw_slides_input=_inp(args.raw_slides_uri))

    elif step == "tile_filter":
        # tile_filter → segment → ... (reuse the data_prep pipeline with prepped input as raw)
        # Tile filter is the first thing that runs on already-prepped tiles.
        # Re-use the from-segment pipeline since filter is embedded in data_prep flow.
        logging.info("tile_filter start is equivalent to running --start_from segment with --filter_tiles")
        if use_sequential:
            job = pipe_segment_seq(prepped_in=_inp(args.prepped_data_uri))
        else:
            if not args.prepped_manifest_uri:
                logging.error("Parallel start_from=tile_filter requires --prepped_manifest_uri")
                return
            job = pipe_segment(prepped_in=_inp(args.prepped_data_uri), manifest_in=_inp(args.prepped_manifest_uri))

    elif step == "segment":
        if use_sequential:
            job = pipe_segment_seq(prepped_in=_inp(args.prepped_data_uri))
        else:
            if not args.prepped_manifest_uri:
                logging.error("Parallel start_from=segment requires --prepped_manifest_uri")
                return
            job = pipe_segment(prepped_in=_inp(args.prepped_data_uri), manifest_in=_inp(args.prepped_manifest_uri))

    elif step == "cluster":
        if use_sequential:
            job = pipe_cluster_seq(prepped_in=_inp(args.prepped_data_uri), segmented_in=_inp(args.segmented_data_uri))
        else:
            if not args.segmented_manifest_uri:
                logging.error("Parallel start_from=cluster requires --segmented_manifest_uri")
                return
            job = pipe_cluster(
                prepped_in=_inp(args.prepped_data_uri),
                segmented_in=_inp(args.segmented_data_uri),
                seg_manifest_in=_inp(args.segmented_manifest_uri),
            )

    elif step == "classify":
        if use_sequential:
            job = pipe_classify_seq(
                prepped_in=_inp(args.prepped_data_uri),
                segmented_in=_inp(args.segmented_data_uri),
                clustered_in=_inp(args.clustered_data_uri),
            )
        else:
            if not args.clustered_manifest_uri:
                logging.error("Parallel start_from=classify requires --clustered_manifest_uri")
                return
            job = pipe_classify(
                prepped_in=_inp(args.prepped_data_uri),
                segmented_in=_inp(args.segmented_data_uri),
                clustered_in=_inp(args.clustered_data_uri),
                clu_manifest_in=_inp(args.clustered_manifest_uri),
            )

    elif step == "post_process":
        job = pipe_post_process(
            prepped_in=_inp(args.prepped_data_uri),
            segmented_in=_inp(args.segmented_data_uri),
            classified_in=_inp(args.classified_data_uri),
        )

    elif step == "annotate":
        if not args.enable_annotations:
            logging.error("start_from=annotate requires --enable_annotations flag")
            return
        job = pipe_annotate(
            postprocess_in=_inp(args.postprocess_data_uri),
            prepped_in=_inp(args.prepped_data_uri),
        )

    elif step == "cluster_tiles":
        if not args.enable_cluster_tiles:
            logging.error("start_from=cluster_tiles requires --enable_cluster_tiles flag")
            return
        job = pipe_cluster_tiles(
            postprocess_in=_inp(args.postprocess_data_uri),
            prepped_in=_inp(args.prepped_data_uri),
        )

    if job is None:
        logging.error("No pipeline job created — check your --start_from and flags.")
        return

    submitted = ml_client.jobs.create_or_update(job, experiment_name=exp_name)
    logging.info("Job submitted: %s", submitted.studio_url)
    print("View in studio:", submitted.studio_url)

# --------------------------------------------------------------------------- #
def validate_environment(args: argparse.Namespace):
    """Validate that all required environment variables and URIs are set."""
    missing = []
    
    if not SUBSCRIPTION_ID:
        missing.append("AZURE_SUBSCRIPTION_ID")
    if not RESOURCE_GROUP:
        missing.append("AZURE_RESOURCE_GROUP")
    if not WORKSPACE_NAME:
        missing.append("AZURE_ML_WORKSPACE_NAME")

    # Steps that involve classification need OpenAI
    steps_needing_openai = {"data_prep", "tile_filter", "segment", "cluster", "classify"}
    if args.start_from in steps_needing_openai and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please create a .env file based on .env.example and fill in your values."
        )

    # Validate required URIs based on start_from
    step = args.start_from
    uri_errors = []
    if step in ("tile_filter", "segment", "cluster", "classify", "post_process", "annotate", "cluster_tiles"):
        if not args.prepped_data_uri:
            uri_errors.append("--prepped_data_uri is required when starting from '{}'".format(step))
    if step in ("cluster", "classify", "post_process"):
        if not args.segmented_data_uri:
            uri_errors.append("--segmented_data_uri is required when starting from '{}'".format(step))
    if step == "classify":
        if not args.clustered_data_uri:
            uri_errors.append("--clustered_data_uri is required when starting from '{}'".format(step))
    if step == "post_process":
        if not args.classified_data_uri:
            uri_errors.append("--classified_data_uri is required when starting from '{}'".format(step))
    if step in ("annotate", "cluster_tiles"):
        if not args.postprocess_data_uri:
            uri_errors.append("--postprocess_data_uri is required when starting from '{}'".format(step))

    if uri_errors:
        raise ValueError(
            "Missing required URI arguments:\n  " + "\n  ".join(uri_errors)
        )

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    run_pipeline()
