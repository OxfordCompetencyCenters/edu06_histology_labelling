from __future__ import annotations
import argparse, logging, os, sys
from datetime import datetime
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

# --------------------------------------------------------------------------- #
# Basic config – edit if your workspace details change
# --------------------------------------------------------------------------- #
SUBSCRIPTION_ID  = "bbeb7561-f822-4950-82f6-64dcae8a93ab"
RESOURCE_GROUP   = "AIMLCC-DEV-RG"
WORKSPACE_NAME   = "edu06_histology_img_segmentation"
COMPUTE_CLUSTER  = "edu06-gpu-compute-cluster"

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
    # strings (e.g. magnification list)
    return str(value).replace(".", "pt").replace(",", "-")

def build_param_string(args):
    """Compact signature of key hyper-params for folder/exp names."""
    parts = [
        f"prob_{format_param_for_name(args.segment_cellprob_threshold)}",
        f"flow_{format_param_for_name(args.segment_flow_threshold)}",
        f"eps_{format_param_for_name(args.cluster_eps)}",
        f"mins_{args.cluster_min_samples}",
    ]
    if args.segment_use_gpu:
        parts.append("segGPU")
    if args.cluster_normalize:
        parts.append("norm")
    if args.cluster_use_gpu:
        parts.append("cluGPU")
    if args.cluster_use_umap:
        parts.append(f"umap_{args.cluster_umap_components}")
    # NEW → record magnifications & target tiles, if any
    parts.append(f"mag_{format_param_for_name(args.magnifications)}")
    if args.num_tiles is not None:
        parts.append(f"ntiles_{args.num_tiles}")
    # NEW → tile filtering params
    if args.filter_tiles:
        parts.append("filtered")
        parts.append(f"edge_{format_param_for_name(args.filter_min_edge_density)}")
    return "_".join(parts)

def build_cluster_command(**kwargs) -> str:
    """Return the shell command for cluster.py with only the flags we need."""
    cmd  = (
        "python cluster.py "
        "--segmentation_path ${{inputs.segmentation_path}} "
        "--prepped_tiles_path ${{inputs.prepped_tiles_path}} "
        "--output_path ${{outputs.cluster_output}} "
    )
    # positional numeric args
    if kwargs["cluster_eps"] is not None:
        cmd += f"--eps {kwargs['cluster_eps']} "
    cmd += f"--min_samples {kwargs['cluster_min_samples']} "
    # flags
    if kwargs["cluster_use_gpu"]:
        cmd += "--gpu "
    if kwargs["cluster_normalize"]:
        cmd += "--normalize_embeddings "
    if kwargs["cluster_use_umap"]:
        cmd += (
            "--use_umap "
            f"--umap_n_components {kwargs['cluster_umap_components']} "
            f"--umap_n_neighbors {kwargs['cluster_umap_neighbors']} "
            f"--umap_min_dist {kwargs['cluster_umap_min_dist']} "
            f"--umap_metric {kwargs['cluster_umap_metric']} "
        )
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
    segment_flow_threshold: float,
    segment_cellprob_threshold: float,
    segment_use_gpu: bool,
    cluster_eps: float | None,
    cluster_min_samples: int,
    cluster_use_gpu: bool,
    cluster_normalize: bool,
    cluster_use_umap: bool,
    cluster_umap_components: int,
    cluster_umap_neighbors: int,
    cluster_umap_min_dist: float,
    cluster_umap_metric: str,
):
    """Create the Azure ML command components used in the pipeline."""
    logging.info("Building component objects …")

    dp_cmd = (
        "python data_prep.py "
        "--input_data ${{inputs.input_data}} "
        "--output_path ${{outputs.output_path}} "
        f"--tile_size 512 "
        f"--magnifications \"{magnifications}\" "
        f"{'--num_tiles ' + str(num_tiles) if num_tiles is not None else ''}"
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

    seg_cmd = (
        "python segment.py "
        "--input_path ${{inputs.prepped_tiles_path}} "
        "--output_path ${{outputs.output_path}} "
        "--model_type cyto2 --chan 2 --chan2 1 "
        f"--flow_threshold {segment_flow_threshold} "
        f"--cellprob_threshold {segment_cellprob_threshold} "
        f"{'--segment_use_gpu' if segment_use_gpu else ''}"
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
        ),
        environment=env,
    )

    classify_env = {"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}
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

    components = {
        "data_prep"   : data_prep_component,
        "segment"     : segment_component,
        "cluster"     : cluster_component,
        "classify"    : classify_component,
        "post_process": post_process_component,
    }
    
    if tile_filter_component:
        components["tile_filter"] = tile_filter_component
    
    return components

# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def run_pipeline():
    # ---------------- CLI ---------------- #
    p = argparse.ArgumentParser("Launch Azure ML histology pipeline")
    p.add_argument("--mode", choices=[
        "prep_only", "full", "seg_cluster_cls", "cluster_cls", "classify_only"
    ], default="full")

    # Input URIs
    p.add_argument("--raw_slides_uri", default="azureml://datastores/workspaceblobstore/paths/UI/2025-03-05_125751_UTC/")
    p.add_argument("--prepped_data_uri", default="azureml://datastores/workspaceblobstore/paths/my_prepped_data/")
    p.add_argument("--segmented_data_uri", default="azureml://datastores/workspaceblobstore/paths/my_segmented_data/")
    p.add_argument("--clustered_data_uri", default="azureml://datastores/workspaceblobstore/paths/my_clustered_data/")

    # Data-prep params
    p.add_argument("--magnifications", type=str, default="1.0",
                   help="Comma-separated list of relative magnifications, e.g. '1.0,0.9,0.8'")
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
    p.add_argument("--segment_flow_threshold", type=float, default=0.4)
    p.add_argument("--segment_cellprob_threshold", type=float, default=0.0)
    p.add_argument("--segment_use_gpu", action="store_true")

    # Clustering
    p.add_argument("--cluster_eps", type=float, default=None)
    p.add_argument("--cluster_min_samples", type=int, default=5)
    p.add_argument("--cluster_use_gpu", action="store_true")
    p.add_argument("--cluster_normalize", action="store_true")
    p.add_argument("--cluster_use_umap", action="store_true")
    p.add_argument("--cluster_umap_components", type=int, default=50)
    p.add_argument("--cluster_umap_neighbors", type=int, default=15)
    p.add_argument("--cluster_umap_min_dist", type=float, default=0.1)
    p.add_argument("--cluster_umap_metric", default="euclidean")

    args = p.parse_args()
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info("Args: %s", vars(args))    # ------------- Paths & names ------------- #
    timestamp     = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    param_string  = build_param_string(args)
    base_uri      = f"azureml://datastores/workspaceblobstore/paths/{timestamp}_v1_{param_string}"
    dp_uri        = f"{base_uri}/data_prep/"
    filter_uri    = f"{base_uri}/tile_filter/" if args.filter_tiles else None
    seg_uri       = f"{base_uri}/segment/"
    clu_uri       = f"{base_uri}/cluster/"
    cls_uri       = f"{base_uri}/classify/"
    post_uri      = f"{base_uri}/postprocess/"

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
        name="edu06_env_revised",
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest",
    )
    ml_client.environments.create_or_update(env)    # ------------- Build components ------------- #
    components = build_components(
        env=env,
        data_prep_output_uri=dp_uri,
        tile_filter_output_uri=filter_uri,
        segment_output_uri=seg_uri,
        cluster_output_uri=clu_uri,
        classify_output_uri=cls_uri,
        postprocess_output_uri=post_uri,
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
        segment_flow_threshold=args.segment_flow_threshold,
        segment_cellprob_threshold=args.segment_cellprob_threshold,
        segment_use_gpu=args.segment_use_gpu,
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
    )    # ------------- Define pipelines (updated for tile filtering) ------------- #
    @pipeline(compute=COMPUTE_CLUSTER, description="Full pipeline")
    def full_pipeline(raw_slides_input):
        prep = components["data_prep"](input_data=raw_slides_input)
        
        # Use filtered tiles if filtering is enabled, otherwise use prep output directly
        if args.filter_tiles and "tile_filter" in components:
            filtered = components["tile_filter"](input_path=prep.outputs.output_path)
            tiles_for_segmentation = filtered.outputs.output_path
            tiles_for_clustering = prep.outputs.output_path  # Use original for clustering reference
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
        return {"final_output": post.outputs.output_path}

    @pipeline(compute=COMPUTE_CLUSTER, description="Data prep only")
    def data_prep_pipeline(slides_in):
        prep = components["data_prep"](input_data=slides_in)
        if args.filter_tiles and "tile_filter" in components:
            filtered = components["tile_filter"](input_path=prep.outputs.output_path)
            return {"prepped": prep.outputs.output_path, "filtered": filtered.outputs.output_path}
        return {"prepped": prep.outputs.output_path}

    @pipeline(compute=COMPUTE_CLUSTER, description="Seg→Clu→Cls→Post")
    def seg_cluster_cls_pipeline(prepped_in):
        # Assume prepped_in is already filtered if needed
        seg = components["segment"](prepped_tiles_path=prepped_in)
        clu = components["cluster"](segmentation_path=seg.outputs.output_path,
                                   prepped_tiles_path=prepped_in)
        cls = components["classify"](segmented_path=seg.outputs.output_path,
                                    prepped_tiles_path=prepped_in,
                                    cluster_path=clu.outputs.cluster_output)
        post = components["post_process"](segmentation_path=seg.outputs.output_path,
                                         classification_path=cls.outputs.output_path)
        return {"final_output": post.outputs.output_path}

    @pipeline(compute=COMPUTE_CLUSTER, description="Clu→Cls→Post")
    def cluster_cls_pipeline(prepped_in, segmented_in):
        clu = components["cluster"](segmentation_path=segmented_in,
                                   prepped_tiles_path=prepped_in)
        cls = components["classify"](segmented_path=segmented_in,
                                    prepped_tiles_path=prepped_in,
                                    cluster_path=clu.outputs.cluster_output)
        post = components["post_process"](segmentation_path=segmented_in,
                                         classification_path=cls.outputs.output_path)
        return {"final_output": post.outputs.output_path}

    @pipeline(compute=COMPUTE_CLUSTER, description="Cls→Post")
    def classify_only_pipeline(prepped_in, segmented_in, cluster_in):
        cls = components["classify"](segmented_path=segmented_in,
                                    prepped_tiles_path=prepped_in,
                                    cluster_path=cluster_in)
        post = components["post_process"](segmentation_path=segmented_in,
                                         classification_path=cls.outputs.output_path)
        return {"final_output": post.outputs.output_path}

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
    else:  # full
        job = full_pipeline(raw_slides_input=Input(type=AssetTypes.URI_FOLDER, path=args.raw_slides_uri))

    submitted = ml_client.jobs.create_or_update(job, experiment_name=exp_name)
    logging.info("Job submitted: %s", submitted.studio_url)
    print("View in studio:", submitted.studio_url)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set!")
    run_pipeline()
