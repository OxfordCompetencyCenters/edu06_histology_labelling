import argparse
import os
import logging
from datetime import datetime
from typing import List

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import Input, Output, command
from azure.ai.ml.entities import Environment
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes

# Validate OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set locally!")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Azure ML Configuration
SUBSCRIPTION_ID = "bbeb7561-f822-4950-82f6-64dcae8a93ab"
RESOURCE_GROUP = "AIMLCC-DEV-RG"
WORKSPACE_NAME = "edu06_histology_img_segmentation"
COMPUTE_CLUSTER = "edu06-gpu-compute-cluster"

def build_enhanced_data_prep_command(tile_sizes: List[int], target_mpp: float, enable_multi_resolution: bool) -> str:
    """Build enhanced data preparation command with multi-resolution support."""
    cmd_parts = [
        "python data_prep.py",
        "--input_data ${{inputs.input_data}}",
        "--output_path ${{outputs.output_path}}",
        f"--target_mpp {target_mpp}"
    ]
    
    if tile_sizes:
        tile_size_str = " ".join(str(size) for size in tile_sizes)
        cmd_parts.append(f"--tile_sizes {tile_size_str}")
    
    if enable_multi_resolution:
        cmd_parts.append("--enable_multi_resolution")
    
    return " ".join(cmd_parts)

def build_sam_med_segment_command(sam_checkpoint: str, device: str = "cuda") -> str:
    """Build the command string for the SAM-Med segment component."""
    return (
        f"python segment_sam_med.py "
        f"--prepped_tiles_path ${{inputs.prepped_tiles_path}} "
        f"--output_path ${{outputs.segment_output}} "
        f"--sam_checkpoint {sam_checkpoint} "
        f"--device {device}"
    )

def build_token_cluster_command(
    token_model: str, 
    clustering_method: str, 
    n_clusters: int, 
    use_crf: bool, 
    device: str = "cuda",
    use_gpu: bool = False,
    use_umap: bool = False,
    umap_n_components: int = 50,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1
) -> str:
    """Build the command string for the advanced token clustering component."""
    cmd_parts = [
        "python token_clustering.py",
        "--segmentation_path ${{inputs.segmentation_path}}",
        "--prepped_tiles_path ${{inputs.prepped_tiles_path}}",
        "--output_path ${{outputs.cluster_output}}",
        f"--token_model {token_model}",
        f"--clustering_method {clustering_method}",
        f"--n_clusters {n_clusters}",
        f"--device {device}"
    ]
    
    if use_crf:
        cmd_parts.append("--use_crf")
    
    if use_gpu:
        cmd_parts.append("--use_gpu")
    
    if use_umap:
        cmd_parts.append("--use_umap")
        cmd_parts.append(f"--umap_n_components {umap_n_components}")
        cmd_parts.append(f"--umap_n_neighbors {umap_n_neighbors}")
        cmd_parts.append(f"--umap_min_dist {umap_min_dist}")

    return " ".join(cmd_parts)


def build_components(
    env,
    data_prep_output_uri: str,
    segment_output_uri: str,
    cluster_output_uri: str,
    classify_output_uri: str,
    postprocess_output_uri: str,
    classify_per_cluster: int,
    # Enhanced Parameters
    sam_checkpoint: str,
    tile_sizes: List[int],
    target_mpp: float,
    enable_multi_resolution: bool,
    device: str,
    # Token Clustering Parameters
    token_model: str,
    clustering_method: str,
    n_clusters: int,
    use_crf: bool,
    # GPU and UMAP Parameters
    use_gpu_clustering: bool,
    use_umap: bool,
    umap_n_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float
):
    """Create command components for the pipeline steps."""
    
    logging.info("Building pipeline components...")
    
    # 1. Enhanced Data Preparation
    data_prep_cmd = build_enhanced_data_prep_command(
        tile_sizes=tile_sizes,
        target_mpp=target_mpp,
        enable_multi_resolution=enable_multi_resolution
    )
    
    data_prep_component = command(
        name="EnhancedDataPrep",
        display_name="Enhanced Data Prep with Multi-Resolution",
        inputs={"input_data": Input(type=AssetTypes.URI_FOLDER)},
        outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=data_prep_output_uri)},
        code="./",
        command=data_prep_cmd,
        environment=env,
    )

    # 2. SAM-Med Segmentation
    segment_cmd = build_sam_med_segment_command(
        sam_checkpoint=sam_checkpoint,
        device=device
    )
    
    segment_component = command(
        name="SAMSegmentation",
        display_name="SAM-Med Cell Segmentation",
        inputs={"prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER)},
        outputs={"segment_output": Output(type=AssetTypes.URI_FOLDER, path=segment_output_uri)},
        code="./",
        command=segment_cmd,
        environment=env,
    )    # 3. Token Clustering
    cluster_cmd = build_token_cluster_command(
        token_model=token_model,
        clustering_method=clustering_method,
        n_clusters=n_clusters,
        use_crf=use_crf,
        device=device,
        use_gpu=use_gpu_clustering,
        use_umap=use_umap,
        umap_n_components=umap_n_components,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist
    )
    
    cluster_component = command(
        name="TokenClustering",
        display_name="Advanced Token Clustering",
        inputs={
            "segmentation_path": Input(type=AssetTypes.URI_FOLDER),
            "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={"cluster_output": Output(type=AssetTypes.URI_FOLDER, path=cluster_output_uri)},
        code="./",
        command=cluster_cmd,
        environment=env,
    )

    # 4. Classification
    env_vars = {"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}
    classify_component = command(
        name="EnhancedClassification",
        display_name="GPT-4o Cell Classification",
        inputs={
            "segmented_path": Input(type=AssetTypes.URI_FOLDER),
            "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER),
            "cluster_path": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=classify_output_uri)},
        code="./",
        command=(
            f"python classify.py "
            f"--segmented_path ${{inputs.segmented_path}} "
            f"--prepped_tiles_path ${{inputs.prepped_tiles_path}} "
            f"--clustered_cells_path ${{inputs.cluster_path}} "
            f"--output_path ${{outputs.output_path}} "
            f"--num_classes 4 "
            f"--classify_per_cluster {classify_per_cluster}"
        ),
        environment=env,
        environment_variables=env_vars,
    )

    # 5. Post-Processing
    post_process_component = command(
        name="EnhancedPostProcess",
        display_name="Post-Processing with Analytics",
        inputs={
            "segmentation_path": Input(type=AssetTypes.URI_FOLDER),
            "classification_path": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={"output_path": Output(type=AssetTypes.URI_FOLDER, path=postprocess_output_uri)},
        code="./",
        command=(
            f"python post_process.py "
            f"--segmentation_path ${{inputs.segmentation_path}} "
            f"--classification_path ${{inputs.classification_path}} "
            f"--output_path ${{outputs.output_path}} "
            f"--generate_analytics"
        ),
        environment=env,
    )

    logging.info("Components built successfully.")
    return {
        "data_prep": data_prep_component,
        "segment": segment_component,
        "cluster": cluster_component,
        "classify": classify_component,
        "post_process": post_process_component
    }

def run_pipeline():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Enhanced Histology Image Analysis Pipeline on Azure ML")

    # Pipeline Mode
    parser.add_argument(
        "--mode", type=str, choices=["prep_only", "full", "seg_cluster_cls", "cluster_cls", "classify_only"],
        default="full", help="Which part of the pipeline to run."
    )

    # Input Data URIs
    parser.add_argument("--raw_slides_uri", type=str, 
                       default="azureml://datastores/workspaceblobstore/paths/UI/2025-03-05_125751_UTC/", 
                       help="URI folder of raw slides (for modes: full, prep_only).")
    parser.add_argument("--prepped_data_uri", type=str, 
                       default="azureml://datastores/workspaceblobstore/paths/my_prepped_data/", 
                       help="URI folder of prepped tiles (for modes: seg_cluster_cls, cluster_cls, classify_only).")
    parser.add_argument("--segmented_data_uri", type=str, 
                       default="azureml://datastores/workspaceblobstore/paths/my_segmented_data/", 
                       help="URI folder of segmented tiles (for modes: cluster_cls, classify_only).")
    parser.add_argument("--clustered_data_uri", type=str, 
                       default="azureml://datastores/workspaceblobstore/paths/my_clustered_data/", 
                       help="URI folder of clustered results (for modes: classify_only).")
    
    # Classification Parameters
    parser.add_argument("--classify_per_cluster", type=int, default=10, 
                       help="Number of bounding boxes per cluster to classify.")
    
    # Enhanced Data Preparation Parameters
    parser.add_argument("--tile_sizes", type=int, nargs="+", default=[256, 512], 
                       help="Tile sizes for multi-resolution processing")
    parser.add_argument("--target_mpp", type=float, default=0.25, 
                       help="Target microns per pixel for normalization")
    parser.add_argument("--enable_multi_resolution", action="store_true", 
                       help="Enable multi-resolution tiling")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to use (cuda/cpu)")

    # SAM-Med Parameters
    parser.add_argument("--sam_checkpoint", type=str, default="sam_med2d_b.pth",
                       help="SAM-Med checkpoint name")    # Token Clustering Parameters
    parser.add_argument("--token_model", type=str, choices=["dinov2", "uni"], default="dinov2",
                       help="Vision transformer model for feature extraction")
    parser.add_argument("--clustering_method", type=str, choices=["kmeans", "hdbscan"], default="kmeans",
                       help="Clustering algorithm for tokens")
    parser.add_argument("--n_clusters", type=int, default=3,
                       help="Number of clusters for k-means")
    parser.add_argument("--use_crf", action="store_true",
                       help="Apply CRF smoothing for spatial consistency")
    
    # GPU and UMAP Parameters
    parser.add_argument("--use_gpu_clustering", action="store_true",
                       help="Use GPU acceleration for clustering (requires cuML)")
    parser.add_argument("--use_umap", action="store_true",
                       help="Apply UMAP dimensionality reduction before clustering")
    parser.add_argument("--umap_n_components", type=int, default=50,
                       help="Target dimensions for UMAP reduction")
    parser.add_argument("--umap_n_neighbors", type=int, default=15,
                       help="UMAP n_neighbors parameter")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                       help="UMAP min_dist parameter")

    args = parser.parse_args()
    
    logging.info("Pipeline configuration:")
    logging.info(f"  Mode: {args.mode}")
    logging.info(f"  SAM checkpoint: {args.sam_checkpoint}")
    logging.info(f"  Token model: {args.token_model}")
    logging.info(f"  Clustering method: {args.clustering_method}")
    logging.info(f"  Number of clusters: {args.n_clusters}")
    logging.info(f"  Device: {args.device}")

    # Generate timestamp for unique outputs
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_base_uri = f"azureml://datastores/workspaceblobstore/paths/{timestamp}_enhanced_pipeline_outputs"

    # Build unique paths for each step
    data_prep_output_uri = f"{output_base_uri}/data_prep/"
    segment_output_uri = f"{output_base_uri}/segment/"
    cluster_output_uri = f"{output_base_uri}/cluster/"
    classify_output_uri = f"{output_base_uri}/classify/"
    postprocess_output_uri = f"{output_base_uri}/postprocess/"    # Connect to Azure ML workspace
    logging.info("Connecting to Azure ML Workspace...")
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME
        )
        logging.info("Connected to Azure ML workspace successfully.")
    except Exception as e:
        logging.error(f"Failed to connect to Azure ML workspace: {e}")
        return

    # Create or update the environment
    env = Environment(
        name="edu06_env_enhanced",
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"
    )
    logging.info("Setting up environment...")
    try:
        ml_client.environments.create_or_update(env)
        logging.info("Environment ready.")
    except Exception as e:
        logging.error(f"Failed to create or update environment: {e}")
        return    # Build the pipeline components
    try:        components = build_components(
            env=env,
            data_prep_output_uri=data_prep_output_uri,
            segment_output_uri=segment_output_uri,
            cluster_output_uri=cluster_output_uri,
            classify_output_uri=classify_output_uri,
            postprocess_output_uri=postprocess_output_uri,
            classify_per_cluster=args.classify_per_cluster,
            # Enhanced pipeline parameters
            sam_checkpoint=args.sam_checkpoint,
            tile_sizes=args.tile_sizes,
            target_mpp=args.target_mpp,
            enable_multi_resolution=args.enable_multi_resolution,
            device=args.device,
            token_model=args.token_model,
            clustering_method=args.clustering_method,
            n_clusters=args.n_clusters,
            use_crf=args.use_crf,
            # GPU and UMAP parameters
            use_gpu_clustering=args.use_gpu_clustering,
            use_umap=args.use_umap,
            umap_n_components=args.umap_n_components,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist
        )
    except Exception as e:
        logging.error(f"Failed to build pipeline components: {e}")
        return    # Define pipeline functions for each mode
    @pipeline(compute=COMPUTE_CLUSTER, description="Complete enhanced pipeline with SAM-Med and token clustering")
    def full_enhanced_pipeline(raw_slides_input):
        prep_step = components["data_prep"](input_data=raw_slides_input)
        seg_step = components["segment"](prepped_tiles_path=prep_step.outputs.output_path)
        cluster_step = components["cluster"](
            segmentation_path=seg_step.outputs.segment_output,
            prepped_tiles_path=prep_step.outputs.output_path)
        cls_step = components["classify"](
            segmented_path=seg_step.outputs.segment_output,
            prepped_tiles_path=prep_step.outputs.output_path,
            cluster_path=cluster_step.outputs.cluster_output
        )
        post_step = components["post_process"](
            segmentation_path=seg_step.outputs.segment_output,
            classification_path=cls_step.outputs.output_path
        )
        return {"final_output": post_step.outputs.output_path}

    @pipeline(compute=COMPUTE_CLUSTER, description="Enhanced data prep only")
    def enhanced_data_prep_pipeline(raw_slides_input):
        prep_step = components["data_prep"](input_data=raw_slides_input)
        return {"prepped_data": prep_step.outputs.output_path}

    @pipeline(compute=COMPUTE_CLUSTER, description="SAM-Med Segment -> Token Cluster -> Classify -> Post-process")
    def seg_cluster_cls_enhanced_pipeline(prepped_tiles_input):
        seg_step = components["segment"](prepped_tiles_path=prepped_tiles_input)
        cluster_step = components["cluster"](
            segmentation_path=seg_step.outputs.segment_output,
            prepped_tiles_path=prepped_tiles_input)
        cls_step = components["classify"](
            segmented_path=seg_step.outputs.segment_output,
            prepped_tiles_path=prepped_tiles_input,
            cluster_path=cluster_step.outputs.cluster_output
        )
        post_step = components["post_process"](
            segmentation_path=seg_step.outputs.segment_output,
            classification_path=cls_step.outputs.output_path
        )
        return {"final_output": post_step.outputs.output_path}

    @pipeline(compute=COMPUTE_CLUSTER, description="Token Cluster -> Classify -> Post-process")
    def cluster_cls_enhanced_pipeline(prepped_tiles_input, segmented_input):
        cluster_step = components["cluster"](
            segmentation_path=segmented_input,
            prepped_tiles_path=prepped_tiles_input)
        cls_step = components["classify"](
            segmented_path=segmented_input,
            prepped_tiles_path=prepped_tiles_input,
            cluster_path=cluster_step.outputs.cluster_output
        )
        post_step = components["post_process"](
            segmentation_path=segmented_input,
            classification_path=cls_step.outputs.output_path
        )
        return {"final_output": post_step.outputs.output_path}

    @pipeline(compute=COMPUTE_CLUSTER, description="Classify + Post-process only")
    def classify_only_enhanced_pipeline(prepped_tiles_input, segmented_input, cluster_input):
        cls_step = components["classify"](
            segmented_path=segmented_input,
            prepped_tiles_path=prepped_tiles_input,
            cluster_path=cluster_input
        )
        post_step = components["post_process"](
            segmentation_path=segmented_input,
            classification_path=cls_step.outputs.output_path
        )
        return {"final_output": post_step.outputs.output_path}

    # Build and submit the pipeline
    logging.info(f"Building {args.mode} pipeline...")
    experiment_name = f"enhanced_histology_{args.mode}_{timestamp}"

    try:
        if args.mode == "prep_only":
            pipeline_job = enhanced_data_prep_pipeline(
                raw_slides_input=Input(type=AssetTypes.URI_FOLDER, path=args.raw_slides_uri)
            )
        elif args.mode == "seg_cluster_cls":
            pipeline_job = seg_cluster_cls_enhanced_pipeline(
                prepped_tiles_input=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri)
            )
        elif args.mode == "cluster_cls":
            pipeline_job = cluster_cls_enhanced_pipeline(
                prepped_tiles_input=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri),
                segmented_input=Input(type=AssetTypes.URI_FOLDER, path=args.segmented_data_uri)
            )
        elif args.mode == "classify_only":
            pipeline_job = classify_only_enhanced_pipeline(
                prepped_tiles_input=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri),
                segmented_input=Input(type=AssetTypes.URI_FOLDER, path=args.segmented_data_uri),
                cluster_input=Input(type=AssetTypes.URI_FOLDER, path=args.clustered_data_uri)
            )
        else:  # args.mode == "full"
            pipeline_job = full_enhanced_pipeline(
                raw_slides_input=Input(type=AssetTypes.URI_FOLDER, path=args.raw_slides_uri)
            )

        if pipeline_job:
            logging.info("Submitting pipeline job to Azure ML...")
            submitted_job = ml_client.jobs.create_or_update(pipeline_job, experiment_name=experiment_name)
            
            logging.info(f"Pipeline submitted successfully!")
            logging.info(f"Job ID: {submitted_job.id}")
            logging.info(f"Studio URL: {submitted_job.studio_url}")
            print(f"{args.mode} pipeline submitted! View here: {submitted_job.studio_url}")
        else:
            logging.error(f"Pipeline job object was not created for mode: {args.mode}")

    except Exception as e:
        logging.error(f"Failed to submit pipeline for mode {args.mode}: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()