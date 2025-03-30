import argparse
import os
import logging
from datetime import datetime

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import Input, Output, command
from azure.ai.ml.entities import Environment
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set locally!")

# --------------------------------------------------
# Setup logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

SUBSCRIPTION_ID = "bbeb7561-f822-4950-82f6-64dcae8a93ab" # Replace if needed
RESOURCE_GROUP = "AIMLCC-DEV-RG" # Replace if needed
WORKSPACE_NAME = "edu06_histology_img_segmentation" # Replace if needed
COMPUTE_CLUSTER = "edu06-gpu-compute-cluster" # Replace if needed

# Define a helper function to build the cluster command dynamically
def build_cluster_command(
    cluster_eps: float | None,
    cluster_min_samples: int,
    cluster_use_gpu: bool,
    cluster_normalize: bool,
    cluster_use_umap: bool,
    cluster_umap_components: int,
    cluster_umap_neighbors: int,
    cluster_umap_min_dist: float,
    cluster_umap_metric: str,
    # Add dbscan metric if you want to control it from pipeline level
    # cluster_dbscan_metric: str = "euclidean"
    ):
    """Builds the command string for the cluster component."""
    base_cmd = (
        "python cluster.py "
        "--segmentation_path ${{inputs.segmentation_path}} "
        "--prepped_tiles_path ${{inputs.prepped_tiles_path}} "
        "--output_path ${{outputs.cluster_output}}"
    )

    # Add optional float/int arguments
    if cluster_eps is not None:
        base_cmd += f" --eps {cluster_eps}"
    # min_samples always has a default, so pass it directly
    base_cmd += f" --min_samples {cluster_min_samples}"

    # Add boolean flags if True
    if cluster_use_gpu:
        base_cmd += " --gpu"
    if cluster_normalize:
        base_cmd += " --normalize_embeddings"
    if cluster_use_umap:
        base_cmd += " --use_umap"
        # Add UMAP parameters only if UMAP is enabled
        base_cmd += f" --umap_n_components {cluster_umap_components}"
        base_cmd += f" --umap_n_neighbors {cluster_umap_neighbors}"
        base_cmd += f" --umap_min_dist {cluster_umap_min_dist}"
        base_cmd += f" --umap_metric {cluster_umap_metric}"

    # Add DBSCAN metric (optional, could add argument if needed)
    # base_cmd += f" --dbscan_metric {cluster_dbscan_metric}"

    logging.info(f"Cluster component command: {base_cmd}")
    return base_cmd


def build_components(env,
                     data_prep_output_uri: str,
                     segment_output_uri: str,
                     cluster_output_uri: str,
                     classify_output_uri: str,
                     postprocess_output_uri: str,
                     classify_per_cluster: int,
                     # --- Add Clustering Parameters ---
                     cluster_eps: float | None,
                     cluster_min_samples: int,
                     cluster_use_gpu: bool,
                     cluster_normalize: bool,
                     cluster_use_umap: bool,
                     cluster_umap_components: int,
                     cluster_umap_neighbors: int,
                     cluster_umap_min_dist: float,
                     cluster_umap_metric: str):
    """
    Creates command components for data_prep, segment, cluster, classify, and post_process.
    """

    logging.info("Building pipeline components...")

    # 1) Data prep
    data_prep_component = command(
        name="DataPrep",
        display_name="Data Prep Step",
        inputs={
            "input_data": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={
            "output_path": Output(
                type=AssetTypes.URI_FOLDER,
                path=data_prep_output_uri
            )
        },
        code="./",
        command=(
            "python data_prep.py "
            "--input_data ${{inputs.input_data}} "
            "--output_path ${{outputs.output_path}} "
            "--tile_size 512 "
            "--overlap 0"
        ),
        environment=env,
    )

    # 2) Segment
    segment_component = command(
        name="Segmentation",
        display_name="Cell Segmentation",
        inputs={
            "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={
            "output_path": Output(
                type=AssetTypes.URI_FOLDER,
                path=segment_output_uri
            )
        },
        code="./",
        command=(
            "python segment.py "
            "--input_path ${{inputs.prepped_tiles_path}} "
            "--output_path ${{outputs.output_path}} "
            "--model_type cyto2 "
            "--chan 2 --chan2 1"
        ),
        environment=env,
    )

    # 3) Cluster (now DBSCAN) - Use the helper function for the command
    cluster_cmd_string = build_cluster_command(
        cluster_eps=cluster_eps,
        cluster_min_samples=cluster_min_samples,
        cluster_use_gpu=cluster_use_gpu,
        cluster_normalize=cluster_normalize,
        cluster_use_umap=cluster_use_umap,
        cluster_umap_components=cluster_umap_components,
        cluster_umap_neighbors=cluster_umap_neighbors,
        cluster_umap_min_dist=cluster_umap_min_dist,
        cluster_umap_metric=cluster_umap_metric,
    )
    cluster_component = command(
        name="Clustering",
        display_name="DBSCAN Clustering of Bounding Boxes",
        inputs={
            "segmentation_path": Input(type=AssetTypes.URI_FOLDER),
            "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={
            "cluster_output": Output(type=AssetTypes.URI_FOLDER, path=cluster_output_uri)
        },
        code="./",  # directory with your updated cluster.py
        command=cluster_cmd_string, # Use the dynamically built command
        environment=env,
    )

    # 4) Classify
    env_vars = {"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}
    classify_component = command(
        name="Classification",
        display_name="Classify Cells",
        inputs={
            "segmented_path": Input(type=AssetTypes.URI_FOLDER),
            "prepped_tiles_path": Input(type=AssetTypes.URI_FOLDER),
            "cluster_path": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={
            "output_path": Output(
                type=AssetTypes.URI_FOLDER,
                path=classify_output_uri
            )
        },
        code="./",
        command=(
            "python classify.py "
            "--segmented_path ${{inputs.segmented_path}} "
            "--prepped_tiles_path ${{inputs.prepped_tiles_path}} "
            "--cluster_output ${{inputs.cluster_path}} "
            "--output_path ${{outputs.output_path}} "
            "--num_classes 4 "
            f"--classify_per_cluster {classify_per_cluster}"
        ),
        environment=env,
        environment_variables=env_vars,
    )

    # 5) Post-process
    post_process_component = command(
        name="PostProcess",
        display_name="Post-Processing",
        inputs={
            "segmentation_path": Input(type=AssetTypes.URI_FOLDER),
            "classification_path": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={
            "output_path": Output(
                type=AssetTypes.URI_FOLDER,
                path=postprocess_output_uri
            )
        },
        code="./",
        command=(
            "python post_process.py "
            "--segmentation_path ${{inputs.segmentation_path}} "
            "--classification_path ${{inputs.classification_path}} "
            "--output_path ${{outputs.output_path}}"
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
    # --------------------------------------------------
    # 1. Parse CLI args
    # --------------------------------------------------
    parser = argparse.ArgumentParser(description="Run Histology Image Analysis Pipeline on Azure ML")

    # --- Pipeline Mode ---
    parser.add_argument(
        "--mode", type=str, choices=["prep_only", "full", "seg_cluster_cls", "cluster_cls", "classify_only"],
        default="full", help="Which part of the pipeline to run."
    )

    # --- Input Data URIs ---
    parser.add_argument("--raw_slides_uri", type=str, default="azureml://datastores/workspaceblobstore/paths/UI/2025-03-05_125751_UTC/", help="URI folder of raw slides (for modes: full, prep_only).")
    parser.add_argument("--prepped_data_uri", type=str, default="azureml://datastores/workspaceblobstore/paths/my_prepped_data/", help="URI folder of prepped tiles (for modes: seg_cluster_cls, cluster_cls, classify_only).")
    parser.add_argument("--segmented_data_uri", type=str, default="azureml://datastores/workspaceblobstore/paths/my_segmented_data/", help="URI folder of segmented tiles (for modes: cluster_cls, classify_only).")
    parser.add_argument("--clustered_data_uri", type=str, default="azureml://datastores/workspaceblobstore/paths/my_clustered_data/", help="URI folder of clustered results (for modes: classify_only).")

    # --- Classification Parameters ---
    parser.add_argument("--classify_per_cluster", type=int, default=10, help="Number of bounding boxes per cluster to classify.")

    # --- Clustering Parameters ---
    parser.add_argument("--cluster_eps", type=float, default=None, help="DBSCAN eps parameter. If None, cluster.py attempts auto-estimation.")
    parser.add_argument("--cluster_min_samples", type=int, default=5, help="DBSCAN min_samples parameter.")
    parser.add_argument("--cluster_use_gpu", action="store_true", help="Enable GPU usage within the clustering component (embedding and cuML DBSCAN if available).")
    parser.add_argument("--cluster_normalize", action="store_true", help="Enable L2 normalization of embeddings before clustering.")
    parser.add_argument("--cluster_use_umap", action="store_true", help="Enable UMAP dimensionality reduction before clustering.")
    parser.add_argument("--cluster_umap_components", type=int, default=50, help="Target dimensions for UMAP reduction.")
    parser.add_argument("--cluster_umap_neighbors", type=int, default=15, help="UMAP n_neighbors parameter.")
    parser.add_argument("--cluster_umap_min_dist", type=float, default=0.1, help="UMAP min_dist parameter.")
    parser.add_argument("--cluster_umap_metric", type=str, default="euclidean", help="Metric for UMAP ('euclidean', 'cosine', etc.).")

    args = parser.parse_args()
    logging.info("Parsed arguments: %s", args)

    # --- Log clustering parameters specifically ---
    logging.info("Clustering parameters for this run:")
    logging.info(f"  --cluster_eps: {'Auto' if args.cluster_eps is None else args.cluster_eps}")
    logging.info(f"  --cluster_min_samples: {args.cluster_min_samples}")
    logging.info(f"  --cluster_use_gpu: {args.cluster_use_gpu}")
    logging.info(f"  --cluster_normalize: {args.cluster_normalize}")
    logging.info(f"  --cluster_use_umap: {args.cluster_use_umap}")
    if args.cluster_use_umap:
        logging.info(f"  --cluster_umap_components: {args.cluster_umap_components}")
        logging.info(f"  --cluster_umap_neighbors: {args.cluster_umap_neighbors}")
        logging.info(f"  --cluster_umap_min_dist: {args.cluster_umap_min_dist}")
        logging.info(f"  --cluster_umap_metric: {args.cluster_umap_metric}")


    # --------------------------------------------------
    # 2. Generate a timestamp prefix for unique outputs
    # --------------------------------------------------
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logging.info("Timestamp for outputs: %s", timestamp)
    output_base_uri = f"azureml://datastores/workspaceblobstore/paths/{timestamp}_pipeline_outputs" # Define base path

    # Build unique paths for each step
    data_prep_output_uri = f"{output_base_uri}/data_prep/"
    segment_output_uri   = f"{output_base_uri}/segment/"
    cluster_output_uri   = f"{output_base_uri}/cluster/"
    classify_output_uri  = f"{output_base_uri}/classify/"
    postprocess_output_uri = f"{output_base_uri}/postprocess/"

    # --------------------------------------------------
    # 3. Connect to Azure ML workspace
    # --------------------------------------------------
    logging.info("Connecting to Azure ML Workspace...")
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME
        )
        logging.info("Connection to ML workspace succeeded.")
    except Exception as e:
        logging.error(f"Failed to connect to Azure ML workspace: {e}", exc_info=True)
        return # Cannot proceed without connection

    # --------------------------------------------------
    # 4. Create or update the environment
    # --------------------------------------------------
    env = Environment(
        name="edu06_env_revised", # Use the name from your YAML file
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"
    )
    logging.info("Creating/updating environment: %s", env.name)
    try:
        ml_client.environments.create_or_update(env)
        logging.info("Environment %s is ready.", env.name)
    except Exception as e:
        logging.error(f"Failed to create or update environment {env.name}: {e}", exc_info=True)
        return # Cannot proceed without environment


    # --------------------------------------------------
    # 5. Build the command components - PASS CLUSTER ARGS HERE
    # --------------------------------------------------
    try:
        components = build_components(
            env=env,
            data_prep_output_uri=data_prep_output_uri,
            segment_output_uri=segment_output_uri,
            cluster_output_uri=cluster_output_uri,
            classify_output_uri=classify_output_uri,
            postprocess_output_uri=postprocess_output_uri,
            classify_per_cluster=args.classify_per_cluster,
            # --- Pass clustering args from CLI ---
            cluster_eps=args.cluster_eps,
            cluster_min_samples=args.cluster_min_samples,
            cluster_use_gpu=args.cluster_use_gpu,
            cluster_normalize=args.cluster_normalize,
            cluster_use_umap=args.cluster_use_umap,
            cluster_umap_components=args.cluster_umap_components,
            cluster_umap_neighbors=args.cluster_umap_neighbors,
            cluster_umap_min_dist=args.cluster_umap_min_dist,
            cluster_umap_metric=args.cluster_umap_metric
        )
    except Exception as e:
        logging.error(f"Failed to build pipeline components: {e}", exc_info=True)
        return

    # --------------------------------------------------
    # 6. Define pipeline functions for each mode
    #    (These remain mostly the same, just call the components)
    # --------------------------------------------------

    @pipeline(compute=COMPUTE_CLUSTER, description="Full pipeline")
    def full_pipeline(raw_slides_input):
        prep_step = components["data_prep"](input_data=raw_slides_input)
        seg_step = components["segment"](prepped_tiles_path=prep_step.outputs.output_path)
        cluster_step = components["cluster"](
            segmentation_path=seg_step.outputs.output_path,
            prepped_tiles_path=prep_step.outputs.output_path)
        cls_step = components["classify"](
            segmented_path=seg_step.outputs.output_path,
            prepped_tiles_path=prep_step.outputs.output_path,
            cluster_path=cluster_step.outputs.cluster_output
        )
        post_step = components["post_process"](
            segmentation_path=seg_step.outputs.output_path,
            classification_path=cls_step.outputs.output_path
        )
        return {"final_output": post_step.outputs.output_path}

    @pipeline(compute=COMPUTE_CLUSTER, description="Data prep only")
    def data_prep_pipeline(raw_slides_input):
        prep_step = components["data_prep"](input_data=raw_slides_input)
        return {"prepped_data": prep_step.outputs.output_path}

    @pipeline(compute=COMPUTE_CLUSTER, description="Segment -> Cluster -> Classify -> Post-process")
    def seg_cluster_cls_pipeline(prepped_tiles_input):
        seg_step = components["segment"](prepped_tiles_path=prepped_tiles_input)
        cluster_step = components["cluster"](
            segmentation_path=seg_step.outputs.output_path,
            prepped_tiles_path=prepped_tiles_input)
        cls_step = components["classify"](
            segmented_path=seg_step.outputs.output_path,
            prepped_tiles_path=prepped_tiles_input,
            cluster_path=cluster_step.outputs.cluster_output
        )
        post_step = components["post_process"](
            segmentation_path=seg_step.outputs.output_path,
            classification_path=cls_step.outputs.output_path
        )
        return {"final_output": post_step.outputs.output_path}

    @pipeline(compute=COMPUTE_CLUSTER, description="Cluster -> Classify -> Post-process")
    def cluster_cls_pipeline(prepped_tiles_input, segmented_input):
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
    def classify_only_pipeline(prepped_tiles_input, segmented_input, cluster_input):
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

    # --------------------------------------------------
    # 7. Build & submit the chosen pipeline
    # --------------------------------------------------
    logging.info("Selected pipeline mode: %s", args.mode)
    pipeline_job = None
    experiment_name = f"histology_{args.mode}_{timestamp}" # Unique experiment name

    try:
        if args.mode == "prep_only":
            pipeline_job = data_prep_pipeline(
                raw_slides_input=Input(type=AssetTypes.URI_FOLDER, path=args.raw_slides_uri)
            )
        elif args.mode == "seg_cluster_cls":
            pipeline_job = seg_cluster_cls_pipeline(
                prepped_tiles_input=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri)
            )
        elif args.mode == "cluster_cls":
            pipeline_job = cluster_cls_pipeline(
                prepped_tiles_input=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri),
                segmented_input=Input(type=AssetTypes.URI_FOLDER, path=args.segmented_data_uri)
            )
        elif args.mode == "classify_only":
            pipeline_job = classify_only_pipeline(
                prepped_tiles_input=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri),
                segmented_input=Input(type=AssetTypes.URI_FOLDER, path=args.segmented_data_uri),
                cluster_input=Input(type=AssetTypes.URI_FOLDER, path=args.clustered_data_uri)
            )
        else:  # args.mode == "full"
            pipeline_job = full_pipeline(
                raw_slides_input=Input(type=AssetTypes.URI_FOLDER, path=args.raw_slides_uri)
            )

        if pipeline_job:
            logging.info("Submitting pipeline job to Azure ML...")
            submitted_job = ml_client.jobs.create_or_update(
                pipeline_job,
                experiment_name=experiment_name
            )
            logging.info("%s pipeline submitted! View here: %s", args.mode, submitted_job.studio_url)
            print(f"{args.mode} pipeline submitted! View here:", submitted_job.studio_url)
        else:
             logging.error("Pipeline job object was not created for mode: %s", args.mode)

    except Exception as e:
        logging.error(f"Failed to define or submit pipeline job for mode {args.mode}: {e}", exc_info=True)


if __name__ == "__main__":
    run_pipeline()