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

SUBSCRIPTION_ID = "bbeb7561-f822-4950-82f6-64dcae8a93ab"
RESOURCE_GROUP = "AIMLCC-DEV-RG"
WORKSPACE_NAME = "edu06_histology_img_segmentation"
COMPUTE_CLUSTER = "edu06-compute-cluster"

def build_components(env,
                     data_prep_output_uri: str,
                     segment_output_uri: str,
                     cluster_output_uri: str,
                     classify_output_uri: str,
                     postprocess_output_uri: str,
                     classify_per_cluster: int):
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
        code="./",  # directory with data_prep.py
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
        code="./",  # directory with segment.py
        command=(
            "python segment.py "
            "--input_path ${{inputs.prepped_tiles_path}} "
            "--output_path ${{outputs.output_path}} "
            "--model_type cyto2 "
            "--chan 2 --chan2 1"
        ),
        environment=env,
    )

    # 3) Cluster (now DBSCAN)
    cluster_component = command(
        name="Clustering",
        display_name="DBSCAN Clustering of Bounding Boxes",
        inputs={
            "segmentation_path": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={
            "cluster_output": Output(type=AssetTypes.URI_FOLDER, path=cluster_output_uri)
        },
        code="./",  # directory with your updated cluster.py
        command=(
            "python cluster.py "
            "--segmentation_path ${{inputs.segmentation_path}} "
            "--output_path ${{outputs.cluster_output}} "
            "--eps 0.5 "
            "--min_samples 5 "
            "--gpu"
        ),
        environment=env,
    )

    # 4) Classify
    #
    # Pipe the classify_per_cluster argument through to classify.py
    #
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "prep_only",
            "rest_only",
            "full",
            "classify_only",
            "rest_cluster"
        ],
        default="full",
        help=("Which pipeline to run: "
              "prep_only, rest_only, classify_only, full, or rest_cluster.")
    )
    parser.add_argument(
        "--raw_slides_uri",
        type=str,
        default="azureml://datastores/workspaceblobstore/paths/UI/2025-03-05_125751_UTC/",
        help="URI folder of raw slides (for data prep)."
    )
    parser.add_argument(
        "--prepped_data_uri",
        type=str,
        default="azureml://datastores/workspaceblobstore/paths/my_prepped_data/",
        help="URI folder of already-prepped tiles."
    )
    parser.add_argument(
        "--segmented_data_uri",
        type=str,
        default="azureml://datastores/workspaceblobstore/paths/my_segmented_data/",
        help="URI folder of already-segmented tiles (for classify_only mode)."
    )
    parser.add_argument(
        "--classify_per_cluster",
        type=int,
        default=10,
        help="Number of bounding boxes per cluster to classify."
    )

    args = parser.parse_args()
    logging.info("Parsed arguments: %s", args)

    # --------------------------------------------------
    # 2. Generate a timestamp prefix for unique outputs
    # --------------------------------------------------
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logging.info("Timestamp for outputs: %s", timestamp)

    # Build unique paths for each step
    data_prep_output_uri = f"azureml://datastores/workspaceblobstore/paths/{timestamp}_pipeline_outputs/data_prep/"
    segment_output_uri   = f"azureml://datastores/workspaceblobstore/paths/{timestamp}_pipeline_outputs/segment/"
    cluster_output_uri   = f"azureml://datastores/workspaceblobstore/paths/{timestamp}_pipeline_outputs/cluster/"
    classify_output_uri  = f"azureml://datastores/workspaceblobstore/paths/{timestamp}_pipeline_outputs/classify/"
    postprocess_output_uri = f"azureml://datastores/workspaceblobstore/paths/{timestamp}_pipeline_outputs/postprocess/"

    # --------------------------------------------------
    # 3. Connect to Azure ML workspace
    # --------------------------------------------------
    logging.info("Connecting to Azure ML Workspace...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
    logging.info("Connection to ML workspace succeeded.")

    # --------------------------------------------------
    # 4. Create or update the environment
    # --------------------------------------------------
    env = Environment(
        name="edu06_env",
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"
    )
    logging.info("Creating/updating environment: %s", env.name)
    ml_client.environments.create_or_update(env)
    logging.info("Environment %s is ready.", env.name)

    # --------------------------------------------------
    # 5. Build the command components
    # --------------------------------------------------
    components = build_components(
        env=env,
        data_prep_output_uri=data_prep_output_uri,
        segment_output_uri=segment_output_uri,
        cluster_output_uri=cluster_output_uri,
        classify_output_uri=classify_output_uri,
        postprocess_output_uri=postprocess_output_uri,
        classify_per_cluster=args.classify_per_cluster
    )

    # --------------------------------------------------
    # 6. Define pipeline functions for each mode
    # --------------------------------------------------
    @pipeline(
        compute=COMPUTE_CLUSTER,
        description="Pipeline for only the data prep step"
    )
    def data_prep_pipeline(raw_slides_input):
        prep_step = components["data_prep"](input_data=raw_slides_input)
        return {"prepped_data": prep_step.outputs.output_path}

    @pipeline(
        compute=COMPUTE_CLUSTER,
        description="Pipeline for segmentation + classification + postprocess (requires prepped data)."
    )
    def rest_pipeline(prepped_tiles_input):
        seg_step = components["segment"](prepped_tiles_path=prepped_tiles_input)
        cls_step = components["classify"](
            segmented_path=seg_step.outputs.output_path,
            prepped_tiles_path=prepped_tiles_input,
            cluster_path=""  # Not using clustering here
        )
        post_step = components["post_process"](
            segmentation_path=seg_step.outputs.output_path,
            classification_path=cls_step.outputs.output_path
        )
        return {"final_output": post_step.outputs.output_path}

    @pipeline(
        compute=COMPUTE_CLUSTER,
        description="Pipeline for classification + postprocess (requires prepped + segmented data)."
    )
    def classify_only_pipeline(prepped_tiles_input, segmented_input):
        cls_step = components["classify"](
            segmented_path=segmented_input,
            prepped_tiles_path=prepped_tiles_input,
            cluster_path=""  # Not using clustering
        )
        post_step = components["post_process"](
            segmentation_path=segmented_input,
            classification_path=cls_step.outputs.output_path
        )
        return {"final_output": post_step.outputs.output_path}

    @pipeline(
        compute=COMPUTE_CLUSTER,
        description="Full pipeline: data prep -> segment -> classify -> post-process"
    )
    def full_pipeline(raw_slides_input):
        prep_step = components["data_prep"](input_data=raw_slides_input)
        seg_step = components["segment"](prepped_tiles_path=prep_step.outputs.output_path)
        cls_step = components["classify"](
            segmented_path=seg_step.outputs.output_path,
            prepped_tiles_path=prep_step.outputs.output_path,
            cluster_path=""  # Not using cluster in the legacy 'full' pipeline
        )
        post_step = components["post_process"](
            segmentation_path=seg_step.outputs.output_path,
            classification_path=cls_step.outputs.output_path
        )
        return {"final_output": post_step.outputs.output_path}

    @pipeline(
        compute=COMPUTE_CLUSTER,
        description="Pipeline for segmentation -> DBSCAN clustering -> classification -> post-process"
    )
    def seg_cluster_cls_pipeline(prepped_tiles_input):
        seg_step = components["segment"](prepped_tiles_path=prepped_tiles_input)
        cluster_step = components["cluster"](segmentation_path=seg_step.outputs.output_path)
        cls_step = components["classify"](
            segmented_path=seg_step.outputs.output_path,
            prepped_tiles_path=prepped_tiles_input,
            cluster_path=cluster_step.outputs.cluster_output
        )
        post_step = components["post_process"](
            segmentation_path=seg_step.outputs.output_path,
            classification_path=cls_step.outputs.output_path
        )
        return {
            "final_output": post_step.outputs.output_path
        }

    # --------------------------------------------------
    # 7. Build & submit the chosen pipeline
    # --------------------------------------------------
    logging.info("Selected pipeline mode: %s", args.mode)
    if args.mode == "prep_only":
        pipeline_job = data_prep_pipeline(
            raw_slides_input=Input(type=AssetTypes.URI_FOLDER, path=args.raw_slides_uri)
        )
        experiment_name = "histology_data_prep_only"

    elif args.mode == "rest_only":
        pipeline_job = rest_pipeline(
            prepped_tiles_input=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri)
        )
        experiment_name = "histology_rest_only"

    elif args.mode == "classify_only":
        pipeline_job = classify_only_pipeline(
            prepped_tiles_input=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri),
            segmented_input=Input(type=AssetTypes.URI_FOLDER, path=args.segmented_data_uri)
        )
        experiment_name = "histology_classify_only"

    elif args.mode == "rest_cluster":
        pipeline_job = seg_cluster_cls_pipeline(
            prepped_tiles_input=Input(type=AssetTypes.URI_FOLDER, path=args.prepped_data_uri)
        )
        experiment_name = "histology_rest_cluster"

    else:  # args.mode == "full"
        pipeline_job = full_pipeline(
            raw_slides_input=Input(type=AssetTypes.URI_FOLDER, path=args.raw_slides_uri)
        )
        experiment_name = "histology_full"

    logging.info("Submitting pipeline job to Azure ML...")
    submitted_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name=experiment_name
    )
    logging.info("%s pipeline submitted! View here: %s", args.mode, submitted_job.studio_url)
    print(f"{args.mode} pipeline submitted! View here:", submitted_job.studio_url)


if __name__ == "__main__":
    run_pipeline()
