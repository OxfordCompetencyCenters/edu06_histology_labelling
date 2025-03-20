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

# --------------------------------------------------
# Setup logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --------------------------------------------------
# Constants for your workspace
# --------------------------------------------------
SUBSCRIPTION_ID = "bbeb7561-f822-4950-82f6-64dcae8a93ab"
RESOURCE_GROUP = "AIMLCC-DEV-RG"
WORKSPACE_NAME = "edu06_histology_img_segmentation"
COMPUTE_CLUSTER = "edu06-compute-cluster"

def build_components(env,
                     data_prep_output_uri: str,
                     segment_output_uri: str,
                     classify_output_uri: str,
                     postprocess_output_uri: str):
    """
    Creates command components for data_prep, segment, classify, and post_process.
    Each output is written to a manually specified path, here with a timestamp prefix.
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
            "input_path": Input(type=AssetTypes.URI_FOLDER)
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
            "--input_path ${{inputs.input_path}} "
            "--output_path ${{outputs.output_path}} "
            "--model_type cyto2 "
            "--chan 2 --chan2 1"
        ),
        environment=env,
    )

    # 3) Classify
    classify_component = command(
        name="Classification",
        display_name="Classify Cells",
        inputs={
            "input_path": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={
            "output_path": Output(
                type=AssetTypes.URI_FOLDER,
                path=classify_output_uri
            )
        },
        code="./",  # directory with classify.py
        command=(
            "python classify.py "
            "--input_path ${{inputs.input_path}} "
            "--output_path ${{outputs.output_path}} "
            "--num_classes 4"
        ),
        environment=env,
    )

    # 4) Post-process
    post_process_component = command(
        name="PostProcess",
        display_name="Post-Processing",
        inputs={
            "input_path": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={
            "output_path": Output(
                type=AssetTypes.URI_FOLDER,
                path=postprocess_output_uri
            )
        },
        code="./",  # directory with post_process.py
        command=(
            "python post_process.py "
            "--input_path ${{inputs.input_path}} "
            "--output_path ${{outputs.output_path}}"
        ),
        environment=env,
    )

    logging.info("Components built successfully.")
    return {
        "data_prep": data_prep_component,
        "segment": segment_component,
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
        choices=["prep_only", "rest_only", "full"],
        default="full",
        help="Which pipeline to run: prep_only, rest_only, or full."
    )
    # Input path to raw slides (when we do data prep)
    parser.add_argument(
        "--raw_slides_uri",
        type=str,
        default="azureml://datastores/workspaceblobstore/paths/UI/2025-03-05_125751_UTC/",
        help="URI folder of raw slides (for data prep)."
    )
    # Input path to prepped data (when we skip data prep)
    parser.add_argument(
        "--prepped_data_uri",
        type=str,
        default="azureml://datastores/workspaceblobstore/paths/my_prepped_data/",
        help="URI folder of already-prepped tiles."
    )
    args = parser.parse_args()

    logging.info("Parsed arguments: %s", args)

    # --------------------------------------------------
    # 2. Generate a timestamp prefix for unique outputs
    # --------------------------------------------------
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logging.info("Timestamp for outputs: %s", timestamp)

    # Build unique paths
    data_prep_output_uri = f"azureml://datastores/workspaceblobstore/paths/{timestamp}_pipeline_outputs/data_prep/"
    segment_output_uri   = f"azureml://datastores/workspaceblobstore/paths/{timestamp}_pipeline_outputs/segment/"
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
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
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
        classify_output_uri=classify_output_uri,
        postprocess_output_uri=postprocess_output_uri
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
        seg_step = components["segment"](input_path=prepped_tiles_input)
        cls_step = components["classify"](input_path=seg_step.outputs.output_path)
        post_step = components["post_process"](input_path=cls_step.outputs.output_path)
        return {"final_output": post_step.outputs.output_path}

    @pipeline(
        compute=COMPUTE_CLUSTER,
        description="Full pipeline: data prep -> segment -> classify -> post-process"
    )
    def full_pipeline(raw_slides_input):
        prep_step = components["data_prep"](input_data=raw_slides_input)
        seg_step = components["segment"](input_path=prep_step.outputs.output_path)
        cls_step = components["classify"](input_path=seg_step.outputs.output_path)
        post_step = components["post_process"](input_path=cls_step.outputs.output_path)
        return {"final_output": post_step.outputs.output_path}

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
