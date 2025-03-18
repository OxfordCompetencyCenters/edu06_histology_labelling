import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import Input, Output, command
from azure.ai.ml.entities import Environment
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes

# Your Azure ML details
SUBSCRIPTION_ID="bbeb7561-f822-4950-82f6-64dcae8a93ab"
RESOURCE_GROUP="AIMLCC-DEV-RG"
WORKSPACE_NAME="edu06_histology_img_segmentation"
COMPUTE_CLUSTER="edu06-compute-cluster"
DATA_ASSET_PATH="azureml://subscriptions/bbeb7561-f822-4950-82f6-64dcae8a93ab/resourcegroups/AIMLCC-DEV-RG/workspaces/edu06_histology_img_segmentation/datastores/workspaceblobstore/paths/UI/2025-03-05_125751_UTC/"


def run_pipeline():
    subscription_id = f"{SUBSCRIPTION_ID}"
    resource_group = f"{RESOURCE_GROUP}"
    workspace_name = f"{WORKSPACE_NAME}"
    compute_cluster = f"{COMPUTE_CLUSTER}"
    data_asset_path = f"{DATA_ASSET_PATH}"

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )

    # Environment from environment.yml
    env = Environment(
        name="edu06_env",
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    ml_client.environments.create_or_update(env)

    # Define pipeline components
    data_prep_component = command(
        name="DataPrep",
        display_name="Data Prep Step",
        inputs={
            "input_data": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={
            "output_path": Output(type=AssetTypes.URI_FOLDER)
        },
        code="./",
        command="python data_prep.py --input_data ${{inputs.input_data}} --output_path ${{outputs.output_path}}",
        environment=env
    )

    segment_component = command(
        name="Segmentation",
        display_name="Cell Segmentation",
        inputs={
            "input_path": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={
            "output_path": Output(type=AssetTypes.URI_FOLDER)
        },
        code="./",
        command="python segment.py --input_path ${{inputs.input_path}} --output_path ${{outputs.output_path}}",
        environment=env
    )

    classify_component = command(
        name="Classification",
        display_name="Classify Cells",
        inputs={
            "input_path": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={
            "output_path": Output(type=AssetTypes.URI_FOLDER)
        },
        code="./",
        command="python classify.py --input_path ${{inputs.input_path}} --output_path ${{outputs.output_path}}",
        environment=env
    )

    post_process_component = command(
        name="PostProcess",
        display_name="Post-Processing",
        inputs={
            "input_path": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={
            "output_path": Output(type=AssetTypes.URI_FOLDER)
        },
        code="./",
        command="python post_process.py --input_path ${{inputs.input_path}} --output_path ${{outputs.output_path}}",
        environment=env
    )

    @pipeline(
        compute=compute_cluster,
        description="Histology Labelling Pipeline"
    )
    def histology_pipeline(input_data):
        # Step 1: Data Prep
        step_data_prep = data_prep_component(input_data=input_data)
        # Step 2: Segmentation
        step_segment = segment_component(input_path=step_data_prep.outputs.output_path)
        # Step 3: Classification
        step_classify = classify_component(input_path=step_segment.outputs.output_path)
        # Step 4: Post-process
        step_postproc = post_process_component(input_path=step_classify.outputs.output_path)
        return {"final_output": step_postproc.outputs.output_path}

    # Instantiate and submit the pipeline
    pipeline_job = histology_pipeline(
        input_data=Input(type=AssetTypes.URI_FOLDER, path=data_asset_path)
    )
    submitted_job = ml_client.jobs.create_or_update(pipeline_job, experiment_name="histology_labelling_experiment")
    print("Pipeline submitted! View here:", submitted_job.studio_url)

if __name__ == "__main__":
    run_pipeline()