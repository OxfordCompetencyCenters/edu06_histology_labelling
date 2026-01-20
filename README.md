# edu06_histology_labelling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- [![DOI](https://zenodo.org/badge/DOI/YOUR_DOI_HERE.svg)](https://doi.org/YOUR_DOI_HERE) -->

Azure ML pipeline for automated histology image analysis combining deep learning segmentation, neural embedding, dimensionality reduction, and unsupervised clustering to identify and group cellular structures in whole slide images (WSI).

## The Problem

Labeling cells in histology slides is **extremely tedious**:
- A single whole slide image can contain **tens of thousands of cells**
- Traditional approaches require manually drawing boundaries around cells
- Each cell must then be individually classified

## Our Solution: Cluster-First Labeling

1. **Automatic Segmentation**: AI detects and draws boundaries for cells (no manual tracing)
2. **Intelligent Clustering**: Groups morphologically similar cells across the slide
3. **Label Once, Apply Everywhere**: Label a single cluster → the label can be propagated to ALL similar cells instantly

**Example**: If your slide has 15,000 cells grouped into 25 clusters, you only need to review and label 25 representative groups instead of 15,000 individual cells.

## Summary

This pipeline combines multiple AI techniques to identify cells and group them by type:

1. **Segmentation** (Cellpose/Cellpose-SAM) - Accurately identifies cellular structures and boundaries
2. **Neural Embedding** (ResNet-50) - Converts segmented cell images into 2048-dimensional feature vectors
3. **Dimensionality Reduction** (UMAP) - Reduces embeddings to 50 dimensions while preserving structure
4. **Clustering** (DBSCAN) - Groups morphologically similar cells across tiles/slides
5. **Classification** (GPT-4o) - Labels representative cells from each cluster (experimental)

The segmentation and clustering components demonstrate strong performance across many tissue types, while classification remains an area for future improvement.

## Pipeline Architecture

![Pipeline Architecture](azureml_pipeline/images/Pipeline%20design%20-%20white%20background.svg)

## Features

### The Power of Cluster-Based Visualization

When exploring a clustered slide, **every cell is color-coded by its cluster**. This transforms how you interact with histology images:

- **See structure at a glance**: Zoom into any region and immediately see which cells are morphologically similar—they share the same color
- **No cross-referencing needed**: You don't have to look up "what does a beta cell look like?" in a textbook, then scan the slide trying to spot one. Instead, all beta cells (or any cell type) are already visually grouped
- **Discover patterns**: Clusters reveal spatial organization you might miss when viewing unlabeled images—cell type distributions, tissue boundaries, and regional variations become obvious
- **Efficient labeling**: Once you identify what a cluster represents, that label can be applied to every cell in that cluster across the slide

### Core Pipeline Components
- **Tiling**: WSI processing into 512×512 tiles with multi-magnification support (downsampling/upsampling)
- **Quality Filtering**: Removes background noise using edge density, brightness, color variance metrics
- **Segmentation**: Cellpose-SAM (default) for robust cell boundary detection
- **Neural Embedding**: ResNet-50 backbone converts cell crops to 2048-dimensional feature vectors
- **Dimensionality Reduction**: UMAP reduces embeddings to 50 dimensions while preserving morphological relationships
- **Clustering**: DBSCAN groups similar cells with GPU acceleration via RAPIDS cuML
- **Classification**: GPT-4o vision model labels representative cells (experimental)
- **Post-processing**: Aggregates results into structured JSON for downstream tools (e.g., CSlide)

### Annotation & Analysis
- **Image Annotation**: Customizable visualization with polygons, bounding boxes, and color-coded cluster labels
- **Representative Tile Extraction**: Selects high-confidence examples from each cluster
- **Filtered Annotations**: Generates focused annotations covering all distinct cell structures

### Supported Pipeline Modes
| Mode | Description |
|------|-------------|
| `full` | Complete pipeline: prep → filter → segment → cluster → classify → post-process |
| `prep_only` | Data preparation and optional tile filtering only |
| `seg_cluster_cls` | Segmentation → Clustering → Classification (from prepped data) |
| `cluster_cls` | Clustering → Classification (from segmented data) |
| `classify_only` | Classification only (from clustered data) |
| `annotate_only` | Generate image annotations from existing results |
| `extract_cluster_tiles_only` | Extract representative tiles from existing results |
| `cluster_tiles_and_filtered_annotations` | Extract tiles and create filtered annotations |

---

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with CUDA 11.8+ support (required for GPU acceleration)
- **Azure ML**: GPU compute cluster (e.g., `Standard_NC6s_v3`)

### Azure Requirements
- Azure subscription with Azure ML workspace
- Sufficient quota for GPU compute
- Azure Key Vault (optional, for secure credential storage)

### Local Dependencies (for submitting pipeline jobs)
The following Python packages are required on your local machine to run `pipeline_job.py` and submit jobs to Azure ML:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install azure-ai-ml azure-identity python-dotenv
```

| Package | Purpose |
|---------|----------|
| `azure-ai-ml` | Azure ML SDK v2 for pipeline definition and job submission |
| `azure-identity` | Azure authentication (DefaultAzureCredential) |
| `python-dotenv` | Load environment variables from `.env` file |

> **Note**: These are separate from the pipeline component dependencies (defined in `azureml_pipeline/environment.yml`), which are installed in the Azure ML compute environment.

### API Keys
- OpenAI API key with GPT-4o access, **OR**
- Azure OpenAI deployment with GPT-4o

---

## Azure ML Setup

### 1. Clone the Repository

```bash
git clone https://github.com/OxfordCompetencyCenters/edu06_histology_labelling.git
cd edu06_histology_labelling
```

### 2. Configure Azure ML Workspace

Duplicate `.env.example` file and rename it to `.env`. Update the workspace configuration in `.env`.

### 3. Upload Data to Azure Blob Storage

Upload your WSI files to your Azure ML workspace's blob storage datastore, then reference the URI in pipeline arguments.

---

## Running the Pipeline

### Azure authentication (required to submit jobs)
This script submits jobs to **Azure Machine Learning**, so you must be authenticated to Azure and have access to the target workspace.

Also ensure your `.env` contains the required values checked by the script:

- `AZURE_SUBSCRIPTION_ID`
- `AZURE_RESOURCE_GROUP`
- `AZURE_ML_WORKSPACE_NAME`
- `OPENAI_API_KEY`

### Full Pipeline (recommended)
```bash
python azureml_pipeline/pipeline_job.py --mode full \
    --segment_use_gpu \
    --cluster_use_gpu \
    --cluster_normalize \
    --cluster_use_umap \
    --raw_slides_uri "azureml://datastores/workspaceblobstore/paths/your_slides/"
```

### With Tile Filtering
```bash
python azureml_pipeline/pipeline_job.py --mode full \
    --filter_tiles \
    --filter_min_edge_density 0.02 \
    --segment_use_gpu \
    --cluster_use_gpu
```

### With Annotations and Representative Tile Extraction
```bash
python azureml_pipeline/pipeline_job.py --mode full \
    --segment_use_gpu \
    --cluster_use_gpu \
    --enable_annotations \
    --enable_cluster_tiles \
    --cluster_analyzer_confidence_threshold 0.75 \
    --cluster_analyzer_max_items 20
```

### From Existing Prepped Data (skip tiling)
```bash
python azureml_pipeline/pipeline_job.py --mode seg_cluster_cls \
    --segment_use_gpu \
    --cluster_use_gpu \
    --prepped_data_uri "azureml://datastores/workspaceblobstore/paths/your_tiles/"
```

### Generate Annotations from Existing Results
```bash
python azureml_pipeline/pipeline_job.py --mode annotate_only \
    --enable_annotations \
    --postprocess_data_uri "azureml://datastores/workspaceblobstore/paths/your_results/" \
    --prepped_data_uri "azureml://datastores/workspaceblobstore/paths/your_tiles/"
```

### Extract Representative Cluster Tiles
```bash
python azureml_pipeline/pipeline_job.py --mode extract_cluster_tiles_only \
    --enable_cluster_tiles \
    --cluster_analyzer_confidence_threshold 0.75 \
    --postprocess_data_uri "azureml://datastores/workspaceblobstore/paths/your_results/" \
    --prepped_data_uri "azureml://datastores/workspaceblobstore/paths/your_tiles/"
```

---

## Pipeline Parameters

### Input/Output URIs
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--raw_slides_uri` | `"azureml://..."` | URI to raw WSI slides (for `full` and `prep_only` modes) |
| `--prepped_data_uri` | `"azureml://..."` | URI to prepped tiles (for `seg_cluster_cls`, `cluster_cls`, `classify_only`, `annotate_only`, etc.) |
| `--segmented_data_uri` | `"azureml://..."` | URI to segmented data (for `cluster_cls` and `classify_only` modes) |
| `--clustered_data_uri` | `"azureml://..."` | URI to clustered data (for `classify_only` mode) |
| `--postprocess_data_uri` | `"azureml://..."` | URI to post-processed results (for `annotate_only`, `extract_cluster_tiles_only`, etc.) |

### Data Preparation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--magnifications` | `"1.0"` | Comma-separated magnification levels (e.g., `"1.0,0.9,0.8"` for downsampling) |
| `--num_tiles` | `None` | Approximate number of tiles per magnification (uniform grid sampling) |

### Tile Filtering
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--filter_tiles` | `False` | Enable tile quality filtering |
| `--filter_min_edge_density` | `0.02` | Minimum edge density (structure content) |
| `--filter_max_bright_ratio` | `0.8` | Maximum ratio of bright pixels (background) |
| `--filter_max_dark_ratio` | `0.8` | Maximum ratio of dark pixels (empty space) |
| `--filter_min_std_intensity` | `10.0` | Minimum intensity standard deviation |
| `--filter_min_laplacian_var` | `50.0` | Minimum Laplacian variance (sharpness) |
| `--filter_min_color_variance` | `5.0` | Minimum color variance across channels |

### Segmentation (Cellpose)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--segment_model_type` | `"cellpose_sam"` | Model: `cyto`, `cyto2`, `cyto3`, `nuclei`, `cellpose_sam`, etc. |
| `--segment_flow_threshold` | `0.4` | Flow threshold (lower = stricter shape filtering) |
| `--segment_cellprob_threshold` | `0.0` | Cell probability threshold (higher = stricter cell detection) |
| `--segment_use_gpu` | `True` | Enable GPU acceleration |
| `--segment_diameter` | `None` | Expected cell diameter in pixels (auto-estimated if None) |
| `--segment_resample` | `False` | Enable resampling for better segmentation of variable-sized objects |
| `--segment_normalize` | `True` | Normalize images before segmentation |
| `--segment_do_3D` | `False` | Enable 3D segmentation (for Z-stacks) |
| `--segment_stitch_threshold` | `0.0` | Threshold for stitching masks across tiles (0.0 = no stitching) |
| `--segment_channels` | `"2,1"` | Channel specification: cytoplasm,nucleus |
| `--segment_use_cellpose_sam` | `False` | Use Cellpose-SAM for enhanced generalization |

### Clustering (ResNet-50 → UMAP → DBSCAN)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cluster_eps` | `None` | DBSCAN epsilon/neighborhood radius (auto-computed if None) |
| `--cluster_min_samples` | `5` | Minimum samples for DBSCAN core point |
| `--cluster_use_gpu` | `True` | Enable GPU acceleration via RAPIDS cuML |
| `--cluster_normalize` | `False` | Normalize ResNet-50 embeddings before clustering |
| `--cluster_use_umap` | `False` | Enable UMAP dimensionality reduction (2048→50) |
| `--cluster_umap_components` | `50` | Number of UMAP output dimensions |
| `--cluster_umap_neighbors` | `15` | UMAP n_neighbors parameter |
| `--cluster_umap_min_dist` | `0.1` | UMAP min_dist parameter |
| `--cluster_umap_metric` | `"euclidean"` | Distance metric for UMAP |
| `--cluster_per_slide` | `False` | Cluster each slide separately vs. globally |
| `--cluster_slide_folders` | `None` | Specific slide folder names to process (when using `--cluster_per_slide`) |

### Classification (GPT-4o)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--classify_per_cluster` | `10` | Number of cells to classify per cluster (set to 0 to skip classification) |

### Annotation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_annotations` | `False` | Enable image annotation generation |
| `--annotation_max_labels` | `100` | Maximum labels per image |
| `--annotation_random_labels` | `False` | Pick labels randomly up to max_labels |
| `--annotation_draw_polygon` | `True` | Draw cell boundary polygons |
| `--annotation_draw_bbox` | `False` | Draw bounding boxes |
| `--annotation_no_text` | `True` | Do not draw text labels |
| `--annotation_text_use_pred_class` | `False` | Include predicted class in text labels |
| `--annotation_text_use_cluster_id` | `False` | Include cluster ID in text labels |
| `--annotation_text_use_cluster_confidence` | `False` | Include cluster confidence in text labels |
| `--annotation_text_scale` | `0.5` | Scale factor for text size |
| `--annotation_color_by` | `"cluster_id"` | Color-code by: `pred_class`, `cluster_id`, `none` |
| `--annotation_filter_unclassified` | `True` | Filter out unclassified cells |

### Representative Tile Extraction
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_cluster_tiles` | `False` | Enable representative tile extraction |
| `--cluster_analyzer_confidence_threshold` | `0.75` | Minimum cluster confidence for extraction |
| `--cluster_analyzer_max_items` | `20` | Maximum representative tiles per cluster |

### Filtered Annotation
Parameters for annotating the representative tiles extracted by the cluster analyzer.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_filtered_annotations` | `False` | Enable annotation of filtered cluster tiles |
| `--filtered_annotation_max_labels` | `100` | Maximum labels per filtered image |
| `--filtered_annotation_random_labels` | `False` | Pick labels randomly up to max_labels |
| `--filtered_annotation_draw_polygon` | `True` | Draw cell boundary polygons |
| `--filtered_annotation_draw_bbox` | `False` | Draw bounding boxes |
| `--filtered_annotation_no_text` | `False` | Do not draw text labels |
| `--filtered_annotation_text_use_pred_class` | `False` | Include predicted class in text labels |
| `--filtered_annotation_text_use_cluster_id` | `False` | Include cluster ID in text labels |
| `--filtered_annotation_text_use_cluster_confidence` | `False` | Include cluster confidence in text labels |
| `--filtered_annotation_text_scale` | `0.5` | Scale factor for text size |
| `--filtered_annotation_color_by` | `"cluster_id"` | Color-code by: `pred_class`, `cluster_id`, `none` |
| `--filtered_annotation_filter_unclassified` | `True` | Filter out unclassified cells |

---

## Performance Observations

### Strong Performance
- **Skeletal muscle**: Excellent segmentation of multinucleated fibers with consistent cluster assignment
- **Pancreas (exocrine)**: High accuracy in detecting acinar cells with clear boundaries
- **Adrenal cortex**: Good performance in zona glomerulosa and fasciculata regions

### Moderate Performance
- **Thyroid gland**: Partial success with follicular cells; struggles with flattened epithelium

### Challenging Cases
- **Posterior pituitary**: Poor performance due to unmyelinated nerve fibers and sparse glial cells
- **Densely packed tissues**: May require tissue-specific parameter tuning

---

## Data

**No data included**: This repository contains code only. Due to data governance requirements, histology images are not included.

To use this pipeline:
1. Provide your own WSI (Whole Slide Image) files in supported format (`.ndpi`)
2. Upload slides to your Azure ML workspace blob storage
3. Reference the data URI in pipeline arguments

---

## Responsible Use & Limitations

### Note

- **Not for clinical use**: This software is for research and educational purposes only and has not been validated for clinical diagnosis
- **Human review required**: Model outputs should always be reviewed by qualified professionals
- **Variable performance**: Segmentation accuracy varies by tissue type and morphology
- **Cost awareness**: GPT-4o API calls incur costs; monitor usage with large datasets

### Current Limitations
- Classification (GPT-4o) performance is experimental and secondary to segmentation/clustering
- Single model configuration may not work optimally for all tissue types
- Lack of ground-truth polygon coordinates limits quantitative evaluation

---

## Citation

If you use this software in your research, please cite it:

```bibtex
@software{edu06_histology_labelling,
  author = {Muhammad Haseeb Ahmad},
  title = {edu06_histology_labelling: Azure ML pipeline for automated histology image analysis},
  year = {2026},
  url = {https://github.com/OxfordCompetencyCenters/edu06_histology_labelling},
  license = {MIT}
}
```

See [CITATION.cff](CITATION.cff) for more citation formats.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Oxford AI Competency Center](https://oerc.ox.ac.uk/ai-centre) for supporting this research
- [Cellpose](https://github.com/MouseLand/cellpose) for cell segmentation (including Cellpose-SAM)
- [OpenAI](https://openai.com/) for GPT-4o vision capabilities
- [RAPIDS](https://rapids.ai/) for GPU-accelerated clustering (cuML DBSCAN)
- [UMAP](https://umap-learn.readthedocs.io/) for dimensionality reduction
- [PyTorch/torchvision](https://pytorch.org/) for ResNet-50 feature extraction
