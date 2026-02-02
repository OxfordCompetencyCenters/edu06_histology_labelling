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
5. **Classification** (Multimodal LLMs) - Labels representative cells from each cluster (experimental)

The segmentation and clustering components demonstrate strong performance across many tissue types, while classification remains an area for future improvement.

## Pipeline Architecture

![Pipeline Architecture](azureml_pipeline/images/Pipeline%20design%20-%20white%20background.svg)

## Features

### The Power of Cluster-Based Visualization

When exploring a clustered slide, **every cell is color-coded by its cluster**. This transforms how you interact with histology images:

- **See structure at a glance**: Zoom into any region and immediately see which cells are morphologically similar—they share the same color
- **Discover patterns**: Clusters reveal spatial organization you might miss when viewing unlabeled images—cell type distributions, tissue boundaries, and regional variations become obvious
- **Efficient labeling**: Once you identify what a cluster represents, that label can be applied to every cell in that cluster across the slide

### Core Pipeline Components
- **Tiling**: WSI processing into 512×512 tiles with multi-magnification support (downsampling/upsampling)
- **Quality Filtering**: Removes background noise using edge density, brightness, color variance metrics
- **Segmentation**: Cellpose-SAM (default) for robust cell boundary detection
- **Neural Embedding**: ResNet-50 backbone converts cell crops to 2048-dimensional feature vectors
- **Dimensionality Reduction**: UMAP reduces embeddings to 50 dimensions while preserving morphological relationships
- **Clustering**: DBSCAN groups similar cells with GPU acceleration via RAPIDS cuML
- **Classification**: Multimodal LLM labels representative cells (experimental)
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
| `annotate_only` | Generate image annotations from existing results |
| `extract_cluster_tiles_only` | Extract representative tiles from existing results |
| `cluster_tiles_and_filtered_annotations` | Extract tiles and create filtered annotations |

> **Note**: Previous resume modes (`seg_cluster_cls`, `cluster_cls`, `classify_only`) have been removed as they required trigger/manifest files from older non-parallel scripts. Use `--mode full` for new runs.

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
pip install "azure-ai-ml>=1.12.0" "azure-identity>=1.14.0" "python-dotenv>=1.0.0"
```

| Package | Version | Purpose |
|---------|---------|----------|
| `azure-ai-ml` | >=1.12.0 | Azure ML SDK v2 for pipeline definition and job submission |
| `azure-identity` | >=1.14.0 | Azure authentication (DefaultAzureCredential) |
| `python-dotenv` | >=1.0.0 | Load environment variables from `.env` file |

> **Note**: These are separate from the pipeline component dependencies (defined in `azureml_pipeline/environment.yml`), which are installed in the Azure ML compute environment.

### API Keys
- OpenAI API key with multimodal-LLM access, **OR**
- Azure OpenAI deployment with multimodal-LLM

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

### Full Pipeline
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
| `--segment_pretrained_model` | `"cpsam"` | Cellpose pretrained model: `cpsam` (Cellpose-SAM), `cyto`, `cyto2`, `cyto3`, `nuclei`, `tissuenet`, `livecell`, `yeast_PhC`, `yeast_BF`, `bact_phase`, `bact_fluor`, `deepbact`, `cyto2_cp3`, `cyto2_omni` |
| `--segment_flow_threshold` | `0.4` | Flow threshold (lower = stricter shape filtering) |
| `--segment_cellprob_threshold` | `0.0` | Cell probability threshold (higher = stricter cell detection) |
| `--segment_use_gpu` | `True` | Enable GPU acceleration |
| `--segment_diameter` | `None` | Expected cell diameter in pixels (auto-estimated if None) |
| `--segment_resample` | `False` | Enable resampling for better segmentation of variable-sized objects |
| `--segment_normalize` | `True` | Normalize images before segmentation |
| `--segmentation_tile_batch_size` | `1` | Number of tiles to segment in a single batch (higher = faster on GPU, more VRAM) |

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
| `--cluster_per_slide` | `False` | Cluster each slide separately (always per-slide in parallel mode) |
| `--cluster_slide_folders` | `None` | Specific slide folder names to process |

### Classification (multimodal-LLM)
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
| `--annotation_no_text` | `True` | Do not draw text labels (default: no text) |
| `--annotation_text_use_pred_class` | `False` | Include predicted class in text labels |
| `--annotation_text_use_cluster_id` | `False` | Include cluster ID in text labels |
| `--annotation_text_use_cluster_confidence` | `False` | Include cluster confidence in text labels |
| `--annotation_text_scale` | `0.5` | Scale factor for text size |
| `--annotation_color_by` | `"cluster_id"` | Color-code by: `pred_class`, `cluster_id`, `none` |
| `--annotation_filter_unclassified` | `True` | Filter out unclassified cells (default: filtered) |

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
| `--filtered_annotation_filter_unclassified` | `True` | Filter out unclassified cells (default: filtered) |

### Multi-Node Parallelization
Control how the pipeline distributes work across compute nodes. The pipeline uses **slide-level parallelization** via Azure ML's `parallel_run_function` where each slide is processed independently:

| Stage | Parallel Entry Script | Description |
|-------|----------------------|-------------|
| Data prep | `parallel_data_prep.py` | Each WSI file is tiled independently |
| Tile filtering | `parallel_tile_filter.py` | Filters tiles per slide |
| Segmentation | `parallel_segment.py` | Cellpose processes each slide's tiles |
| Clustering | `parallel_cluster.py` | Per-slide UMAP + DBSCAN clustering |
| Classification | `parallel_classify.py` | Each slide classified independently |
| Post-processing | `post_process.py` | Aggregates per-slide results (single node) |
| Annotation | `annotate_images.py` | Per-slide output files (single node) |

When `--max_nodes > 1`:
- Multiple slides are processed in parallel across compute nodes
- Each node processes one slide at a time (mini-batch size = 1)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_nodes` | `1` | Maximum number of compute nodes (1 = single node, no parallelization) |
| `--processes_per_node` | `1` | Number of processes per node (set >1 for multi-GPU nodes) |
| `--mini_batch_size` | `1` | Number of slides per mini-batch (default: 1 slide per batch) |
| `--mini_batch_error_threshold` | `5` | Number of failed mini-batches allowed before failing the job |
| `--max_retries` | `3` | Max retries per mini-batch on failure/timeout (useful for low-priority VMs) |
| `--retry_timeout` | `300` | Timeout in seconds for each mini-batch retry |
| `--use_separate_clustering_cluster` | `False` | Use a separate compute cluster for clustering (e.g., high-RAM CPU cluster) |
| `--clustering_use_gpu` | `False` | Use GPU for clustering on the clustering cluster |

#### Example: Multi-Node Full Pipeline (10 nodes)
```bash
python azureml_pipeline/pipeline_job.py --mode full \
    --max_nodes 10 \
    --segment_use_gpu \
    --cluster_use_gpu \
    --raw_slides_uri "azureml://datastores/workspaceblobstore/paths/your_slides/"
```

To use a separate cluster for clustering, set `AZURE_ML_CLUSTERING_CLUSTER` in your `.env` file:
```bash
AZURE_ML_CLUSTERING_CLUSTER=your-highmem-cpu-cluster
```

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
- **Cost awareness**: Multimodal-LLM API calls incur costs; monitor usage with large datasets

### Current Limitations
- Classification (multimodal-LLM) performance is experimental and secondary to segmentation/clustering
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

## Third-Party Library Citations

This project relies on several open-source libraries. Please cite them if you use this work in your research.

### Cellpose

**Main Paper:**
Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods*, 18(1), 100-106.

```bibtex
@article{stringer2021cellpose,
  title={Cellpose: a generalist algorithm for cellular segmentation},
  author={Stringer, Carsen and Wang, Tim and Michaelos, Michalis and Pachitariu, Marius},
  journal={Nature Methods},
  volume={18},
  number={1},
  pages={100--106},
  year={2021},
  publisher={Nature Publishing Group}
}
```

**U-Net (cyto/cyto2 models):**
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. *arXiv preprint arXiv:1505.04597*.

```bibtex
@article{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  journal={arXiv preprint arXiv:1505.04597},
  year={2015}
}
```

**LIVECell (livecell model):**
Edlund, C., Jackson, T. L., Madej, T., Fuhrmann, J., & Rittscher, J. (2021). LIVECell—A large, high-quality, manually annotated dataset for label-free live cell segmentation. *Nature methods*, 18(9), 1038-1041.

```bibtex
@article{edlund2021livecell,
  title={LIVECell—A large, high-quality, manually annotated dataset for label-free live cell segmentation},
  author={Edlund, Christoffer and Jackson, T-Y Linus and Madej, Tomasz and Fuhrmann, Jiri and Rittscher, Jens},
  journal={Nature methods},
  volume={18},
  number={9},
  pages={1038--1041},
  year={2021},
  publisher={Nature Publishing Group}
}
```

### ResNet

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

### PyTorch

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in neural information processing systems* (pp. 8024-8035).

```bibtex
@inproceedings{paszke2019pytorch,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  booktitle={Advances in Neural Information Processing Systems 32},
  pages={8024--8035},
  year={2019},
  publisher={Curran Associates, Inc.}
}
```

### Scikit-learn

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of machine learning research*, 12(Oct), 2825-2830.

```bibtex
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
          and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
          and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
          Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
```

### OpenCV

Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.

```bibtex
@article{bradski2000opencv,
  title={The OpenCV Library},
  author={Bradski, G.},
  journal={Dr. Dobb's Journal of Software Tools},
  year={2000}
}
```

### RAPIDS (CuPy and cuML)

The RAPIDS Team. (2019). RAPIDS: Open GPU Data Science. In *Proceedings of the 2019 IEEE International Conference on Big Data (Big Data)* (pp. 5348-5349).

```bibtex
@inproceedings{rapids2019,
  author = {The RAPIDS Team},
  title = {{RAPIDS}: Open GPU Data Science},
  booktitle = {Proceedings of the 2019 IEEE International Conference on Big Data (Big Data)},
  year = {2019},
  pages = {5348-5349},
  doi = {10.1109/BigData47090.2019.9006348}
}
```

### Kneed

Arun Raj, K., & N, A. (2021). *arun-raj-ag/kneed: v0.8.1*. Zenodo. https://doi.org/10.5281/zenodo.5525221

```bibtex
@misc{arun_raj_2021_5525221,
  author       = {Arun Raj, Kevin and N, Aarshay},
  title        = {arun-raj-ag/kneed: v0.8.1},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.8.1},
  doi          = {10.5281/zenodo.5525221},
  url          = {https://doi.org/10.5281/zenodo.5525221}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Damion Young](https://www.medsci.ox.ac.uk/for-staff/staff/damion-young) and [Sharmila Rajendran](https://www.medsci.ox.ac.uk/for-staff/resources/educational-strategy-and-quality-assurance/teaching-excellence-awards/teaching-excellence-awards-2025) from [University of Oxford Medical Sciences Division](https://www.medsci.ox.ac.uk/) for ideating and formulating the research problem.
- [University of Oxford AI Competency Center](https://oerc.ox.ac.uk/ai-centre) for supporting this research
- [Cellpose](https://github.com/MouseLand/cellpose) for cell segmentation (including Cellpose-SAM)
- [OpenAI](https://openai.com/) for multimodal-LLM vision capabilities
- [RAPIDS](https://rapids.ai/) for GPU-accelerated clustering (cuML DBSCAN)
- [UMAP](https://umap-learn.readthedocs.io/) for dimensionality reduction
- [PyTorch/torchvision](https://pytorch.org/) for ResNet-50 feature extraction
