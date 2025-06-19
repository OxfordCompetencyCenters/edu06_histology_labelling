# edu06_histology_labelling

Azure ML pipeline for automated histology image analysis with cell segmentation, clustering, and classification.

## Features

### v1 - Cellpose + GPT-4o
- **Data Preparation**: WSI tiling with multi-magnification support
- **🆕 Tile Filtering**: Smart background noise removal before segmentation
- **Cell Segmentation**: Cellpose-based cell boundary detection
- **Feature Clustering**: DBSCAN clustering with optional UMAP dimensionality reduction
- **Cell Classification**: GPT-4o vision model for cell type identification
- **Post-processing**: Results aggregation and visualization

### v2 - SAM-Med + GPT-4o
- SAM-Med segmentation with token-based clustering
- Advanced embedding techniques for improved clustering

## Quick Start

### Run with tile filtering (recommended):
```bash
python v1\ -\ cellpose\ +\ gpt-4o/pipeline_job.py --filter_tiles --mode full
```

### Traditional pipeline (no filtering):
```bash
python v1\ -\ cellpose\ +\ gpt-4o/pipeline_job.py --mode full
```

## Tile Filtering Benefits

The new tile filtering component (see `v1 - cellpose + gpt-4o/TILE_FILTERING.md`) provides:
- **30-50% reduction** in processing time by filtering background noise
- **Improved segmentation quality** by focusing on meaningful tissue regions
- **Configurable strictness** with detailed filtering statistics
- **Transparent decision making** with per-tile quality metrics

Enable with `--filter_tiles` flag and customize with filtering parameters.