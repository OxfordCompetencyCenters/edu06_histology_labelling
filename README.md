# edu06_histology_labelling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- [![DOI](https://zenodo.org/badge/DOI/YOUR_DOI_HERE.svg)](https://doi.org/YOUR_DOI_HERE) -->

Azure ML pipeline for automated histology image analysis with cell segmentation, clustering, and classification.

## Features

### v1 - Cellpose + GPT-4o
- **Data Preparation**: WSI tiling with multi-magnification support
- **🆕 Tile Filtering**: Smart background noise removal before segmentation
- **Cell Segmentation**: Cellpose-based cell boundary detection
- **Feature Clustering**: DBSCAN clustering with optional UMAP dimensionality reduction
- **Cell Classification**: GPT-4o vision model for cell type identification
- **Post-processing**: Results aggregation and visualization

### v2 - SAM-Med + GPT-4o (Coming Soon)
- SAM-Med segmentation with token-based clustering
- Advanced embedding techniques for improved clustering

---

## Prerequisites

### System Requirements
- **OS**: Linux (recommended), Windows, or macOS
- **GPU**: NVIDIA GPU with CUDA 11.8+ support (required for GPU acceleration)
- **RAM**: 16GB+ recommended
- **Storage**: Varies based on WSI dataset size

### Azure Requirements
- Azure subscription with Azure ML workspace
- Sufficient quota for GPU compute (e.g., `Standard_NC6s_v3`)
- Azure Key Vault (optional, for secure credential storage)

### API Keys
- OpenAI API key with GPT-4o access, **OR**
- Azure OpenAI deployment with GPT-4o

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/OxfordCompetencyCenters/edu06_histology_labelling.git
cd edu06_histology_labelling
```

### 2. Create Conda Environment

```bash
# Create environment from file
conda env create -f v1_cellpose_gpt4o/environment.yml

# Activate environment
conda activate edu06_env
```

### 3. Configure Credentials

```bash
# Copy example configuration files
cp .env.example .env
cp config.example.yaml config.yaml

# Edit .env with your credentials
# IMPORTANT: Never commit .env or config.yaml to version control
```

Required environment variables in `.env`:
- `AZURE_SUBSCRIPTION_ID` - Your Azure subscription
- `AZURE_RESOURCE_GROUP` - Resource group name
- `AZURE_ML_WORKSPACE_NAME` - Azure ML workspace name
- `OPENAI_API_KEY` - Your OpenAI API key (or Azure OpenAI credentials)

---

## Quick Start

### Run with tile filtering (recommended):
```bash
python v1_cellpose_gpt4o/pipeline_job.py --filter_tiles --mode full
```

### Traditional pipeline (no filtering):
```bash
python v1_cellpose_gpt4o/pipeline_job.py --mode full
```

### Run locally (without Azure ML):
```bash
# Individual pipeline stages
python v1_cellpose_gpt4o/data_prep.py --input /path/to/wsi --output /path/to/tiles
python v1_cellpose_gpt4o/tile_filter.py --input /path/to/tiles --output /path/to/filtered
python v1_cellpose_gpt4o/segment.py --input /path/to/filtered --output /path/to/masks
python v1_cellpose_gpt4o/cluster.py --input /path/to/masks --output /path/to/clusters
python v1_cellpose_gpt4o/classify.py --input /path/to/clusters --output /path/to/results
```

---

## Tile Filtering Benefits

The tile filtering component (see `v1_cellpose_gpt4o/TILE_FILTERING.md`) provides:
- **30-50% reduction** in processing time by filtering background noise
- **Improved segmentation quality** by focusing on meaningful tissue regions
- **Configurable strictness** with detailed filtering statistics
- **Transparent decision making** with per-tile quality metrics

Enable with `--filter_tiles` flag and customize with filtering parameters.

---

## Data

**⚠️ No data included**: This repository contains code only. Due to data governance requirements, histology images are not included.

To use this pipeline:
1. Provide your own WSI (Whole Slide Image) files in supported formats (`.svs`, `.tif`, `.ndpi`, etc.)
2. Update paths in `config.yaml` or via command-line arguments
3. Ensure you have appropriate permissions to use the images

---

## Responsible Use & Limitations

### ⚠️ Important Disclaimers

- **Not for clinical use**: This software is for research purposes only and has not been validated for clinical diagnosis
- **Human review required**: Model outputs should always be reviewed by qualified professionals
- **Errors expected**: Both segmentation and classification models may produce incorrect results
- **Cost awareness**: GPT-4o API calls incur costs; monitor usage especially with large datasets

### Intended Use Cases
- Research and educational purposes
- Workflow automation prototyping
- Comparative analysis of segmentation methods

---

## Project Structure

```
edu06_histology_labelling/
├── v1_cellpose_gpt4o/       # Main pipeline (Cellpose + GPT-4o)
│   ├── pipeline_job.py      # Azure ML pipeline orchestration
│   ├── data_prep.py         # WSI tiling
│   ├── tile_filter.py       # Background filtering
│   ├── segment.py           # Cellpose segmentation
│   ├── cluster.py           # DBSCAN/UMAP clustering
│   ├── classify.py          # GPT-4o classification
│   ├── post_process.py      # Results aggregation
│   ├── annotate_images.py   # Visualization
│   └── environment.yml      # Conda environment
├── .env.example             # Environment variables template
├── config.example.yaml      # Configuration template
├── LICENSE                  # Apache 2.0 license
├── CITATION.cff             # Citation information
├── CHANGELOG.md             # Version history
├── SECURITY.md              # Security policy
└── README.md                # This file
```

---

## Citation

If you use this software in your research, please cite it:

```bibtex
@software{edu06_histology_labelling,
  author = {YOUR_NAME},
  title = {edu06_histology_labelling: Azure ML pipeline for automated histology image analysis},
  year = {2026},
  url = {https://github.com/OxfordCompetencyCenters/edu06_histology_labelling},
  license = {MIT}
}
```

See [CITATION.cff](CITATION.cff) for more citation formats.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Cellpose](https://github.com/MouseLand/cellpose) for cell segmentation
- [OpenAI](https://openai.com/) for GPT-4o vision capabilities
- [RAPIDS](https://rapids.ai/) for GPU-accelerated data processing
- Oxford Competency Centers for supporting this research
