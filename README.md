# SCS-CNN: An Interpretable-by-Design Spatio-Channel Scaling Attention for Spatio-Temporal Forecasting

This repository contains the official PyTorch implementation for the paper: **"SCS-CNN: An Interpretable-by-Design Spatio-Channel Scaling Attention for Spatio-Temporal Forecasting"**.

Our work introduces the Spatio-Channel Scaling Convolutional Neural Network (SCS-CNN), a novel architecture designed for accurate and physically interpretable El Niño-Southern Oscillation (ENSO) forecasting.

## Key Features

- **Interpretable by Design:** The core SCS-Attention module provides direct insight into the model's learned spatial and channel-wise priorities.
- **Demonstrates a certain long-term prediction capability:** Achieves results of reference value for the Niño 3.4 index up to 18 months in advance.
- **Robust Training Strategy:** Implements a two-stage pre-training and fine-tuning workflow using both climate model (CMIP5) and reanalysis (SODA) data.
- **Modular and Reproducible:** The codebase is structured for clarity, scalability, and easy reproduction of our results.

## Project Structure

```
SCS-CNN-for-ENSO/
│
├── data/
│   ├── (Your .nc files go here)
│
├── configs/
│   ├── pretrain_base.yaml
│   └── finetune_base.yaml
│
├── src/
│   ├── data_loader.py
│   ├── models/
│   │   └── scs_cnn.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── lr_scheduler.py
│   │   └── losses.py
│   └── utils/
│       ├── config_parser.py
│       └── checkpoint.py
│
├── scripts/
│   ├── run_experiment.sh
│   └── plot_results.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/SCS-CNN-for-ENSO.git
    cd SCS-CNN-for-ENSO
    ```

2.  **Create a Conda environment (recommended):**
    A dedicated environment ensures that all dependencies are managed correctly without conflicts.
    ```bash
    conda create -n scs_cnn python=3.8
    conda activate scs_cnn
    ```

3.  **Install dependencies:**
    The `requirements.txt` file lists all necessary packages. We highly recommend installing `cartopy` via Conda to handle its complex geospatial dependencies.
    ```bash
    conda install -c conda-forge --file requirements.txt
    ```

## Data Preparation

You will need to download the following datasets and place them in the `data/` directory:

- **CMIP5:**
  - `CMIP5.input.36mon.1861_2001.nc`
  - `CMIP5.label.12mon.1863_2003.nc`
- **SODA:**
  - `SODA.input.36mon.1871_1970.nc`
  - `SODA.label.12mon.1873_1972.nc`
- **GODAS:**
  - `GODAS.input.36mon.1980_2015.nc`
  - `GODAS.label.12mon.1982_2017.nc`

These datasets are publicly available from their respective climate data portals.

## How to Run Experiments

The entire experimental workflow is managed by the `scripts/run_experiment.sh` script, which automates the process across different lead times and model configurations.

The script performs three main stages for each configuration:
1.  **Pre-training on CMIP5:** Trains the model on a large, diverse dataset to learn general features.
2.  **Fine-tuning on SODA:** Adapts the pre-trained model to observationally-constrained data.
3.  **Generating Interpretability Maps:** Calculates and saves saliency maps during fine-tuning.

#### Running an Experiment

To run a full experiment for a specific lead time (e.g., 9 months) and target month (e.g., January):

1.  Make the script executable:
    ```bash
    chmod +x scripts/run_experiment.sh
    ```
2.  Execute the script with the desired lead and target months. This example will train an ensemble of 10 models for a 9-month lead forecast targeting January.
    ```bash
    ./scripts/run_experiment.sh 9 1
    ```

The script will generate trained models, logs, and interpretability data in the `experiments/` directory, which is created automatically.

## Generating Figures

After running the experiments, you can generate all the figures from the paper using the `scripts/plot_results.py` script. This script is designed to find the necessary experiment outputs and produce the final visualizations.
