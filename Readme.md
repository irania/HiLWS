# HiLWS: Human-in-the-Loop Weak Supervision for Remote Neurological Video Assessment

This repository contains the official implementation of **HiLWS**, a cascaded human-in-the-loop weak supervision framework for curating and annotating hand motor task videos from clinical and home settings. The method addresses label ambiguity and data quality heterogeneity in remote assessments of motor symptoms such as Parkinson’s Disease.

> 🧠 *Presented at ICML 2025 – DataWorld Workshop*

## Table of Contents

* [Overview](#overview)
* [Installation](#installation)
* [Data](#data)
* [HiLWS Framework](#hilws-framework)
* [Evaluation](#evaluation)


---

## Overview

HiLWS is a two-stage weak supervision pipeline:

1. **Initial Weak Label Fusion**: Aggregates multiple noisy expert annotations into probabilistic labels.
2. **Model Training & Refinement**: Trains machine learning models on the probabilistic labels and refines predictions with targeted expert corrections in a second weak supervision stage.

The full pipeline includes:

* Quality filtering
* Optimized pose estimation
* Task segmentation
* Context-sensitive evaluation (e.g., FPR₀, MAE, entropy)

## Installation

1. Clone the repository   
```bash
git clone https://github.com/your-username/hilws.git
cd hilws
```
2. Create and activate the Conda environment:   
```
   conda env create -f environment.yml   
```
3. Activate the Conda environment:
```
   conda activate booth_reports

    pip install -e .
```

## Data

The framework supports:

* Clinical video recordings with structured protocol
* Home video data with diverse visual and behavioral quality

⚠️ Due to privacy constraints, raw video data is not included. Please contact the authors for access.

## HiLWS Framework

```
          ┌──────────────────────┐
          │  Raw Video + Labels  │
          └────────┬─────────────┘
                   ▼
        ┌────────────────────────────┐
        │ Quality Filtering & Pose   │
        └────────┬───────────────┬───┘
                 ▼               ▼
     ┌────────────────┐   ┌───────────────┐
     │ Probabilistic  │   │ Model Training│
     │ Label Fusion   │   └──────┬────────┘
     └──────┬─────────┘          ▼
            ▼             ┌───────────────────────────────────────────────┐
       Stage 1 Labels     │ Stage 2: combine models and stage 1 labels    │
                          └───────────────────────────────────────────────┘
```

## Evaluation

Metrics reported:

* Mean Absolute Error (MAE) ↓
* F1 Score ↑
* False Positive Rate at class 0 (FPR₀) ↓
* Entropy of predicted labels

See `HiLws_analysis.ipynb` for sample outputs and comparison across label strategies.
