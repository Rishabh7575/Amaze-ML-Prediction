# Amazon Price Prediction Project : Ai assisted readme file but tech stacks added by me 

A clean, modular, and beginner-friendly machine learning pipeline for predicting Amazon product prices using multi-modal feature fusion (text embeddings and image embeddings).

---

##  Key Features

* **Modular Source Code**: Cleanly organized modules under the `src/` package.
* **Multi-Modal Feature Fusion**: Integrates text embeddings (via Sentence Transformers) and visual embeddings (via Vision Transformers) seamlessly.
* **Robust Error Handling**: Graceful fallback to zero-embeddings if dependencies are missing or image downloads fail.
* **Robust Cross-Validation**: Uses 5-Fold cross-validation to train and evaluate:
  * A scaled **Linear Regression baseline**.
  * A highly-optimized **LightGBM regressor**.
* **Model Ensembling**: Simple blended predictions for superior submission accuracy.

---

## 📁 Repository Layout

```text
amazon-pred-proj/
├── README.md               # Project documentation and guide
├── requirements.txt        # Python library dependencies
├── main.py                 # CLI entrypoint to run the pipeline
├── app/                    # Web Application dashboard
│   └── app.py              # Streamlit dashboard and inference module
├── outputs/                # Managed pipeline output directory
│   ├── submissions/        # Saved final CSV submission files
│   ├── cached_embeddings/  # Cached .npy features for text & image modalities
│   ├── logs/               # Execution log files (pipeline.log)
│   ├── predictions/        # Saved OOF and test .npy prediction files
│   └── models/             # Serialized trained model weights (.pkl)
└── src/                    # Source code package
    ├── __init__.py         # Package initialization
    ├── config.py           # Central config, directory initializations, and paths
    ├── utils.py            # Image downloading and HTTP helper utilities
    ├── metrics.py          # Custom ML evaluation metrics (SMAPE)
    ├── text_embeddings.py  # Text embedding generator with cache support
    ├── image_embeddings.py # Image embedding generator with cache support
    ├── feature_fusion.py   # Feature fusion (prepare_features) with cache integration
    ├── regression_models.py# Model training & validation (LR / LightGBM)
    └── pipeline.py         # End-to-end logging, ensembling, and execution engine
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd amazon-pred-proj
   ```

2. **Install requirements**:
   Ensure you have Python 3.8+ installed. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Execution Guide

### 1. Run the Training & Validation Pipeline
Execute the pipeline via the CLI entrypoint at the repository root:
```bash
python main.py
```
This script will:
* Set up standard logging outputs to `outputs/logs/pipeline.log`.
* Load your datasets, assert validation schemas, and extract text and image features.
* Save computed embeddings to `outputs/cached_embeddings/` (enabling instant load on consecutive runs).
* Scale and train the regression models.
* Cache the trained weights to `outputs/models/` and OOF predictions to `outputs/predictions/`.
* Blend baseline predictions and format a final file in `outputs/submissions/submission.csv`.

### 2. Launch the Streamlit Dashboard
To run the interactive pricing predictor dashboard:
```bash
streamlit run app/app.py
```
The Streamlit app lets you inspect cache outputs, review traces in real-time, compare OOF metrics, and input descriptions/image URLs for instant multi-modal price predictions!

---

