# Intelligent Breast Cancer Diagnosis System

This repository implements a machine learning workflow to classify breast tissue samples as either **malignant** or **benign** based on features extracted from fine needle aspirate (FNA) images.

## Overview

The goal of this project is to evaluate and compare three popular machine learning algorithms—**Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Support Vector Machine (SVM)**—for binary breast cancer classification.

### Preprocessing & Workflow

* **Dataset**: `data/Assignment3-Breast-Cancer-Diagnose.csv`

* **Target Mapping**: Converted diagnoses to numeric labels (`1` for Malignant, `0` for Benign).


* **Data Cleaning & Scaling**: Removed empty attributes (feature index 30), applied mean imputation (`SimpleImputer`) for missing values, and standardized features via `StandardScaler`.


* **Data Splitting**: 80% Training (455 samples) and 20% Testing (114 samples) using `random_state=42`.


* **Hyperparameter Tuning**: Executed 5-fold cross-validation (`GridSearchCV`) across all pipelines.



---

## Model Evaluation & Performance

Models were evaluated on the test set using Accuracy, Precision, Recall, and F1-Score.

| Model | Accuracy | Precision | Recall | F1-Score | Best Parameters |
| --- | --- | --- | --- | --- | --- |
| **Support Vector Machine (SVM)**<br> | **0.9825**<br> | **1.0000**<br> | **0.9535**<br> | **0.9762**<br> | `C=0.1`, `kernel='linear'`<br> |
| **Logistic Regression**<br> | 0.9737

 | 0.9762

 | 0.9535

 | 0.9647

 | `C=0.1`<br> |
| **K-Nearest Neighbors (KNN)**<br> | 0.9474

 | 0.9302

 | 0.9302

 | 0.9302

 | `n_neighbors=5`<br> |

**Key Findings:**

* **SVM** demonstrated the highest performance, achieving perfect precision ($1.0000$) with no false positives.


* **Logistic Regression** produced comparable high accuracy while offering simplicity and interpretability.


* All three algorithms achieved $>94\%$ test accuracy.



---

## Getting Started

### Prerequisites

* Python 3.x
* Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/breast-cancer-diagnosis.git
cd breast-cancer-diagnosis

# Install the libraries listed in the notebook
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Open the analysis notebook
jupyter notebook notebooks/main.ipynb

```

---

## Project Files

- `notebooks/main.ipynb` — preprocessing, model training, and evaluation
- `data/Assignment3-Breast-Cancer-Diagnose.csv` — dataset
- `reports/report assignment 3.pdf` — project report

## Future Enhancements

* Implement ensemble models (e.g., Random Forests, Gradient Boosting).


* Perform feature selection and importance analysis to identify top FNA indicators.


* Implement nested cross-validation for hyperparameter tuning.
