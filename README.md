# Cancer_Classification
A machine learning model that classifies cancer type based on RNA-Seq gene expression patterns. Understanding which genes are differentially expressed in different cancers helps design targeted delivery systems.
# Cancer Type Classification from Gene Expression Data

A machine learning pipeline for classifying cancer types based on RNA-Seq gene expression patterns using the TCGA PANCAN dataset.

## Project Overview

This project develops and compares multiple machine learning models to classify five cancer types from gene expression data. The pipeline includes comprehensive exploratory data analysis, preprocessing, dimensionality reduction, model training, hyperparameter tuning, and detailed evaluation.

### Cancer Types Classified
| Abbreviation | Cancer Type |
|--------------|-------------|
| BRCA | Breast Invasive Carcinoma |
| COAD | Colon Adenocarcinoma |
| KIRC | Kidney Renal Clear Cell Carcinoma |
| LUAD | Lung Adenocarcinoma |
| PRAD | Prostate Adenocarcinoma |

## Dataset

- **Source:** [TCGA PANCAN HiSeq](https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq)
- **Samples:** 801 patients
- **Features:** 20,531 genes (RNA-Seq expression values)
- **Classes:** 5 cancer types

### Class Distribution
```
BRCA: 300 samples (37.5%)
KIRC: 146 samples (18.2%)
LUAD: 141 samples (17.6%)
PRAD: 136 samples (17.0%)
COAD:  78 samples (9.7%)
```


## Pipeline Steps

### Step 1: Exploratory Data Analysis (EDA)
- Dataset overview and statistics
- Class distribution analysis
- Data quality checks (missing values, duplicates)
- Gene expression distribution visualization
- Correlation analysis

### Step 2: Data Preprocessing
- Missing value handling (median imputation)
- Feature normalization (StandardScaler)
- Label encoding
- Stratified train-test split (80-20)

### Step 3: Dimensionality Reduction
- PCA analysis for variance retention
- Reduction from 20,531 to ~530 features (95% variance)
- 2D and 3D visualization of cancer clusters

### Step 4: Model Building
Five classification models trained and compared:

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Decision Tree | Single tree classifier |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosting with regularization |
| Gradient Boosting | Sklearn gradient boosting |

**Hyperparameter Tuning:** GridSearchCV applied to XGBoost

### Step 5: Model Evaluation
Comprehensive metrics calculated:
- Accuracy
- Precision (Weighted & Macro)
- Recall (Weighted & Macro)
- F1-Score (Weighted & Macro)
- AUC-ROC (One-vs-Rest)
- TPR (True Positive Rate)
- TNR (True Negative Rate)
- Confusion Matrix

### Step 6: Feature Importance Analysis
- Logistic Regression coefficients interpretation
- Tree-based feature importance scores
- Mapping PCA components back to original genes
- Top contributing genes identification

## Results

### Model Performance Summary

| Model | Test Accuracy | F1-Score | AUC-ROC |
|-------|---------------|----------|---------|
| Logistic Regression | **99.38%** | **0.994** | **0.999** |
| Gradient Boosting | 98.76% | 0.988 | 0.998 |
| XGBoost | 97.52% | 0.975 | 0.997 |
| Random Forest | 87.58% | 0.874 | 0.982 |
| Decision Tree | 85.09% | 0.849 | 0.912 |

**Best Model:** Logistic Regression achieved 99.38% accuracy with only 1 misclassification out of 161 test samples.

### Key Findings
1. Linear models (Logistic Regression) outperformed complex ensemble methods
2. PCA effectively reduced dimensionality while preserving class separability
3. KIRC and BRCA show distinct expression patterns
4. Decision Tree showed overfitting tendencies

## Installation

### Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

### Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

## Usage

### Run Jupyter Notebook
```bash
jupyter notebook Cancer_Classification_Complete.ipynb
```

### Run Python Script
```bash
python cancer_classification.py
```

## Visualizations

The pipeline generates 12 visualization files:

1. **Class Distribution** - Bar and pie charts of cancer type frequencies
2. **EDA Visualizations** - Expression distributions and correlations
3. **PCA Variance** - Explained variance by principal components
4. **PCA 2D** - Cancer samples in 2D PCA space
5. **PCA 3D** - Cancer samples in 3D PCA space
6. **Model Comparison** - Bar charts of all evaluation metrics
7. **Confusion Matrices** - Per-model classification results
8. **ROC Curves** - One-vs-Rest ROC curves per class
9. **Overfitting Analysis** - Train vs Test accuracy comparison
10. **PCA Feature Importance** - Important principal components
11. **Gene Importance** - Top contributing genes
12. **Executive Summary** - Dashboard with key results

## Limitations

- Class imbalance (BRCA: 300 vs COAD: 78 samples)
- Gene names are anonymized (gene_0, gene_1, etc.)
- Limited to 5 cancer types
- No external validation dataset

## Future Improvements

- Apply SMOTE or class weighting for imbalanced data
- Use real gene names for biological interpretation
- Extend to more cancer types
- Validate on independent datasets
- Implement deep learning approaches (neural networks)
- Add cross-validation with multiple random seeds

## References

- [TCGA Pan-Cancer Analysis Project](https://www.cancer.gov/tcga)
- [UCI Machine Learning Repository - Gene Expression Cancer RNA-Seq](https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq)

## License

This project is for educational purposes. The TCGA data is publicly available for research use.

---

*Generated with machine learning pipeline for cancer classification*

