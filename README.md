# Car Accident Prediction on Driving Simulator Data Using Firefly-Optimized SVM with SMOTE

This repository contains the code, data, and results for the conference paper: **"Car Accident Prediction on Driving Simulator Data Using Firefly-Optimized SVM with SMOTE-Based Class Balancing"**.

##  Project Overview

Road accidents are rare events, leading to highly imbalanced datasets. In our dataset (5,000 observations from a driving simulator), only **2%** of the records represent accidents. Standard classifiers tend to ignore this minority class, achieving high accuracy but null Recall. 

This project implements a robust pipeline to address this issue:
1. **Data Preprocessing:** Stratified split (80/20) and Standard Scaling.
2. **Class Balancing:** SMOTE applied **only on the training set** to avoid data leakage.
3. **Hyperparameter Optimization:** Firefly Algorithm (FA) to find the optimal `C` and `gamma` for an SVM with an RBF kernel.
4. **Fair Evaluation:** Decision threshold tuning applied to all compared models to maximize the F1-score.
5. **Uncertainty Quantification:** Bootstrap 95% confidence intervals for Recall and F1-score.

## Key Results

The final model (**SVM + SMOTE + Firefly**) achieves a **Recall of 0.65** (detecting 13 out of 20 accidents) at a deliberately low threshold (0.0017). 

| Model | Threshold | Accuracy | Precision | Recall | F1 | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM Baseline** | 0.0719 | 0.971 | 0.263 | 0.250 | 0.256 | 0.810 |
| **SVM + SMOTE** | 0.9492 | 0.977 | 0.364 | 0.200 | 0.258 | 0.726 |
| **SVM + SMOTE + Firefly** | 0.0017 | 0.900 | 0.123 | **0.650** | 0.206 | 0.746 |

> **Why such a low threshold?** In road safety, the cost of a False Negative (missing an accident) is 100 to 1000 times higher than the cost of a False Positive (a false alarm). The threshold is a deliberate business choice to prioritize Recall over Precision.

## Repository Structure

*   `accident_prediction.py`: The main Python script containing the entire pipeline (preprocessing, SMOTE, Firefly optimization, evaluation, and figure generation).
*   `crash_data_reduced_5000.csv`: The dataset used for the experiments.
*   `results_comparison_dynamic.csv`: The output CSV file containing the final metrics and confidence intervals.
*   `figures/`: Folder containing the generated plots (ROC curves, confusion matrix, Firefly convergence, etc.).

##  Requirements

To run the code, you will need the following Python libraries:
```
pip install numpy pandas matplotlib scikit-learn imbalanced-learn
```

##  How to Run

1. Clone the repository:
```
git clone https://github.com/safaa241/car-accident-firefly.git
```
2. Navigate to the project folder:
```
cd car-accident-firefly
```
3. Run the script:
```
python accident_prediction.py
```
The script will output the metrics in the console, generate the figures, and save the results in `results_comparison_dynamic.csv`.

##  Figures Generated

*   `fig_firefly_convergence.png`: Shows the convergence of the Firefly algorithm and the trajectory of the optimal hyperparameters (C and gamma).
*   `fig_confusion_matrix.png`: Confusion matrix of the final model at the optimal threshold.
*   `fig_roc_curves.png`: Comparative ROC curves for the three models.
*   `fig_metrics_comparison.png`: Bar chart comparing Accuracy, Precision, Recall, F1, and AUC.
