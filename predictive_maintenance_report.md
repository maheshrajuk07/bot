# Predicting Machine Failures with SVM, KNN, and MLP
_A Small Benchmark on the AI4I 2020 Predictive Maintenance Dataset_

**Author:** Mahesh  
**Date:** 2025-12-01  

---

## 1. Overview
I compared three models to predict machine failure:
1. **Support Vector Machine (SVM)**
2. **K-Nearest Neighbors (KNN)**
3. **Multi-Layer Perceptron (MLP)**

## 2. Visualizations

### Class Balance
Failures are rare (~3.4%).
![Class Balance](class_balance.png)

### Performance Comparison
![Metrics](metrics_comparison.png)

| Model | Accuracy | ROC AUC | PR AUC | F1 (Fail) |
|-------|----------|---------|--------|-----------|
| SVM (RBF) | 0.915 | 0.967 | 0.647 | 0.426 |
| KNN (5-Neighbors) | 0.975 | 0.876 | 0.581 | 0.463 |
| MLP (Neural Net) | 0.983 | 0.975 | 0.795 | 0.729 |

### ROC Curves
![ROC Curves](roc_curves.png)

### Precision-Recall Curves
![PR Curves](pr_curves.png)

## 3. Best Model Analysis: MLP (Neural Net)

### Feature Importance (Permutation)
Using permutation importance to understand what drives the MLP (Neural Net) model:
![Importance](feature_importance.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Data Insights (Torque vs Wear)
![Scatter](scatter.png)

## 4. Conclusion
The **MLP (Neural Net)** performed best with an F1-score of **0.729**. 
Neural networks and distance-based methods proved effective for this tabular sensor data.
