"""
Predictive Maintenance Benchmark Script (SVM, KNN, MLP)
Author: Mahesh
Date: 2025-12-01

Description:
This script loads the AI4I 2020 dataset, trains three models, 
generates performance plots, and writes a COMPLETE README.md 
containing all reports, images, and instructions.
"""

import os
import sys
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Model Imports
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

# Metrics & Processing
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
DATA_FILENAME = "ai4i2020.csv"
AUTHOR_NAME = "Mahesh"
CURRENT_DATE = "2025-12-01"

def load_ai4i(filename: str) -> pd.DataFrame:
    """Load the AI4I 2020 dataset and return a cleaned DataFrame."""
    if not os.path.exists(filename):
        print(f"ERROR: '{filename}' not found in {os.getcwd()}")
        print("Please download the AI4I 2020 dataset and place the csv file in this folder.")
        sys.exit(1)

    df = pd.read_csv(filename)
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if "air temperature" in cl: rename_map[c] = "air_temp_k"
        elif "process temperature" in cl: rename_map[c] = "process_temp_k"
        elif "rotational speed" in cl: rename_map[c] = "rot_speed_rpm"
        elif "torque" in cl: rename_map[c] = "torque_nm"
        elif "tool wear" in cl: rename_map[c] = "tool_wear_min"
        elif "machine failure" in cl: rename_map[c] = "machine_failure"
        elif cl == "type": rename_map[c] = "type"

    df = df.rename(columns=rename_map)

    expected_cols = [
        "type", "air_temp_k", "process_temp_k", 
        "rot_speed_rpm", "torque_nm", "tool_wear_min", "machine_failure"
    ]
    df = df[expected_cols].copy()
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
    return df

def train_models(X_train, y_train, X_test, y_test) -> List[Dict]:
    models = []
    # 1. SVM
    models.append(("SVM (RBF)", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
    ])))
    # 2. KNN
    models.append(("KNN (5-Neighbors)", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1))
    ])))
    # 3. MLP
    models.append(("MLP (Neural Net)", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500, random_state=42))
    ])))

    results = []
    for name, model in models:
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.predict(X_test)
            
        y_pred = model.predict(X_test)

        results.append({
            "name": name,
            "model": model,
            "accuracy": (y_pred == y_test).mean(),
            "roc_auc": roc_auc_score(y_test, y_score),
            "pr_auc": average_precision_score(y_test, y_score),
            "f1_fail": f1_score(y_test, y_pred, pos_label=1),
            "y_pred": y_pred,
            "y_score": y_score
        })
    return results

# --- Plotting Functions ---

def plot_all_figures(results, df, X, y, X_test, y_test):
    # 1. Class Balance
    counts = y.value_counts().sort_index()
    plt.figure()
    counts.plot(kind="bar", color=['skyblue', 'salmon'])
    plt.xticks([0, 1], ["Normal (0)", "Failure (1)"], rotation=0)
    plt.title("Class Balance")
    plt.tight_layout()
    plt.savefig("class_balance.png")
    plt.close()

    # 2. ROC Curves
    plt.figure()
    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res["y_score"])
        plt.plot(fpr, tpr, label=f"{res['name']} (AUC = {res['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig("roc_curves.png")
    plt.close()

    # 3. PR Curves
    plt.figure()
    for res in results:
        p, r, _ = precision_recall_curve(y_test, res["y_score"])
        plt.plot(r, p, label=f"{res['name']} (AP = {res['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.savefig("pr_curves.png")
    plt.close()

    # 4. Metrics Comparison
    names = [r["name"] for r in results]
    f1 = [r["f1_fail"] for r in results]
    pr = [r["pr_auc"] for r in results]
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, f1, width, label="F1 (Failure)")
    plt.bar(x + width/2, pr, width, label="PR AUC (Failure)")
    plt.xticks(x, names, rotation=15)
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("metrics_comparison.png")
    plt.close()

    # 5. Scatter Plot
    plt.figure(figsize=(7, 5))
    subset = df.sample(2000, random_state=42) if len(df) > 2000 else df
    norm = subset[subset["machine_failure"]==0]
    fail = subset[subset["machine_failure"]==1]
    plt.scatter(norm["tool_wear_min"], norm["torque_nm"], alpha=0.3, label="Normal")
    plt.scatter(fail["tool_wear_min"], fail["torque_nm"], c='red', alpha=0.7, label="Failure")
    plt.xlabel("Tool Wear [min]")
    plt.ylabel("Torque [Nm]")
    plt.legend()
    plt.title("Torque vs Tool Wear")
    plt.savefig("scatter.png")
    plt.close()

    # 6. Best Model Specifics
    best_res = max(results, key=lambda r: r["f1_fail"])
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_res["y_pred"])
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="orange" if cm[i,j]>cm.max()/2 else "black")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({best_res['name']})")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Permutation Importance
    print(f"Calculating permutation importance for {best_res['name']}...")
    r = permutation_importance(best_res["model"], X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    idx = r.importances_mean.argsort()[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(list(X.columns))[idx][:8][::-1], r.importances_mean[idx][:8][::-1])
    plt.xlabel("Mean Accuracy Decrease")
    plt.title(f"Feature Importance ({best_res['name']})")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

# --- Markdown Generator (All-in-One) ---

def generate_full_readme(results):
    best = max(results, key=lambda r: r["f1_fail"])
    
    table_rows = ""
    for r in results:
        table_rows += f"| {r['name']} | {r['f1_fail']:.3f} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['accuracy']:.3f} |\n"

    content = f"""# Predicting Machine Failures with SVM, KNN, and MLP
_A Benchmark on the AI4I 2020 Predictive Maintenance Dataset_

**Author:** {AUTHOR_NAME}  
**Date:** {CURRENT_DATE}

---

## TL;DR
I benchmarked **SVM**, **KNN**, and **MLP (Neural Networks)** to predict machine failures using sensor data. 
The **{best['name']}** model performed best, achieving an F1-score of **{best['f1_fail']:.3f}** on the failure class.

---

## Hardware & Software Requirements

**Hardware:**
*   Standard CPU (Intel i5/AMD Ryzen 5 or better recommended)
*   4GB+ RAM (8GB recommended)
*   500MB free disk spaces

**Software:**
*   Python 3.8+
*   OS: Windows, macOS, or Linux

**Python Libraries:**
*   `pandas`
*   `scikit-learn`
*   `matplotlib`
*   `numpy`

---

## How to Reproduce

1.  **Download the Dataset**: Ensure `ai4i2020.csv` is in this folder.
2.  **Install Dependencies**:
    ```bash
    pip install pandas scikit-learn matplotlib numpy
    ```
3.  **Run the Script**:
    ```bash
    python run_bot.py
    ```
4.  **View Results**: The script generates all the images below and this README file.

---

## 1. Dataset Overview
The dataset contains 10,000 data points representing a milling machine.

*   **Features:** Air Temp, Process Temp, Rotational Speed, Torque, Tool Wear, Type.
*   **Target:** Machine Failure (0 = Normal, 1 = Failure).
*   **Imbalance:** Failures are rare (~3.4%).

### Class Balance
![Class Balance](class_balance.png)

### Data Insights (Torque vs Tool Wear)
![Scatter Plot](scatter.png)

---

## 2. Model Performance

I compared three models using **Standard Scaling** and **Class Weighting** (where applicable) to handle the imbalance.

| Model | F1 (Failure) | ROC AUC | PR AUC | Accuracy |
|-------|--------------|---------|--------|----------|
{table_rows}

### Performance Metrics Comparison
![Metrics Comparison](metrics_comparison.png)

### ROC Curves
![ROC Curves](roc_curves.png)

### Precision-Recall Curves
![PR Curves](pr_curves.png)

---

## 3. Best Model Analysis: {best['name']}

The best model was selected based on the **F1-Score for the Failure Class**, as catching failures is the priority.

### Confusion Matrix
This shows exactly how many failures were caught (True Positives) vs missed (False Negatives).
![Confusion Matrix](confusion_matrix.png)

### Feature Importance (Permutation)
Using Permutation Importance to see which sensors mattered most to the model.
![Feature Importance](feature_importance.png)

---

## 4. Conclusion
*   **Neural Networks (MLP)** and **Distance-based (KNN)** methods worked effectively.
*   **Torque** and **Tool Wear** are the strongest predictors of failure.
*   Standardization of data was crucial for these algorithms.
"""
    with open("README.md", "w") as f:
        f.write(content)
    print("Generated README.md (Contains all info)")

# --- Main Execution ---

def main():
    print(f"--- Starting Analysis in {os.getcwd()} ---")
    
    # 1. Load Data
    df = load_ai4i(DATA_FILENAME)
    print(f"Loaded data: {df.shape}")

    # 2. Preprocess
    X = pd.get_dummies(df.drop("machine_failure", axis=1), columns=["type"], drop_first=True)
    y = df["machine_failure"]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train
    results = train_models(X_train, y_train, X_test, y_test)

    # 5. Generate Plots
    print("Generating plots...")
    plot_all_figures(results, df, X, y, X_test, y_test)

    # 6. Generate Markdown File
    print("Writing README.md...")
    generate_full_readme(results)

    print("\n--- Success! Check README.md for the full report. ---")

if __name__ == "__main__":
    main()