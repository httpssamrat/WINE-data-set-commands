"""
=============================================================
CSE3231 - Machine Learning Lab
utils.py — Shared helper functions used across all lab files
Dataset   : Wine Recognition Dataset (UCI ML Repository)
Student   : SHABD SHARMA | 23FE10CSE00234 | Section J
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, ConfusionMatrixDisplay)

# ------------------------------------------------------------------
# COLUMN NAMES (used by every lab)
# ------------------------------------------------------------------
WINE_COLUMNS = [
    "Class", "Alcohol", "Malic_Acid", "Ash", "Alcalinity_of_Ash",
    "Magnesium", "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols",
    "Proanthocyanins", "Color_Intensity", "Hue", "OD280_OD315", "Proline"
]

FEATURE_COLUMNS = [c for c in WINE_COLUMNS if c != "Class"]

CLASS_NAMES = ["Class 1", "Class 2", "Class 3"]


# ------------------------------------------------------------------
# DATASET LOADER
# ------------------------------------------------------------------
def load_wine(path: str = "../data/wine.data") -> pd.DataFrame:
    """Load Wine dataset and return a labelled DataFrame."""
    df = pd.read_csv(path, header=None, names=WINE_COLUMNS)
    print(f"[utils] Wine dataset loaded — {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"[utils] Class distribution:\n{df['Class'].value_counts().sort_index()}\n")
    return df


# ------------------------------------------------------------------
# EVALUATION PRINTER
# ------------------------------------------------------------------
def print_classification_metrics(y_true, y_pred, model_name: str = "Model"):
    """Print accuracy, confusion matrix, and classification report."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*55}")
    print(f" {model_name}")
    print(f"{'='*55}")
    print(f" Accuracy         : {acc:.4f}")
    print(f"\n Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(f"\n Classification Report:\n{classification_report(y_true, y_pred)}")
    return acc


# ------------------------------------------------------------------
# CONFUSION MATRIX PLOTTER
# ------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix",
                           cmap: str = "Blues", save_path: str = None):
    """Plot a styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=CLASS_NAMES, cmap=cmap, ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[utils] Saved confusion matrix → {save_path}")
    plt.show()
    plt.close()


# ------------------------------------------------------------------
# CORRELATION HEATMAP
# ------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap",
                              save_path: str = None):
    """Plot a full-feature correlation heatmap."""
    plt.figure(figsize=(14, 11))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm",
                fmt=".2f", linewidths=0.4, square=True, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=15, pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[utils] Saved heatmap → {save_path}")
    plt.show()
    plt.close()


# ------------------------------------------------------------------
# FEATURE BOXPLOT (per class)
# ------------------------------------------------------------------
def plot_feature_boxplots(df: pd.DataFrame, features: list,
                           save_path: str = None):
    """Grid of boxplots for each feature, grouped by wine class."""
    n     = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    palette = {1: "#e74c3c", 2: "#3498db", 3: "#2ecc71"}

    for ax, feat in zip(axes.flatten(), features):
        sns.boxplot(data=df, x="Class", y=feat, palette=palette, ax=ax)
        ax.set_title(f"Distribution of {feat}")

    for extra_ax in axes.flatten()[n:]:
        extra_ax.set_visible(False)

    plt.suptitle("Wine Feature Distribution per Class", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Saved boxplots → {save_path}")
    plt.show()
    plt.close()


# ------------------------------------------------------------------
# ACCURACY BAR CHART
# ------------------------------------------------------------------
def plot_accuracy_comparison(scores: dict, title: str = "Model Accuracy Comparison",
                              save_path: str = None):
    """Bar chart comparing multiple model accuracies."""
    palette = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(max(6, len(scores) * 1.5), 5))
    bars = ax.bar(scores.keys(), scores.values(),
                  color=palette[:len(scores)], edgecolor="white")
    ax.set_ylim([max(0, min(scores.values()) - 0.08), 1.02])
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.005,
                f"{yval:.4f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[utils] Saved accuracy chart → {save_path}")
    plt.show()
    plt.close()


# ------------------------------------------------------------------
# QUICK DEMO
# ------------------------------------------------------------------
if __name__ == "__main__":
    df = load_wine()
    print(df.head())
    print("\nFeature Columns:", FEATURE_COLUMNS)
