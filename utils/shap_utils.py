# shap_utils.py

import shap
import matplotlib.pyplot as plt
import numpy as np
import os


def init_shap_explainer(model, X_train):
    """
    Initialize the SHAP explainer for the given model.

    Args:
        model: Trained ML model (e.g., XGBoost, RandomForest, LogisticRegression).
        X_train (pd.DataFrame or np.ndarray): Training data used for background.

    Returns:
        shap.Explainer object
    """
    if hasattr(model, "predict_proba") and hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model, data=X_train)
    else:
        explainer = shap.Explainer(model.predict, X_train)
    return explainer


def compute_shap_values(explainer, X):
    """
    Compute SHAP values using the explainer on given data.

    Args:
        explainer: SHAP explainer object.
        X (pd.DataFrame or np.ndarray): Data to compute SHAP values on.

    Returns:
        shap_values
    """
    shap_values = explainer(X)
    return shap_values


def plot_summary(shap_values, X, output_dir="shap_outputs"):
    """
    Plot SHAP summary plot and save as image.

    Args:
        shap_values: SHAP values object.
        X: Input feature set.
        output_dir: Directory to save SHAP plots.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=300)
    plt.close()


def plot_feature_importance(shap_values, X, output_dir="shap_outputs"):
    """
    Plot SHAP bar feature importance plot.

    Args:
        shap_values: SHAP values object.
        X: Input feature set.
        output_dir: Directory to save SHAP plots.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_bar.png"), dpi=300)
    plt.close()

import numpy as np

def get_top_features(shap_values, X, top_n=5):
    if isinstance(shap_values, list):  # For XGBoost binary
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = dict(zip(X.columns, mean_abs_shap))
    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    return dict(list(sorted_features.items())[:top_n])
