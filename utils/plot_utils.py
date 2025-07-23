# plot_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap


def plot_correlation_heatmap(df: pd.DataFrame, figsize=(12, 10)):
    """
    Plots a correlation heatmap of the dataset features.

    Args:
        df (pd.DataFrame): The DataFrame containing feature columns.
        figsize (tuple): Figure size for the plot.
    """
    plt.figure(figsize=figsize)
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("📊 Feature Correlation Heatmap", fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plots the top N important features from a trained model.

    Args:
        model: Trained tree-based model (e.g., RandomForest, XGBoost).
        feature_names (list): List of feature names.
        top_n (int): Number of top features to plot.
    """
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
    plt.title("🔍 Top Feature Importances", fontsize=15)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_shap_summary(model, X, plot_type="bar"):
    """
    Plots SHAP summary plot for a trained model.

    Args:
        model: Trained tree-based model compatible with SHAP.
        X (pd.DataFrame): Input features.
        plot_type (str): 'bar' or 'dot'
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # For binary classification, shap_values is a list: [class_0_values, class_1_values]
    shap.summary_plot(shap_values[1], X, plot_type=plot_type)
