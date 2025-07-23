import pandas as pd
import shap
import joblib
import os
from utils.shap_utils import get_top_features

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "data (1).csv")

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def generate_shap_summary(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    summary = get_top_features(shap_values, X, top_n=5)
    return summary

if __name__ == "__main__":
    print("🔍 Loading model...")
    model = load_model()

    print("📊 Loading data...")
    data = load_data()

    # Assume label is in the last column
    X = data.iloc[:, :-1]

    print("⚙️ Generating SHAP summary...")
    summary = generate_shap_summary(model, X)

    print("✅ Top 5 Feature Importance Summary:")
    for i, (feature, value) in enumerate(summary.items(), 1):
        print(f"{i}. {feature}: {value:.4f}")
