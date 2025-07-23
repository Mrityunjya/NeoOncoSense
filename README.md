# 🧠 NeooncoSense: Explainable AI for Breast Cancer Prediction

**NeooncoSense** is a full-stack ML application built with **XGBoost**, **Random Forest**, and **SHAP** to provide **transparent, real-time breast cancer predictions**. Unlike static dashboards, this is a **live Streamlit web app** that simulates real clinical intelligence—**delivering insights, visual explanations, and downloadable medical-style reports in one unified flow**.

---

## 🚀 Key Features

- **🎯 Clinical-Grade Prediction Engine**  
  Trained on the UCI Breast Cancer Diagnostic dataset using ensemble models.

- **🧠 SHAP-Based Interpretability**  
  Feature impact visualization for each prediction (Force Plot, Waterfall).

- **📄 Intelligent PDF Report Generator**  
  Downloadable diagnostic report styled for clarity and medical readability.

- **⚡ Live Web Interface (Streamlit)**  
  Fully functional form-based interface with real-time model inference and visual feedback.

- **🧱 Modular Architecture**  
  Scalable, production-oriented layout with separate utility modules for SHAP, plotting, preprocessing, and reporting.

---

## 🧪 Run Locally

```bash
git clone https://github.com/your-username/neooncosense.git
cd neooncosense
pip install -r requirements.txt
cd app
streamlit run app.py
```

---

## 🖼️ Project Structure

```
NEOONCOSENSE/
│
├── app/
│ ├── pycache/ 
│ ├── static/
│ ├── templates/ 
│ ├── app.py 
│ └── generate_summary.py 
│
├── data/
│ └── data (1).csv 
│
├── llm/
│ └── prompt_templates.py 
│
├── models/
│ ├── random_forest_model.pkl 
│ └── xgboost_model.pkl 
│
├── notebooks/
│ └── Breast_Cancer_Detection_AI_Project.ipynb 
│
├── utils/
│ ├── plot_utils.py 
│ ├── preprocess.py
│ └── shap_utils.py 
│
└── README.md 
```

---

##  Visual Output Samples

### 🔍 SHAP Force Plot

<p align="center">
  <img src="assets/shap_force_example.png" width="600"/>
</p>

### 📄 PDF Diagnostic Report

<p align="center">
  <img src="assets/pdf_example_page.png" width="500"/>
</p>

---

## 📄 License

MIT License © 2025
