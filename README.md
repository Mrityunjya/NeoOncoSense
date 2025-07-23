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

<img width="1515" height="870" alt="bcp1" src="https://github.com/user-attachments/assets/066e2a21-b0c2-4833-820d-58307c89e426" />

<img width="1576" height="862" alt="image" src="https://github.com/user-attachments/assets/68230b50-13f5-4fac-8507-e340ed077ef9" />


---

## 📄 License

MIT License © 2025
