# Churn Prediction App

This Streamlit app performs advanced EDA with interactive Plotly visualizations and trains multiple classification models for churn prediction.

Requirements
- Python 3.8+
- Install dependencies: pip install -r requirements.txt

Run

```powershell
# from the project root
pip install -r requirements.txt
streamlit run "d:\churn prediction\app.py"
```

Notes
- XGBoost, LightGBM and CatBoost are optional; if not installed the app will still run but those models won't be available.
- Plotly is used for interactive EDA. If Plotly is not installed the app falls back to matplotlib/seaborn static plots.
