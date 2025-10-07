# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pickle
try:
    import joblib
    joblib_installed = True
except Exception:
    joblib_installed = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    plotly_installed = True
except Exception:
    plotly_installed = False

# Optional: install LightGBM and CatBoost if you want
try:
    import lightgbm as lgb
    lightgbm_installed = True
except:
    lightgbm_installed = False

try:
    from catboost import CatBoostClassifier
    catboost_installed = True
except:
    catboost_installed = False

st.set_page_config(page_title="User Churn Prediction", layout="wide")
st.title("🛡️ User Churn Prediction App (Advanced EDA + Models)")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"]) 

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Organize app into tabs: Data, EDA, Modeling
    tab_data, tab_eda, tab_model = st.tabs(["Data", "EDA (Plotly)", "Modeling"])

    # --- Data tab ---
    with tab_data:
        st.subheader("Dataset Preview")
        st.dataframe(data.head())
        st.markdown("**Dataset Info**")
        buffer = []
        data.info(buf=buffer)
        st.text('\n'.join(buffer))
        st.write("Missing values per column:")
        st.dataframe(data.isnull().sum().rename('missing_count'))

        st.markdown("---")
        st.subheader("Basic Summary Stats")
        st.dataframe(data.describe(include='all').T)

    # --- EDA tab ---
    with tab_eda:
        st.subheader("Advanced Exploratory Data Analysis (interactive)")
        if not plotly_installed:
            st.warning("Plotly is not installed in the environment. Install it to enable interactive charts: `pip install plotly`")
        sample_size = st.slider("Sample size for EDA (for performance)", min_value=100, max_value=min(50000, len(data)), value=min(5000, len(data)), step=100)
        df = data.sample(n=min(len(data), sample_size), random_state=42)

        st.markdown("### Missing Values")
        missing = df.isnull().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if not missing.empty:
            if plotly_installed:
                fig_mv = px.bar(missing.reset_index().rename(columns={'index':'column', 0:'missing'}), x='index', y=0, labels={'index':'column', 0:'missing'}, title='Missing values by column')
                fig_mv.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_mv, use_container_width=True)
            else:
                st.dataframe(missing)
        else:
            st.info("No missing values found in the sampled data.")

        st.markdown("---")
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("Variable explorer")
            all_cols = df.columns.tolist()
            select_x = st.selectbox("Select column to explore", all_cols)
            # Determine numeric/categorical
            if pd.api.types.is_numeric_dtype(df[select_x]):
                st.markdown("Numeric column detected — distribution and outliers")
                if plotly_installed:
                    fig_hist = px.histogram(df, x=select_x, nbins=50, title=f'Distribution of {select_x}', marginal='box')
                    st.plotly_chart(fig_hist, use_container_width=True)
                    fig_violin = px.violin(df, y=select_x, box=True, points='outliers', title=f'Violin plot of {select_x}')
                    st.plotly_chart(fig_violin, use_container_width=True)
                else:
                    fig, ax = plt.subplots(1,2, figsize=(12,4))
                    sns.histplot(df[select_x].dropna(), kde=True, ax=ax[0])
                    sns.boxplot(x=df[select_x].dropna(), ax=ax[1])
                    st.pyplot(fig)
            else:
                st.markdown("Categorical column detected — counts and target relationship")
                if plotly_installed:
                    fig_bar = px.histogram(df, x=select_x, title=f'Counts of {select_x}')
                    fig_bar.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.write(df[select_x].value_counts())

        with col2:
            st.subheader("Pairwise & Correlation")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                select_corr = st.multiselect("Select numeric columns for pairwise / correlation", numeric_cols, default=numeric_cols[:6])
                if len(select_corr) >= 2:
                    corr = df[select_corr].corr()
                    if plotly_installed:
                        fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu', zmin=-1, zmax=1, title='Correlation matrix')
                        st.plotly_chart(fig_corr, use_container_width=True)
                        fig_scatter = px.scatter_matrix(df[select_corr], dimensions=select_corr, title='Scatter matrix')
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(8,6))
                        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu', ax=ax)
                        st.pyplot(fig)
                else:
                    st.info('Select at least two numeric columns to compute correlation / scatter matrix.')
            else:
                st.info('Not enough numeric columns for pairwise analysis.')

        st.markdown('---')
        st.subheader('Target analysis (if you choose a target)')
        target_column = st.selectbox('Select the target column (optional) for EDA', [None] + list(df.columns))
        if target_column:
            st.write(df[target_column].value_counts())
            # Show target vs numeric boxplots
            num_for_target = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
            if num_for_target:
                sel = st.selectbox('Select numeric column to compare vs target', num_for_target)
                if plotly_installed:
                    fig_bt = px.box(df, x=target_column, y=sel, points='outliers', title=f'{sel} distribution by {target_column}')
                    st.plotly_chart(fig_bt, use_container_width=True)
                else:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=target_column, y=sel, data=df, ax=ax)
                    st.pyplot(fig)

    # --- Modeling tab ---
    with tab_model:
        st.subheader('Modeling & Evaluation')
        # Step 2: Target selection
        target_column = st.selectbox("Select the target column (churn column)", data.columns, key='model_target')
        # Step 3: Feature selection
        features = st.multiselect("Select feature columns", [col for col in data.columns if col != target_column], default=[col for col in data.columns if col != target_column], key='model_features')

        X = data[features].copy()
        y = data[target_column].copy()

        # Encode categorical features
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Encode target if categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))

        # Optional: Scale features (we'll fit if training locally or use uploaded preprocessor)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Step 4: Train-test split
        test_size = st.slider("Select test size (percentage)", min_value=10, max_value=50, value=20, key='test_size')
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size/100, random_state=42)

        # Step 5: Model source selection: train locally or upload pretrained
        st.markdown('### Model source')
        model_mode = st.radio('Choose how you want to get a model', ['Train locally', 'Upload pretrained model'], index=0)

        results = []
        model_objects = {}

        if model_mode == 'Train locally':
            st.markdown('### Training Multiple Models')
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            }

            if lightgbm_installed:
                models["LightGBM"] = lgb.LGBMClassifier(random_state=42)
            if catboost_installed:
                models["CatBoost"] = CatBoostClassifier(verbose=0, random_state=42)

            selected_models = st.multiselect('Select models to train', list(models.keys()), default=list(models.keys()))

            for name in selected_models:
                model = models[name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                try:
                    roc = roc_auc_score(y_test, y_pred_prob)
                except Exception:
                    roc = np.nan
                results.append([name, acc, f1, roc])
                model_objects[name] = model

            results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1-score", "ROC-AUC"]).sort_values(by='ROC-AUC', ascending=False)
            st.subheader("Model Performance Comparison")
            st.dataframe(results_df)

            if not results_df.empty:
                best_model_name = results_df.iloc[0]["Model"]
                st.success(f"Best Model (by ROC-AUC): {best_model_name}")
                best_model = model_objects[best_model_name]

                # Confusion Matrix
                st.subheader(f"Confusion Matrix ({best_model_name})")
                y_pred_best = best_model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred_best)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                # ROC Curve
                try:
                    st.subheader(f"ROC Curve ({best_model_name})")
                    y_pred_prob_best = best_model.predict_proba(X_test)[:,1] if hasattr(best_model, "predict_proba") else y_pred_best
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_best)
                    roc_auc = roc_auc_score(y_test, y_pred_prob_best)
                    fig2, ax2 = plt.subplots()
                    ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
                    ax2.plot([0,1],[0,1],'--', color='gray')
                    ax2.set_xlabel("False Positive Rate")
                    ax2.set_ylabel("True Positive Rate")
                    ax2.set_title("ROC Curve")
                    ax2.legend()
                    st.pyplot(fig2)
                except Exception:
                    st.info('ROC curve could not be computed for this model.')

                # Feature Importance for tree-based models
                tree_models = ["Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
                if best_model_name in tree_models and hasattr(best_model, 'feature_importances_'):
                    st.subheader(f"Feature Importance ({best_model_name})")
                    importance = best_model.feature_importances_
                    feature_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
                    feature_imp = feature_imp.sort_values(by='Importance', ascending=False)
                    fig3, ax3 = plt.subplots(figsize=(10,6))
                    sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax3)
                    st.pyplot(fig3)

        else:
            st.markdown('### Upload pretrained model')
            st.info('Upload a saved model (.pkl or .joblib). Optionally upload a preprocessor (scaler/encoders) used during training.')
            uploaded_model_file = st.file_uploader('Upload model file (.pkl or .joblib)', type=['pkl','joblib'], key='uploaded_model')
            uploaded_preproc_file = st.file_uploader('Optional: upload preprocessor (scaler/encoders) (.pkl or .joblib)', type=['pkl','joblib'], key='uploaded_preproc')

            loaded_model = None
            loaded_preproc = None

            if uploaded_model_file is not None:
                # Try loading with joblib then pickle
                try:
                    if joblib_installed:
                        uploaded_model_file.seek(0)
                        loaded_model = joblib.load(uploaded_model_file)
                    else:
                        uploaded_model_file.seek(0)
                        loaded_model = pickle.load(uploaded_model_file)
                    st.success('Model loaded successfully.')
                except Exception as e:
                    st.error(f'Failed to load model: {e}')

            if uploaded_preproc_file is not None:
                try:
                    if joblib_installed:
                        uploaded_preproc_file.seek(0)
                        loaded_preproc = joblib.load(uploaded_preproc_file)
                    else:
                        uploaded_preproc_file.seek(0)
                        loaded_preproc = pickle.load(uploaded_preproc_file)
                    st.success('Preprocessor loaded successfully.')
                except Exception as e:
                    st.error(f'Failed to load preprocessor: {e}')

            if loaded_model is not None:
                # If a preprocessor (dict or object) was uploaded, try to use it
                if loaded_preproc is not None:
                    # If preprocessor is a dict with scaler/encoders, handle accordingly
                    if isinstance(loaded_preproc, dict):
                        scaler_obj = loaded_preproc.get('scaler')
                        encoders = loaded_preproc.get('encoders')
                    else:
                        scaler_obj = getattr(loaded_preproc, 'scaler', None) if hasattr(loaded_preproc, '__dict__') else None
                        encoders = getattr(loaded_preproc, 'encoders', None) if hasattr(loaded_preproc, '__dict__') else None
                else:
                    scaler_obj = None
                    encoders = None

                # If encoders provided, apply them; otherwise fallback to LabelEncoder fit
                X_proc = X.copy()
                if encoders and isinstance(encoders, dict):
                    for col, enc in encoders.items():
                        if col in X_proc.columns:
                            try:
                                X_proc[col] = enc.transform(X_proc[col].astype(str))
                            except Exception:
                                X_proc[col] = enc.fit_transform(X_proc[col].astype(str))
                else:
                    for col in X_proc.select_dtypes(include=['object', 'category']).columns:
                        le = LabelEncoder()
                        X_proc[col] = le.fit_transform(X_proc[col].astype(str))

                # Scale
                if scaler_obj is not None:
                    try:
                        X_scaled = scaler_obj.transform(X_proc)
                    except Exception:
                        X_scaled = scaler.fit_transform(X_proc)
                else:
                    X_scaled = scaler.fit_transform(X_proc)

                # Evaluate uploaded model on test set
                try:
                    y_pred = loaded_model.predict(X_test if 'X_test' in locals() else X_scaled)
                    y_pred_prob = loaded_model.predict_proba(X_test if 'X_test' in locals() else X_scaled)[:,1] if hasattr(loaded_model, 'predict_proba') else y_pred
                    acc = accuracy_score(y_test, y_pred) if 'y_test' in locals() else accuracy_score(y, y_pred)
                    f1 = f1_score(y_test, y_pred) if 'y_test' in locals() else f1_score(y, y_pred)
                    try:
                        roc = roc_auc_score(y_test, y_pred_prob) if 'y_test' in locals() else roc_auc_score(y, y_pred_prob)
                    except Exception:
                        roc = np.nan
                    st.subheader('Uploaded model evaluation')
                    st.dataframe(pd.DataFrame([[ 'Uploaded model', acc, f1, roc]], columns=["Model","Accuracy","F1-score","ROC-AUC"]))

                    # Confusion matrix
                    st.subheader('Confusion Matrix (Uploaded model)')
                    cm = confusion_matrix(y_test, y_pred) if 'y_test' in locals() else confusion_matrix(y, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)

                    # ROC
                    if hasattr(loaded_model, 'predict_proba'):
                        try:
                            st.subheader('ROC Curve (Uploaded model)')
                            fpr, tpr, _ = roc_curve(y_test if 'y_test' in locals() else y, y_pred_prob)
                            roc_auc = roc_auc_score(y_test if 'y_test' in locals() else y, y_pred_prob)
                            fig2, ax2 = plt.subplots()
                            ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
                            ax2.plot([0,1],[0,1],'--', color='gray')
                            ax2.set_xlabel('False Positive Rate')
                            ax2.set_ylabel('True Positive Rate')
                            ax2.set_title('ROC Curve')
                            ax2.legend()
                            st.pyplot(fig2)
                        except Exception:
                            st.info('ROC curve could not be computed for uploaded model.')
                except Exception as e:
                    st.error(f'Failed to run predictions with the uploaded model: {e}')

        st.success("✅ Churn prediction completed successfully!")
else:
    st.info("Please upload a CSV file to start.")
