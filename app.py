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
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
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

# Optional SHAP for explainability
try:
    import shap
    shap_installed = True
except Exception:
    shap_installed = False

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

# Deployment note: ensure safe_roc_auc is a top-level function (fix for NameError seen in some deployments)


def safe_f1(y_true, y_pred):
    """Compute F1 robustly. Falls back to macro/weighted average and zero_division handling when binary fails."""
    try:
        # try binary with safe zero_division
        return f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        try:
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        except Exception:
            try:
                return f1_score(y_true, y_pred, average='weighted', zero_division=0)
            except Exception:
                return 0.0


        def safe_roc_auc(y_true, y_score):
            """Compute ROC AUC safely for binary and multiclass when possible.
            Returns np.nan when ROC-AUC is not applicable or fails.
            - For binary targets expects a 1d score (probabilities) or a 2D proba matrix (takes column 1).
            - For multiclass expects a 2D probability matrix shaped (n_samples, n_classes).
            """
            try:
                y_true_arr = np.array(y_true)
                y_score_arr = np.array(y_score)
                # remove NaNs in y_true
                mask = ~pd.isnull(y_true_arr)
                y_true_arr = y_true_arr[mask]
                y_score_arr = y_score_arr[mask]
                uniq = np.unique(y_true_arr)
                if len(uniq) <= 1:
                    return np.nan
                # binary
                if len(uniq) == 2:
                    # if score is 2D, try to take the positive class column
                    if y_score_arr.ndim == 1:
                        return roc_auc_score(y_true_arr, y_score_arr)
                    elif y_score_arr.ndim == 2 and y_score_arr.shape[1] >= 2:
                        return roc_auc_score(y_true_arr, y_score_arr[:, 1])
                    else:
                        return np.nan
                # multiclass
                if len(uniq) > 2:
                    if y_score_arr.ndim == 2 and y_score_arr.shape[1] == len(uniq):
                        return roc_auc_score(y_true_arr, y_score_arr, multi_class='ovr', average='macro')
                    else:
                        return np.nan
            except Exception:
                return np.nan


def generate_business_insights(data: pd.DataFrame, target_col: str, best_model=None, features_list=None, contract_col_override=None, monthly_col_override=None):
    """Generate plain-language business insights based on data and model feature importances.
    Returns a dict with messages and recommended actions."""
    insights = {}
    y_raw = data[target_col]

    # infer churn mask
    try:
        if pd.api.types.is_numeric_dtype(y_raw):
            uniq = sorted(y_raw.dropna().unique())
            if set(uniq) <= {0, 1}:
                churn_mask = (y_raw == 1)
            else:
                # fallback: less frequent value considered churn
                churn_mask = (y_raw != y_raw.mode()[0])
        else:
            low = y_raw.astype(str).str.lower()
            if (low == 'yes').any() or (low == 'y').any() or (low == 'true').any():
                churn_mask = low.isin(['yes', 'y', 'true'])
            else:
                # assume minority class is churn
                churn_mask = (y_raw != y_raw.mode()[0])
    except Exception:
        churn_mask = pd.Series(False, index=y_raw.index)

    # Contract churn comparison (allow override)
    contract_col = contract_col_override if (contract_col_override in data.columns if contract_col_override is not None else False) else None
    if contract_col is None:
        for cand in ['Contract', 'contract', 'CONTRACT', 'ContractType']:
            if cand in data.columns:
                contract_col = cand
                break
    if contract_col is not None:
        grp = data.groupby(contract_col)[target_col].apply(lambda s: ((s.astype(str).str.lower().isin(['yes','y','true']) if s.dtype == object else (s==1)) if True else s))
        # compute rates using churn_mask
        rates = data.groupby(contract_col).apply(lambda df: churn_mask.loc[df.index].mean())
        # try to find month-to-month vs yearly
        month_rate = None
        yearly_rate = None
        for k in rates.index:
            if 'month' in str(k).lower():
                month_rate = rates[k]
        yearly_vals = [k for k in rates.index if ('year' in str(k).lower() or 'one' in str(k).lower() or 'two' in str(k).lower())]
        if yearly_vals:
            yearly_rate = rates.loc[yearly_vals].mean()
        if month_rate is not None and yearly_rate is not None and yearly_rate > 0:
            multiplier = month_rate / yearly_rate
            insights['contract'] = f"Customers on month-to-month contracts churn {multiplier:.2f}× more than yearly contracts (month-to-month: {month_rate:.1%}, yearly avg: {yearly_rate:.1%})."
        else:
            # show top and bottom
            sr = rates.sort_values(ascending=False)
            insights['contract'] = f"Churn rates by {contract_col}: " + ', '.join([f"{idx}: {val:.1%}" for idx,val in sr.items()])
    else:
        insights['contract'] = 'Contract-type column not found; cannot compute contract-based churn insight.'

    # Monthly charges effect (allow override)
    monthly_col = monthly_col_override if (monthly_col_override in data.columns if monthly_col_override is not None else False) else None
    if monthly_col is None:
        for cand in ['MonthlyCharges', 'Monthly Charge', 'Monthly', 'monthlycharges']:
            if cand in data.columns:
                monthly_col = cand
                break
    if monthly_col is not None:
        try:
            mean_churn = data.loc[churn_mask, monthly_col].astype(float).mean()
            mean_keep = data.loc[~churn_mask, monthly_col].astype(float).mean()
            if pd.notna(mean_churn) and pd.notna(mean_keep) and mean_keep > 0:
                pct = (mean_churn - mean_keep) / mean_keep * 100
                insights['monthly'] = f"High monthly charges increase churn probability: churners pay on average ${mean_churn:,.2f} vs ${mean_keep:,.2f} for non-churners ({pct:.0f}% higher)."
            else:
                insights['monthly'] = 'Could not compute monthly charge comparison (missing or non-numeric).'
        except Exception:
            insights['monthly'] = 'Could not compute monthly charge comparison (error).'
    else:
        insights['monthly'] = 'Monthly charge column not found.'

    # Top features from model
    if best_model is not None and features_list is not None:
        try:
            if hasattr(best_model, 'feature_importances_'):
                imp = best_model.feature_importances_
                fi = pd.DataFrame({'feature': features_list, 'importance': imp}).sort_values('importance', ascending=False)
                top = fi.head(5)
                insights['top_features'] = 'Top features linked to churn: ' + ', '.join([f"{r['feature']}" for _, r in top.iterrows()])
            else:
                insights['top_features'] = 'Model does not expose feature_importances_; consider using permutation importance or SHAP for more insights.'
        except Exception:
            insights['top_features'] = 'Could not compute feature importance.'
    else:
        insights['top_features'] = 'No model feature importances available.'

    # Actionable steps
    actions = []
    actions.append('Target high-risk customers (identified by the model) with retention offers and personalized incentives, especially month-to-month customers.')
    actions.append('Consider pricing review or discounts for customers with high monthly charges; offer multi-month discounts or bundled services to reduce churn.')
    actions.append('Investigate the top features above for product or service improvements (e.g., reduce complaint types, simplify billing, improve service reliability).')
    insights['actions'] = actions

    # Return insights
    return insights

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"]) 

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Organize app into tabs: Data, EDA, Modeling
    tab_data, tab_eda, tab_model = st.tabs(["Data", "EDA (Plotly)", "Modeling"])

    # --- Data tab ---
    with tab_data:
        st.subheader("Dataset Preview")
        st.write(f"Dataset size: {len(data):,} rows × {data.shape[1]} columns")
        st.dataframe(data.head())
        # Let user pick which columns represent contract / monthly charges
        st.markdown('---')
        st.subheader('Column mapping for business insights')
        col_options = [None] + list(data.columns)
        contract_col_sel = st.selectbox('Select contract column (for business insight)', col_options, index=0)
        monthly_col_sel = st.selectbox('Select monthly charge column (for business insight)', col_options, index=0)
        st.markdown("**Dataset Info**")
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())
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
            if plotly_installed:
                fig_td = px.histogram(df, x=target_column, title=f'Distribution of {target_column}')
                st.plotly_chart(fig_td, use_container_width=True)
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

        # Check target distribution and warn if many unique values (possible regression label)
        allow_modeling = True
        try:
            unique_vals = y.dropna().unique()
            n_unique = len(unique_vals)
            n_samples = len(y.dropna())
            suspicious = False
            # heuristics: if unique classes > 10 or unique classes > 50% of samples -> suspicious
            if n_unique > 10 or (n_samples > 0 and (n_unique / n_samples) > 0.5):
                suspicious = True
            if suspicious:
                st.warning(f"Target column appears to have {n_unique} unique values across {n_samples} non-null samples. This may be a regression target rather than classification. Many classification metrics (ROC-AUC, confusion matrix) may be misleading.")
                allow_modeling = st.checkbox('I understand and want to proceed with classification metrics on this target (only proceed if intended)', value=False)
        except Exception:
            allow_modeling = True

    # --- Preprocessing options ---
        st.markdown('#### Preprocessing options')
        impute_method = st.selectbox('Missing value imputation', ['None', 'Mean', 'Median', 'Most frequent'], index=0)
        encoding_method = st.selectbox('Categorical encoding', ['LabelEncoder', 'OneHot'], index=0)
        scaling_method = st.selectbox('Scaling', ['Standard', 'None'], index=0)
        class_balance = st.selectbox('Class balancing (after split)', ['None', 'Upsample minority', 'Downsample majority'], index=0)

        # Encode target if categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))

        # Apply imputation if requested
        X_proc = X.copy()
        if impute_method != 'None':
            strategy = 'mean' if impute_method == 'Mean' else ('median' if impute_method == 'Median' else 'most_frequent')
            imputer = SimpleImputer(strategy=strategy)
            # Only apply to numeric columns
            num_cols = X_proc.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                X_proc[num_cols] = imputer.fit_transform(X_proc[num_cols])

        # Encode categorical features
        if encoding_method == 'LabelEncoder':
            for col in X_proc.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X_proc[col] = le.fit_transform(X_proc[col].astype(str))
        else:
            # One-hot encode and avoid creating too many columns
            cat_cols = X_proc.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                X_proc = pd.get_dummies(X_proc, columns=cat_cols, drop_first=True)

        # Optional: Scale features (we'll fit if training locally or use uploaded preprocessor)
        scaler = StandardScaler()
        if scaling_method == 'Standard':
            try:
                X_scaled = scaler.fit_transform(X_proc)
            except Exception:
                # fallback to original
                X_scaled = scaler.fit_transform(X_proc.fillna(0))
        else:
            X_scaled = X_proc.values

        # Step 4: Train-test split
        test_size = st.slider("Select test size (percentage)", min_value=10, max_value=50, value=20, key='test_size')
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size/100, random_state=42)

        # Step 5: Model source selection: train locally or upload pretrained
        st.markdown('### Model source')
        if not allow_modeling:
            st.info('Modeling disabled due to suspicious target distribution. Check the confirmation box above to proceed with modeling.')
            model_mode = None
        else:
            model_mode = st.radio('Choose how you want to get a model', ['Train locally', 'Upload pretrained model'], index=0)

        results = []
        model_objects = {}

        if model_mode == 'Train locally':
            st.markdown('### Training Multiple Models')
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                # removed deprecated use_label_encoder param to avoid warnings
                "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42)
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
                f1 = safe_f1(y_test, y_pred)
                # compute ROC-AUC safely
                roc = safe_roc_auc(y_test, y_pred_prob)
                results.append([name, acc, f1, roc])
                model_objects[name] = model

            results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1-score", "ROC-AUC"]).sort_values(by='ROC-AUC', ascending=False)
            st.subheader("Model Performance Comparison")
            metric = st.selectbox('Metric for comparison', ['ROC-AUC','Accuracy','F1-score'], index=0)
            if plotly_installed and not results_df.empty:
                fig_cmp = px.bar(results_df, x=metric, y='Model', orientation='h', text=metric, title=f'Model comparison ({metric})')
                fig_cmp.update_layout(height=min(400, 100 + len(results_df)*40))
                st.plotly_chart(fig_cmp, use_container_width=True)
            st.dataframe(results_df)

            # Multi-model ROC overlay
            if plotly_installed and model_objects:
                show_multi_roc = st.checkbox('Show multi-model ROC overlay', value=False)
                if show_multi_roc:
                    fig_multi = go.Figure()
                    for name, mdl in model_objects.items():
                        try:
                            if hasattr(mdl, 'predict_proba'):
                                probs_all = mdl.predict_proba(X_test)
                                # attempt safe AUC
                                auc_m = safe_roc_auc(y_test, probs_all)
                                if np.isnan(auc_m):
                                    continue
                                # for binary, take positive column for curve
                                if probs_all.ndim == 1 or (probs_all.ndim == 2 and probs_all.shape[1] == 1):
                                    probs = probs_all
                                else:
                                    probs = probs_all[:, 1]
                                fpr_m, tpr_m, _ = roc_curve(y_test, probs)
                                fig_multi.add_trace(go.Scatter(x=fpr_m, y=tpr_m, mode='lines', name=f"{name} (AUC {auc_m:.3f})"))
                        except Exception:
                            # skip models without probability or failing
                            continue
                    fig_multi.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), showlegend=False))
                    fig_multi.update_layout(title='Multi-model ROC overlay', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=500)
                    st.plotly_chart(fig_multi, use_container_width=True)

            if not results_df.empty:
                best_model_name = results_df.iloc[0]["Model"]
                st.success(f"Best Model (by ROC-AUC): {best_model_name}")
                best_model = model_objects[best_model_name]

                # Interactive threshold and Confusion Matrix
                st.subheader(f"Confusion Matrix & Threshold ({best_model_name})")
                # If model supports predict_proba, let user pick a threshold
                if hasattr(best_model, 'predict_proba'):
                    y_pred_prob_best = best_model.predict_proba(X_test)[:,1]
                    threshold = st.slider('Prediction threshold', 0.0, 1.0, 0.5, step=0.01)
                    y_pred_best = (y_pred_prob_best >= threshold).astype(int)
                else:
                    y_pred_best = best_model.predict(X_test)
                classes = np.unique(y_test)
                cm = confusion_matrix(y_test, y_pred_best, labels=classes)
                if plotly_installed:
                    fig_cm = px.imshow(cm, x=[str(c) for c in classes], y=[str(c) for c in classes], color_continuous_scale='Blues', text_auto=True)
                    fig_cm.update_layout(title=f'Confusion Matrix ({best_model_name})', xaxis_title='Predicted', yaxis_title='Actual', height=min(600, 200 + len(data)//10))
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

                # show derived metrics at chosen threshold
                try:
                    acc_t = accuracy_score(y_test, y_pred_best)
                    f1_t = safe_f1(y_test, y_pred_best)
                    st.write(f"Accuracy: {acc_t:.4f} — F1-score: {f1_t:.4f}")
                except Exception:
                    pass

                # ROC Curve
                try:
                    st.subheader(f"ROC Curve ({best_model_name})")
                    if hasattr(best_model, "predict_proba"):
                        probs_all_best = best_model.predict_proba(X_test)
                    else:
                        probs_all_best = None
                    auc_val = safe_roc_auc(y_test, probs_all_best if probs_all_best is not None else y_pred_best)
                    if np.isnan(auc_val):
                        st.info('ROC-AUC not applicable for this target/problem (maybe multiclass without proper probability matrix or single-class).')
                    else:
                        # get binary score for curve plotting
                        if probs_all_best is not None and probs_all_best.ndim == 2 and probs_all_best.shape[1] >= 2:
                            scores_for_curve = probs_all_best[:, 1]
                        elif probs_all_best is not None and probs_all_best.ndim == 1:
                            scores_for_curve = probs_all_best
                        else:
                            scores_for_curve = y_pred_best
                        fpr, tpr, _ = roc_curve(y_test, scores_for_curve)
                        roc_auc = auc_val
                        if plotly_installed:
                            fig_roc = go.Figure()
                            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.4f}'))
                            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), showlegend=False))
                            fig_roc.update_layout(title=f'ROC Curve ({best_model_name})', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=min(500, 200 + len(data)//10))
                            st.plotly_chart(fig_roc, use_container_width=True)
                        else:
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
                    if plotly_installed:
                        fig3 = px.bar(feature_imp, x='Importance', y='Feature', orientation='h', title=f'Feature importance ({best_model_name})')
                        fig3.update_layout(height=min(600, 100 + len(feature_imp)*30))
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        fig3, ax3 = plt.subplots(figsize=(10,6))
                        sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax3)
                        st.pyplot(fig3)

                # Business insights for business users
                try:
                    insights = generate_business_insights(data, target_column, best_model=best_model, features_list=features, contract_col_override=contract_col_sel, monthly_col_override=monthly_col_sel)
                    st.markdown('---')
                    st.header('Business insights & suggested actions')
                    st.write(insights.get('contract'))
                    st.write(insights.get('monthly'))
                    st.write(insights.get('top_features'))
                    st.subheader('Recommended actions')
                    for a in insights.get('actions', []):
                        st.write('- ' + a)
                except Exception as e:
                    st.info('Could not generate business insights: ' + str(e))

                # Export insights as text file
                try:
                    insights_txt = '\n'.join([insights.get('contract',''), insights.get('monthly',''), insights.get('top_features',''), '\nRecommended actions:'] + insights.get('actions', []))
                    st.download_button('Export insights (txt)', insights_txt, file_name='business_insights.txt')
                except Exception:
                    pass

                # SHAP explainability for tree models
                if shap_installed and best_model_name in ['Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']:
                    st.markdown('---')
                    st.subheader('SHAP explanations (global)')
                    try:
                        # compute shap values for a subset
                        explainer = shap.TreeExplainer(best_model)
                        sample = pd.DataFrame(X_test).astype(float)
                        sample_small = sample.sample(n=min(200, sample.shape[0]), random_state=42)
                        shap_values = explainer.shap_values(sample_small)
                        st.pyplot(shap.summary_plot(shap_values, sample_small, show=False))
                    except Exception as e:
                        st.info('Could not compute SHAP explanations: ' + str(e))

                # Download trained model
                st.markdown('---')
                st.markdown('**Download trained model**')
                buf = io.BytesIO()
                try:
                    if joblib_installed:
                        joblib.dump(best_model, buf)
                        buf.seek(0)
                        st.download_button('Download model (joblib)', buf, file_name='best_model.joblib')
                    else:
                        pickle.dump(best_model, buf)
                        buf.seek(0)
                        st.download_button('Download model (pickle)', buf, file_name='best_model.pkl')
                except Exception as e:
                    st.error(f'Could not prepare model for download: {e}')

                # Single-row prediction UI
                st.markdown('---')
                st.subheader('Predict with the selected model (single row)')
                with st.form('single_predict'):
                    input_values = {}
                    for col in X.columns:
                        if pd.api.types.is_numeric_dtype(data[col]):
                            v = st.number_input(f'{col}', value=float(data[col].dropna().median()) if not data[col].dropna().empty else 0.0)
                        else:
                            opts = list(data[col].dropna().unique())[:200]
                            v = st.selectbox(f'{col}', options=opts)
                        input_values[col] = v
                    submitted = st.form_submit_button('Predict')
                    if submitted:
                        single_df = pd.DataFrame([input_values])
                        # Apply same preprocessing steps
                        single_proc = single_df.copy()
                        if impute_method != 'None':
                            # only numeric
                            num_cols = single_proc.select_dtypes(include=[np.number]).columns.tolist()
                            if num_cols:
                                single_proc[num_cols] = imputer.transform(single_proc[num_cols])
                        if encoding_method == 'LabelEncoder':
                            for col in single_proc.select_dtypes(include=['object', 'category']).columns:
                                le = LabelEncoder()
                                # fit on original column values to align labels
                                le.fit(data[col].astype(str))
                                single_proc[col] = le.transform(single_proc[col].astype(str))
                        else:
                            if cat_cols:
                                single_proc = pd.get_dummies(single_proc, columns=cat_cols, drop_first=True)
                                # align columns with training
                                for c in set(X_proc.columns) - set(single_proc.columns):
                                    single_proc[c] = 0
                                single_proc = single_proc[X_proc.columns]
                        if scaling_method == 'Standard':
                            single_scaled = scaler.transform(single_proc)
                        else:
                            single_scaled = single_proc.values
                        try:
                            pred = best_model.predict(single_scaled)
                            prob = best_model.predict_proba(single_scaled)[:,1] if hasattr(best_model, 'predict_proba') else None
                            st.write('Prediction:', pred[0])
                            if prob is not None:
                                st.write('Probability:', float(prob[0]))
                        except Exception as e:
                            st.error(f'Prediction failed: {e}')

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
                    if allow_modeling:
                        acc = accuracy_score(y_test, y_pred) if 'y_test' in locals() else accuracy_score(y, y_pred)
                        f1 = safe_f1(y_test, y_pred) if 'y_test' in locals() else safe_f1(y, y_pred)
                        roc = safe_roc_auc(y_test, y_pred_prob) if 'y_test' in locals() else safe_roc_auc(y, y_pred_prob)
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
                                auc_val = safe_roc_auc(y_test if 'y_test' in locals() else y, y_pred_prob)
                                if np.isnan(auc_val):
                                    st.info('ROC-AUC not applicable for this target/problem (maybe multiclass without proper probability matrix or single-class).')
                                else:
                                    fpr, tpr, _ = roc_curve(y_test if 'y_test' in locals() else y, y_pred_prob)
                                    roc_auc = auc_val
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
                    else:
                        st.info('Target appears to be continuous or has many unique values; skipping classification metrics to avoid misleading results. If you want to force evaluation as classification, confirm above.')
                    try:
                        insights = generate_business_insights(data, target_column, best_model=loaded_model, features_list=features)
                        st.markdown('---')
                        st.header('Business insights & suggested actions (uploaded model)')
                        st.write(insights.get('contract'))
                        st.write(insights.get('monthly'))
                        st.write(insights.get('top_features'))
                        st.subheader('Recommended actions')
                        for a in insights.get('actions', []):
                            st.write('- ' + a)
                    except Exception as e:
                        st.info('Could not generate business insights for uploaded model: ' + str(e))
                except Exception as e:
                    st.error(f'Failed to run predictions with the uploaded model: {e}')

        st.success("✅ Churn prediction completed successfully!")
else:
    st.info("Please upload a CSV file to start.")
