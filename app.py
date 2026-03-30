import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ML Workbench",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS  – clean, light, professional
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Force light mode ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stSidebar"], .main {
    background-color: #f7f8fc !important;
    color: #1a1a2e !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e5ef;
    min-width: 230px !important;
    max-width: 230px !important;
}
[data-testid="stSidebar"] h1 {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #1a1a2e;
}

/* ── Step badges in sidebar ── */
.step-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 0.88rem;
    font-weight: 500;
    transition: background 0.2s;
}
.step-badge.active   { background: #e8f0fe; color: #1a56db; border-left: 3px solid #1a56db; }
.step-badge.done     { background: #f0fdf4; color: #15803d; border-left: 3px solid #22c55e; }
.step-badge.locked   { background: #f1f5f9; color: #94a3b8; border-left: 3px solid #cbd5e1; }

/* ── Section cards ── */
.card {
    background: #ffffff;
    border: 1px solid #e2e5ef;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

/* ── Dimension pill ── */
.dim-pill {
    display: inline-block;
    background: #e8f0fe;
    color: #1a56db;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 2px 4px;
}

/* ── Table header ── */
thead tr th {
    background: #f1f5f9 !important;
    font-weight: 600 !important;
    color: #374151 !important;
}

/* ── Buttons ── */
div.stButton > button {
    background: #1a56db;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.45rem 1.4rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    transition: background 0.2s;
}
div.stButton > button:hover { background: #1648c0; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #f8faff;
    border: 1px solid #dde3f0;
    border-radius: 10px;
    padding: 12px 16px;
}

/* ── Expander ── */
details { border: 1px solid #e2e5ef !important; border-radius: 8px !important; }

/* ── Alerts ── */
.stAlert { border-radius: 8px !important; }

/* ── Headings ── */
h2 { font-size: 1.25rem !important; font-weight: 700 !important; color: #1a1a2e !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #374151 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "step": 1,
    "raw_df": None,
    "confirmed_df": None,
    "type_map": {},          # col -> "Numerical" | "Categorical"
    "step1_done": False,
    "step2_done": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def infer_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "Numerical"
    return "Categorical"


def apply_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    df = df.copy()
    for col, typ in type_map.items():
        if typ == "Numerical":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(str)
    return df


def encode_for_model(df: pd.DataFrame, type_map: dict, qoi: str):
    """One-hot encode categoricals (excl. QOI), label-encode binary QOI."""
    df = df.copy()
    cat_cols = [c for c, t in type_map.items() if t == "Categorical" and c != qoi]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def is_binary_categorical(series: pd.Series, type_map: dict, col: str) -> bool:
    return type_map.get(col) == "Categorical" and series.nunique() == 2


def sidebar_steps():
    with st.sidebar:
        st.markdown("### ⚗️ ML Workbench")
        st.markdown("---")
        steps = [
            (1, "Load Data"),
            (2, "Variable Types"),
            (3, "Model & Compare"),
        ]
        for num, label in steps:
            if num == st.session_state.step:
                cls = "active"
            elif (num == 2 and st.session_state.step1_done) or \
                 (num == 3 and st.session_state.step2_done) or \
                 num < st.session_state.step:
                cls = "done"
            else:
                cls = "locked"
            icon = {"active": "▶", "done": "✓", "locked": "🔒"}[cls]
            st.markdown(
                f'<div class="step-badge {cls}">{icon} &nbsp; Step {num} — {label}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.caption("Steps unlock sequentially.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def step1():
    st.markdown("## Step 1 — Load Data")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload an Excel file (.xlsx, single sheet)",
        type=["xlsx"],
        help="Only .xlsx files with exactly one sheet are supported.",
    )

    if uploaded:
        try:
            df = pd.read_excel(BytesIO(uploaded.read()), sheet_name=0)
            st.session_state.raw_df = df

            st.success(f"**{uploaded.name}** loaded successfully.")
            col1, col2 = st.columns(2)
            col1.markdown(
                f'<div style="margin-top:10px">Rows &nbsp;<span class="dim-pill">{df.shape[0]:,}</span></div>',
                unsafe_allow_html=True,
            )
            col2.markdown(
                f'<div style="margin-top:10px">Columns &nbsp;<span class="dim-pill">{df.shape[1]}</span></div>',
                unsafe_allow_html=True,
            )

            st.markdown("**Preview (first 5 rows):**")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("✅ Confirm & Proceed to Step 2"):
                st.session_state.step1_done = True
                st.session_state.step = 2
                # initialise type map from raw data
                st.session_state.type_map = {
                    col: infer_type(df[col]) for col in df.columns
                }
                st.rerun()

        except Exception as e:
            st.error(f"Could not read file: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – VARIABLE TYPES
# ══════════════════════════════════════════════════════════════════════════════
def step2():
    st.markdown("## Step 2 — Variable Types")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        "Review the inferred data types below. Use the **Override** column to correct "
        "any misclassification, then click **Confirm Types**."
    )

    df = st.session_state.raw_df
    type_map = st.session_state.type_map.copy()

    # Build editable table via columns
    header_cols = st.columns([3, 2, 2])
    header_cols[0].markdown("**Variable**")
    header_cols[1].markdown("**Inferred Type**")
    header_cols[2].markdown("**Override**")
    st.markdown('<hr style="margin:4px 0 10px 0; border-color:#e2e5ef">', unsafe_allow_html=True)

    new_type_map = {}
    for col in df.columns:
        row = st.columns([3, 2, 2])
        row[0].markdown(f"`{col}`")
        inferred = type_map[col]
        row[1].markdown(
            f'<span style="color:{"#1a56db" if inferred=="Numerical" else "#7c3aed"};'
            f'font-weight:600">{inferred}</span>',
            unsafe_allow_html=True,
        )
        options = ["Numerical", "Categorical"]
        override = row[2].selectbox(
            label="",
            options=options,
            index=options.index(inferred),
            key=f"type_{col}",
            label_visibility="collapsed",
        )
        new_type_map[col] = override

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("✅ Confirm Types & Proceed to Step 3"):
        st.session_state.type_map = new_type_map
        confirmed = apply_types(df, new_type_map)
        st.session_state.confirmed_df = confirmed
        st.session_state.step2_done = True
        st.session_state.step = 3
        st.rerun()

    # Show back button
    if st.button("← Back to Step 1"):
        st.session_state.step = 1
        st.session_state.step1_done = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – MODEL & COMPARE
# ══════════════════════════════════════════════════════════════════════════════
def step3():
    st.markdown("## Step 3 — Define QOI & Build Models")

    df = st.session_state.confirmed_df
    type_map = st.session_state.type_map

    # ── QOI Selection ────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Quantity of Interest (QOI)")

    eligible = []
    for col in df.columns:
        t = type_map[col]
        if t == "Numerical":
            eligible.append(col)
        elif t == "Categorical" and df[col].nunique() == 2:
            eligible.append(col)
        # non-binary categoricals excluded silently

    if not eligible:
        st.error("No eligible QOI columns found. Need at least one numerical or binary categorical column.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    qoi = st.selectbox(
        "Select QOI (target variable)",
        options=eligible,
        help="Non-binary categorical columns are excluded. Numerical → regression models; Binary categorical → classification models.",
    )

    task = "regression" if type_map[qoi] == "Numerical" else "classification"
    label_col = {
        "regression": "🔢 Regression task",
        "classification": "🔵 Binary Classification task",
    }[task]
    st.info(f"**Detected task:** {label_col}  \n"
            f"Models: {'OLS · Neural Network · Random Forest' if task == 'regression' else 'Logistic Regression · Neural Network · Random Forest'}")

    # ── Train-test split ──────────────────────────────────────────────────────
    train_pct = st.slider(
        "Training set size (%)",
        min_value=50, max_value=95, value=80, step=5,
        help="Remaining data used for testing.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Prepare data ──────────────────────────────────────────────────────────
    df_enc = encode_for_model(df, type_map, qoi)

    # Drop rows with NaN in QOI
    df_enc = df_enc.dropna(subset=[qoi])

    if task == "classification":
        le = LabelEncoder()
        df_enc[qoi] = le.fit_transform(df_enc[qoi].astype(str))

    feature_cols = [c for c in df_enc.columns if c != qoi]
    X = df_enc[feature_cols].copy()
    y = df_enc[qoi].copy()

    # Drop rows with NaN in features
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, train_size=train_pct / 100, random_state=42
    )

    # ── Model sections ────────────────────────────────────────────────────────
    if "model_results" not in st.session_state:
        st.session_state.model_results = {}

    model_results = st.session_state.model_results

    # ── Helper: run & store result ────────────────────────────────────────────
    def run_model(name, model):
        model.fit(X_tr, y_tr)
        if task == "regression":
            tr_pred = model.predict(X_tr)
            te_pred = model.predict(X_te)
            return {
                "name": name,
                "train_mse": mean_squared_error(y_tr, tr_pred),
                "test_mse": mean_squared_error(y_te, te_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_tr, tr_pred)),
                "test_rmse": np.sqrt(mean_squared_error(y_te, te_pred)),
            }
        else:
            tr_pred = model.predict(X_tr)
            te_pred = model.predict(X_te)
            return {
                "name": name,
                "train_acc": accuracy_score(y_tr, tr_pred),
                "test_acc": accuracy_score(y_te, te_pred),
                "train_f1": f1_score(y_tr, tr_pred, zero_division=0),
                "test_f1": f1_score(y_te, te_pred, zero_division=0),
                "cm_train": confusion_matrix(y_tr, tr_pred),
                "cm_test": confusion_matrix(y_te, te_pred),
                "model": model,
                "le": le,
            }

    # ══ MODEL 1 ══════════════════════════════════════════════════════════════
    with st.expander("**Model 1 — " + ("OLS Linear Regression" if task == "regression" else "Logistic Regression (GLM)") + "**", expanded=True):
        m1_name = "OLS" if task == "regression" else "Logistic Regression"

        if task == "regression":
            st.markdown("*OLS has no tunable hyperparameters — it fits analytically.*")
        else:
            col_a, col_b = st.columns(2)
            m1_C = col_a.number_input(
                "Regularisation C",
                min_value=0.001, max_value=100.0, value=1.0, step=0.1,
                help="Inverse of regularisation strength. Smaller → stronger regularisation. Range: 0.001 – 100.",
                key="m1_C",
            )
            m1_max_iter = col_b.number_input(
                "Max iterations",
                min_value=100, max_value=10000, value=1000, step=100,
                help="Maximum solver iterations. Range: 100 – 10 000.",
                key="m1_iter",
            )

        if st.button(f"▶ Run {m1_name}", key="run_m1"):
            with st.spinner("Fitting..."):
                if task == "regression":
                    mdl = LinearRegression()
                else:
                    mdl = LogisticRegression(C=m1_C, max_iter=int(m1_max_iter), random_state=42)
                model_results["m1"] = run_model(m1_name, mdl)
            st.success("Done!")

        if "m1" in model_results:
            r = model_results["m1"]
            if task == "regression":
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Train MSE", f"{r['train_mse']:.4f}")
                c2.metric("Test MSE",  f"{r['test_mse']:.4f}")
                c3.metric("Train RMSE", f"{r['train_rmse']:.4f}")
                c4.metric("Test RMSE",  f"{r['test_rmse']:.4f}")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Train Accuracy", f"{r['train_acc']:.4f}")
                c2.metric("Test Accuracy",  f"{r['test_acc']:.4f}")
                c3.metric("Train F1", f"{r['train_f1']:.4f}")
                c4.metric("Test F1",  f"{r['test_f1']:.4f}")

    # ══ MODEL 2 ══════════════════════════════════════════════════════════════
    with st.expander("**Model 2 — Neural Network (MLP)**", expanded=True):
        m2_name = "Neural Network (MLP)"
        col_a, col_b, col_c = st.columns(3)
        m2_layers = col_a.text_input(
            "Hidden layer sizes (comma-separated)",
            value="100,50",
            help="E.g. '100,50' → two hidden layers of 100 and 50 neurons. Each value: 1 – 1 000.",
            key="m2_layers",
        )
        m2_alpha = col_b.number_input(
            "L2 regularisation (alpha)",
            min_value=1e-6, max_value=10.0, value=1e-4, format="%.6f",
            help="L2 penalty term. Range: 0.000001 – 10.",
            key="m2_alpha",
        )
        m2_max_iter = col_c.number_input(
            "Max iterations",
            min_value=50, max_value=2000, value=500, step=50,
            help="Max training epochs. Range: 50 – 2 000.",
            key="m2_iter",
        )

        if st.button("▶ Run Neural Network", key="run_m2"):
            try:
                layers = tuple(int(x.strip()) for x in m2_layers.split(",") if x.strip())
                assert all(1 <= n <= 1000 for n in layers), "Each layer size must be 1–1000."
                with st.spinner("Fitting..."):
                    if task == "regression":
                        mdl = MLPRegressor(
                            hidden_layer_sizes=layers,
                            alpha=m2_alpha,
                            max_iter=int(m2_max_iter),
                            random_state=42,
                        )
                    else:
                        mdl = MLPClassifier(
                            hidden_layer_sizes=layers,
                            alpha=m2_alpha,
                            max_iter=int(m2_max_iter),
                            random_state=42,
                        )
                    model_results["m2"] = run_model(m2_name, mdl)
                st.success("Done!")
            except Exception as e:
                st.error(f"Invalid parameters: {e}")

        if "m2" in model_results:
            r = model_results["m2"]
            if task == "regression":
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Train MSE", f"{r['train_mse']:.4f}")
                c2.metric("Test MSE",  f"{r['test_mse']:.4f}")
                c3.metric("Train RMSE", f"{r['train_rmse']:.4f}")
                c4.metric("Test RMSE",  f"{r['test_rmse']:.4f}")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Train Accuracy", f"{r['train_acc']:.4f}")
                c2.metric("Test Accuracy",  f"{r['test_acc']:.4f}")
                c3.metric("Train F1", f"{r['train_f1']:.4f}")
                c4.metric("Test F1",  f"{r['test_f1']:.4f}")

    # ══ MODEL 3 ══════════════════════════════════════════════════════════════
    with st.expander("**Model 3 — Random Forest**", expanded=True):
        m3_name = "Random Forest"
        col_a, col_b, col_c = st.columns(3)
        m3_n = col_a.number_input(
            "Number of trees",
            min_value=10, max_value=1000, value=100, step=10,
            help="More trees → more stable but slower. Range: 10 – 1 000.",
            key="m3_n",
        )
        m3_depth = col_b.number_input(
            "Max depth (0 = unlimited)",
            min_value=0, max_value=100, value=0, step=1,
            help="Maximum depth of each tree. 0 means nodes expand until leaves are pure. Range: 0 – 100.",
            key="m3_depth",
        )
        m3_min_split = col_c.number_input(
            "Min samples to split",
            min_value=2, max_value=100, value=2, step=1,
            help="Minimum samples required to split an internal node. Range: 2 – 100.",
            key="m3_split",
        )

        if st.button("▶ Run Random Forest", key="run_m3"):
            with st.spinner("Fitting..."):
                depth = None if m3_depth == 0 else int(m3_depth)
                if task == "regression":
                    mdl = RandomForestRegressor(
                        n_estimators=int(m3_n),
                        max_depth=depth,
                        min_samples_split=int(m3_min_split),
                        random_state=42,
                    )
                else:
                    mdl = RandomForestClassifier(
                        n_estimators=int(m3_n),
                        max_depth=depth,
                        min_samples_split=int(m3_min_split),
                        random_state=42,
                    )
                model_results["m3"] = run_model(m3_name, mdl)
            st.success("Done!")

        if "m3" in model_results:
            r = model_results["m3"]
            if task == "regression":
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Train MSE", f"{r['train_mse']:.4f}")
                c2.metric("Test MSE",  f"{r['test_mse']:.4f}")
                c3.metric("Train RMSE", f"{r['train_rmse']:.4f}")
                c4.metric("Test RMSE",  f"{r['test_rmse']:.4f}")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Train Accuracy", f"{r['train_acc']:.4f}")
                c2.metric("Test Accuracy",  f"{r['test_acc']:.4f}")
                c3.metric("Train F1", f"{r['train_f1']:.4f}")
                c4.metric("Test F1",  f"{r['test_f1']:.4f}")

    # ══ COMBINED SUMMARY TABLE ═════════════════════════════════════════════
    run_keys = [k for k in ["m1", "m2", "m3"] if k in model_results]
    if run_keys:
        st.markdown("---")
        st.markdown("### 📊 Combined Model Comparison")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if task == "regression":
            rows = []
            for k in run_keys:
                r = model_results[k]
                rows.append({
                    "Model": r["name"],
                    "Train MSE": round(r["train_mse"], 6),
                    "Test MSE": round(r["test_mse"], 6),
                    "Train RMSE": round(r["train_rmse"], 6),
                    "Test RMSE": round(r["test_rmse"], 6),
                })
            summary = pd.DataFrame(rows).set_index("Model")

            # Highlight best (lowest) test MSE
            def highlight_min(s):
                is_min = s == s.min()
                return ["background-color: #dcfce7; color: #15803d; font-weight:700"
                        if v else "" for v in is_min]

            styled = summary.style.apply(highlight_min, subset=["Test MSE", "Test RMSE"])
            st.dataframe(styled, use_container_width=True)
            st.caption("🟢 Green = best (lowest) value in column.")

        else:
            # Classification: summary table + confusion matrices
            rows = []
            for k in run_keys:
                r = model_results[k]
                rows.append({
                    "Model": r["name"],
                    "Train Accuracy": round(r["train_acc"], 4),
                    "Test Accuracy": round(r["test_acc"], 4),
                    "Train F1": round(r["train_f1"], 4),
                    "Test F1": round(r["test_f1"], 4),
                })
            summary = pd.DataFrame(rows).set_index("Model")

            def highlight_max(s):
                is_max = s == s.max()
                return ["background-color: #dcfce7; color: #15803d; font-weight:700"
                        if v else "" for v in is_max]

            styled = summary.style.apply(
                highlight_max, subset=["Test Accuracy", "Test F1"]
            )
            st.dataframe(styled, use_container_width=True)
            st.caption("🟢 Green = best (highest) value in column.")

            # Confusion matrices side by side
            st.markdown("**Confusion Matrices (Test Set)**")
            cm_cols = st.columns(len(run_keys))
            for i, k in enumerate(run_keys):
                r = model_results[k]
                fig, ax = plt.subplots(figsize=(3.2, 2.8))
                fig.patch.set_facecolor("#ffffff")
                ax.set_facecolor("#ffffff")
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=r["cm_test"],
                    display_labels=r["le"].classes_,
                )
                disp.plot(ax=ax, colorbar=False, cmap="Blues")
                ax.set_title(r["name"], fontsize=9, fontweight="bold", color="#1a1a2e")
                ax.tick_params(labelsize=8)
                plt.tight_layout()
                cm_cols[i].pyplot(fig)
                plt.close(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    # Back button
    st.markdown("---")
    if st.button("← Back to Step 2"):
        st.session_state.step = 2
        st.session_state.step2_done = False
        st.session_state.model_results = {}
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    sidebar_steps()

    step = st.session_state.step
    if step == 1:
        step1()
    elif step == 2:
        if st.session_state.step1_done:
            step2()
        else:
            st.warning("Please complete Step 1 first.")
    elif step == 3:
        if st.session_state.step2_done:
            step3()
        else:
            st.warning("Please complete Step 2 first.")


if __name__ == "__main__":
    main()
