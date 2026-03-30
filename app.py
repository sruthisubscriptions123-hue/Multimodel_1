import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import datetime
import json
import subprocess
import tempfile
import os

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ML Workbench",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"],
[data-testid="stSidebar"], .main {
    background-color: #f7f8fc !important;
    color: #1a1a2e !important;
}
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e5ef;
    min-width: 230px !important;
    max-width: 230px !important;
}
.step-badge {
    display:flex; align-items:center; gap:10px; padding:10px 14px;
    border-radius:8px; margin-bottom:6px; font-size:.88rem; font-weight:500;
}
.step-badge.active { background:#e8f0fe; color:#1a56db; border-left:3px solid #1a56db; }
.step-badge.done   { background:#f0fdf4; color:#15803d; border-left:3px solid #22c55e; }
.step-badge.locked { background:#f1f5f9; color:#94a3b8; border-left:3px solid #cbd5e1; }
.card {
    background:#ffffff; border:1px solid #e2e5ef; border-radius:12px;
    padding:24px 28px; margin-bottom:20px; box-shadow:0 1px 4px rgba(0,0,0,.04);
}
.dim-pill {
    display:inline-block; background:#e8f0fe; color:#1a56db;
    border-radius:20px; padding:4px 16px; font-size:.95rem; font-weight:600; margin:2px 4px;
}
div.stButton > button {
    background:#1a56db; color:white; border:none; border-radius:8px;
    padding:.45rem 1.4rem; font-weight:600; letter-spacing:.03em;
}
div.stButton > button:hover { background:#1648c0; }
[data-testid="stMetric"] {
    background:#f8faff; border:1px solid #dde3f0; border-radius:10px; padding:12px 16px;
}
details { border:1px solid #e2e5ef !important; border-radius:8px !important; }
.stAlert { border-radius:8px !important; }
h2 { font-size:1.25rem !important; font-weight:700 !important; color:#1a1a2e !important; }
h3 { font-size:1.05rem !important; font-weight:600 !important; color:#374151 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "step": 1,
    "raw_df": None,
    "confirmed_df": None,
    "type_map": {},
    "step1_done": False,
    "step2_done": False,
    "step3_done": False,
    "model_results": {},
    "last_qoi": None,
    "last_task": None,
    "last_train_pct": 80,
    "file_name": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def infer_type(series):
    return "Numerical" if pd.api.types.is_numeric_dtype(series) else "Categorical"

def apply_types(df, type_map):
    df = df.copy()
    for col, typ in type_map.items():
        if typ == "Numerical":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(str)
    return df

def encode_for_model(df, type_map, qoi):
    df = df.copy()
    cat_cols = [c for c, t in type_map.items() if t == "Categorical" and c != qoi]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def sidebar_steps():
    with st.sidebar:
        st.markdown("### ⚗️ ML Workbench")
        st.markdown("---")
        steps = [(1,"Load Data"),(2,"Variable Types"),(3,"Model & Compare"),(4,"Report")]
        for num, label in steps:
            if num == st.session_state.step:
                cls = "active"
            elif num < st.session_state.step:
                cls = "done"
            else:
                cls = "locked"
            icon = {"active":"▶","done":"✓","locked":"🔒"}[cls]
            st.markdown(
                f'<div class="step-badge {cls}">{icon} &nbsp; Step {num} — {label}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.caption("Steps unlock sequentially.")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def step1():
    st.markdown("## Step 1 — Load Data")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload an Excel file (.xlsx, single sheet)", type=["xlsx"])
    if uploaded:
        try:
            df = pd.read_excel(BytesIO(uploaded.read()), sheet_name=0)
            st.session_state.raw_df = df
            st.session_state.file_name = uploaded.name
            st.success(f"**{uploaded.name}** loaded successfully.")
            c1, c2 = st.columns(2)
            c1.markdown(f'<div style="margin-top:10px">Rows &nbsp;<span class="dim-pill">{df.shape[0]:,}</span></div>', unsafe_allow_html=True)
            c2.markdown(f'<div style="margin-top:10px">Columns &nbsp;<span class="dim-pill">{df.shape[1]}</span></div>', unsafe_allow_html=True)
            st.markdown("**Preview (first 5 rows):**")
            st.dataframe(df.head(), use_container_width=True)
            if st.button("Confirm & Proceed to Step 2"):
                st.session_state.step1_done = True
                st.session_state.step = 2
                st.session_state.type_map = {col: infer_type(df[col]) for col in df.columns}
                st.rerun()
        except Exception as e:
            st.error(f"Could not read file: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — VARIABLE TYPES
# ══════════════════════════════════════════════════════════════════════════════
def step2():
    st.markdown("## Step 2 — Variable Types")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("Review inferred types. Override where needed, then confirm.")
    df       = st.session_state.raw_df
    type_map = st.session_state.type_map.copy()
    hc = st.columns([3,2,2])
    hc[0].markdown("**Variable**")
    hc[1].markdown("**Inferred Type**")
    hc[2].markdown("**Override**")
    st.markdown('<hr style="margin:4px 0 10px 0;border-color:#e2e5ef">', unsafe_allow_html=True)
    new_type_map = {}
    for col in df.columns:
        row      = st.columns([3,2,2])
        inferred = type_map[col]
        row[0].markdown(f"`{col}`")
        row[1].markdown(
            f'<span style="color:{"#1a56db" if inferred=="Numerical" else "#7c3aed"};font-weight:600">{inferred}</span>',
            unsafe_allow_html=True,
        )
        opts = ["Numerical","Categorical"]
        new_type_map[col] = row[2].selectbox(
            "", opts, index=opts.index(inferred), key=f"type_{col}", label_visibility="collapsed"
        )
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("Confirm Types & Proceed to Step 3"):
        st.session_state.type_map     = new_type_map
        st.session_state.confirmed_df = apply_types(df, new_type_map)
        st.session_state.step2_done   = True
        st.session_state.step         = 3
        st.rerun()
    if st.button("Back to Step 1"):
        st.session_state.step = 1
        st.session_state.step1_done = False
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — MODEL & COMPARE
# ══════════════════════════════════════════════════════════════════════════════
def step3():
    st.markdown("## Step 3 — Define QOI & Build Models")
    df       = st.session_state.confirmed_df
    type_map = st.session_state.type_map

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Target Variable (QOI)")
    eligible = [c for c in df.columns
                if type_map[c]=="Numerical" or
                   (type_map[c]=="Categorical" and df[c].nunique()==2)]
    if not eligible:
        st.error("No eligible QOI columns found.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    qoi  = st.selectbox("Select QOI", eligible,
                        help="Non-binary categoricals excluded. Numerical → regression; Binary categorical → classification.")
    task = "regression" if type_map[qoi]=="Numerical" else "classification"
    st.session_state.last_qoi  = qoi
    st.session_state.last_task = task

    st.info(f"**Task:** {'Regression' if task=='regression' else 'Binary Classification'}  \n"
            f"**Models:** {'OLS · Neural Network · Random Forest' if task=='regression' else 'Logistic Regression · Neural Network · Random Forest'}")

    train_pct = st.slider("Training set size (%)", 50, 95, 80, 5)
    st.session_state.last_train_pct = train_pct
    st.markdown("</div>", unsafe_allow_html=True)

    # Prepare data
    df_enc = encode_for_model(df, type_map, qoi).dropna(subset=[qoi])
    le = None
    if task == "classification":
        le = LabelEncoder()
        df_enc[qoi] = le.fit_transform(df_enc[qoi].astype(str))

    feature_cols = [c for c in df_enc.columns if c != qoi]
    X = df_enc[feature_cols].copy()
    y = df_enc[qoi].copy()
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, train_size=train_pct/100, random_state=42)

    model_results = st.session_state.model_results

    def run_model(name, mdl):
        mdl.fit(X_tr, y_tr)
        if task == "regression":
            tr_p = mdl.predict(X_tr)
            te_p = mdl.predict(X_te)
            return {"name": name,
                    "train_mse":  float(mean_squared_error(y_tr, tr_p)),
                    "test_mse":   float(mean_squared_error(y_te, te_p)),
                    "train_rmse": float(np.sqrt(mean_squared_error(y_tr, tr_p))),
                    "test_rmse":  float(np.sqrt(mean_squared_error(y_te, te_p)))}
        else:
            tp, ep = mdl.predict(X_tr), mdl.predict(X_te)
            return {"name": name,
                    "train_acc": float(accuracy_score(y_tr, tp)),
                    "test_acc":  float(accuracy_score(y_te, ep)),
                    "train_f1":  float(f1_score(y_tr, tp, zero_division=0)),
                    "test_f1":   float(f1_score(y_te, ep, zero_division=0)),
                    "cm_train":  confusion_matrix(y_tr, tp).tolist(),
                    "cm_test":   confusion_matrix(y_te, ep).tolist(),
                    "le_classes": le.classes_.tolist()}

    # ── Model 1 ───────────────────────────────────────────────────────────────
    m1_label = "OLS Linear Regression" if task=="regression" else "Logistic Regression (GLM)"
    with st.expander(f"**Model 1 — {m1_label}**", expanded=True):
        m1_params = {}
        if task == "regression":
            st.markdown("*OLS fits analytically — no tunable hyperparameters.*")
        else:
            ca, cb = st.columns(2)
            m1_params["C"] = ca.number_input(
                "Regularisation C", 0.001, 100.0, 1.0, 0.1,
                help="Inverse regularisation strength. Smaller = stronger. Range: 0.001–100.", key="m1_C")
            m1_params["max_iter"] = cb.number_input(
                "Max iterations", 100, 10000, 1000, 100,
                help="Maximum solver iterations. Range: 100–10 000.", key="m1_iter")
        if st.button(f"Run {m1_label}", key="run_m1"):
            with st.spinner("Fitting..."):
                mdl = LinearRegression() if task=="regression" else \
                      LogisticRegression(C=m1_params["C"], max_iter=int(m1_params["max_iter"]), random_state=42)
                model_results["m1"] = run_model(m1_label, mdl)
                model_results["m1"]["params"] = {} if task=="regression" else {k: float(v) if isinstance(v, float) else v for k,v in m1_params.items()}
            st.success("Done!")
        if "m1" in model_results:
            _show_metrics(model_results["m1"], task)

    # ── Model 2 ───────────────────────────────────────────────────────────────
    with st.expander("**Model 2 — Neural Network (MLP)**", expanded=True):
        ca, cb, cc = st.columns(3)
        m2_layers = ca.text_input("Hidden layer sizes (comma-separated)", "100,50",
            help="E.g. '100,50' = two hidden layers. Each value: 1–1 000.", key="m2_layers")
        m2_alpha  = cb.number_input("L2 regularisation (alpha)", 1e-6, 10.0, 1e-4, format="%.6f",
            help="L2 penalty. Range: 0.000001–10.", key="m2_alpha")
        m2_iter   = cc.number_input("Max iterations", 50, 2000, 500, 50,
            help="Training epochs. Range: 50–2 000.", key="m2_iter")
        if st.button("Run Neural Network", key="run_m2"):
            try:
                layers = tuple(int(x.strip()) for x in m2_layers.split(",") if x.strip())
                assert all(1 <= n <= 1000 for n in layers), "Each layer size must be 1–1 000."
                with st.spinner("Fitting..."):
                    MLP = MLPRegressor if task=="regression" else MLPClassifier
                    mdl = MLP(hidden_layer_sizes=layers, alpha=m2_alpha, max_iter=int(m2_iter), random_state=42)
                    model_results["m2"] = run_model("Neural Network (MLP)", mdl)
                    model_results["m2"]["params"] = {
                        "hidden_layers": m2_layers,
                        "alpha":         round(float(m2_alpha), 6),
                        "max_iter":      int(m2_iter),
                    }
                st.success("Done!")
            except Exception as e:
                st.error(f"Invalid parameters: {e}")
        if "m2" in model_results:
            _show_metrics(model_results["m2"], task)

    # ── Model 3 ───────────────────────────────────────────────────────────────
    with st.expander("**Model 3 — Random Forest**", expanded=True):
        ca, cb, cc = st.columns(3)
        m3_n     = ca.number_input("Number of trees", 10, 1000, 100, 10,
            help="More trees = more stable, slower. Range: 10–1 000.", key="m3_n")
        m3_depth = cb.number_input("Max depth (0 = unlimited)", 0, 100, 0, 1,
            help="Max depth per tree. 0 = unlimited. Range: 0–100.", key="m3_depth")
        m3_split = cc.number_input("Min samples to split", 2, 100, 2, 1,
            help="Min samples to split an internal node. Range: 2–100.", key="m3_split")
        if st.button("Run Random Forest", key="run_m3"):
            with st.spinner("Fitting..."):
                depth = None if m3_depth == 0 else int(m3_depth)
                RF  = RandomForestRegressor if task=="regression" else RandomForestClassifier
                mdl = RF(n_estimators=int(m3_n), max_depth=depth,
                         min_samples_split=int(m3_split), random_state=42)
                model_results["m3"] = run_model("Random Forest", mdl)
                model_results["m3"]["params"] = {
                    "n_estimators":      int(m3_n),
                    "max_depth":         "unlimited" if m3_depth == 0 else int(m3_depth),
                    "min_samples_split": int(m3_split),
                }
            st.success("Done!")
        if "m3" in model_results:
            _show_metrics(model_results["m3"], task)

    # ── Summary ───────────────────────────────────────────────────────────────
    run_keys = [k for k in ["m1","m2","m3"] if k in model_results]
    if run_keys:
        st.markdown("---")
        st.markdown("### Combined Model Comparison")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        _show_summary(model_results, run_keys, task)
        st.markdown("</div>", unsafe_allow_html=True)

        if len(run_keys) == 3:
            st.session_state.step3_done = True
            st.markdown("---")
            if st.button("Proceed to Report (Step 4)"):
                st.session_state.step = 4
                st.rerun()

    if st.button("Back to Step 2"):
        st.session_state.step         = 2
        st.session_state.step2_done   = False
        st.session_state.model_results = {}
        st.rerun()


def _show_metrics(r, task):
    if task == "regression":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train MSE",  f"{r['train_mse']:.4f}")
        c2.metric("Test MSE",   f"{r['test_mse']:.4f}")
        c3.metric("Train RMSE", f"{r['train_rmse']:.4f}")
        c4.metric("Test RMSE",  f"{r['test_rmse']:.4f}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train Accuracy", f"{r['train_acc']:.4f}")
        c2.metric("Test Accuracy",  f"{r['test_acc']:.4f}")
        c3.metric("Train F1",       f"{r['train_f1']:.4f}")
        c4.metric("Test F1",        f"{r['test_f1']:.4f}")


def _show_summary(model_results, run_keys, task):
    if task == "regression":
        rows = [{"Model":      model_results[k]["name"],
                 "Train MSE":  round(model_results[k]["train_mse"],  6),
                 "Test MSE":   round(model_results[k]["test_mse"],   6),
                 "Train RMSE": round(model_results[k]["train_rmse"], 6),
                 "Test RMSE":  round(model_results[k]["test_rmse"],  6)} for k in run_keys]
        summary = pd.DataFrame(rows).set_index("Model")
        def hl_min(s):
            return ["background-color:#dcfce7;color:#15803d;font-weight:700"
                    if v == s.min() else "" for v in s]
        st.dataframe(summary.style.apply(hl_min, subset=["Test MSE","Test RMSE"]),
                     use_container_width=True)
        st.caption("Green = best (lowest) value.")
    else:
        rows = [{"Model":          model_results[k]["name"],
                 "Train Accuracy": round(model_results[k]["train_acc"], 4),
                 "Test Accuracy":  round(model_results[k]["test_acc"],  4),
                 "Train F1":       round(model_results[k]["train_f1"],  4),
                 "Test F1":        round(model_results[k]["test_f1"],   4)} for k in run_keys]
        summary = pd.DataFrame(rows).set_index("Model")
        def hl_max(s):
            return ["background-color:#dcfce7;color:#15803d;font-weight:700"
                    if v == s.max() else "" for v in s]
        st.dataframe(summary.style.apply(hl_max, subset=["Test Accuracy","Test F1"]),
                     use_container_width=True)
        st.caption("Green = best (highest) value.")
        st.markdown("**Confusion Matrices (Test Set)**")
        cm_cols = st.columns(len(run_keys))
        for i, k in enumerate(run_keys):
            r = model_results[k]
            cm  = np.array(r["cm_test"])
            cls = r["le_classes"]
            fig, ax = plt.subplots(figsize=(3.2, 2.8))
            fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
            ConfusionMatrixDisplay(cm, display_labels=cls).plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(r["name"], fontsize=9, fontweight="bold", color="#1a1a2e")
            ax.tick_params(labelsize=8); plt.tight_layout()
            cm_cols[i].pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT BUILDER  (calls Node.js generate_report.js)
# ══════════════════════════════════════════════════════════════════════════════

# Path to the JS script (same directory as this file)
_JS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_report.js")


def _auto_recommendation(model_results, task):
    """Generate a plain-English recommendation from results."""
    names  = {k: model_results[k]["name"] for k in ["m1","m2","m3"]}
    if task == "regression":
        scores = {k: model_results[k]["test_mse"] for k in ["m1","m2","m3"]}
        best_k = min(scores, key=scores.get)
        worst_k = max(scores, key=scores.get)
        best_name  = names[best_k]
        worst_name = names[worst_k]
        best_mse   = scores[best_k]
        worst_mse  = scores[worst_k]
        pct_better = 100 * (worst_mse - best_mse) / worst_mse if worst_mse > 0 else 0
        return (
            f"{best_name} achieved the lowest Test MSE ({best_mse:.6f}), outperforming "
            f"{worst_name} ({worst_mse:.6f}) by {pct_better:.1f}% on the held-out test set. "
            f"On this basis, {best_name} is recommended as the primary model for predicting {model_results['m1']['name']}. "
            f"If interpretability is a priority, OLS Linear Regression provides transparent, "
            f"auditable coefficients at the cost of some predictive accuracy. "
            f"Random Forest results can be further improved by tuning n_estimators and max_depth."
        )
    else:
        scores = {k: model_results[k]["test_acc"] for k in ["m1","m2","m3"]}
        best_k = max(scores, key=scores.get)
        worst_k = min(scores, key=scores.get)
        best_name = names[best_k]
        worst_name = names[worst_k]
        best_acc  = scores[best_k]
        worst_acc = scores[worst_k]
        return (
            f"{best_name} achieved the highest Test Accuracy ({best_acc:.4f}), "
            f"outperforming {worst_name} ({worst_acc:.4f}) on the held-out test set. "
            f"{best_name} is recommended as the primary classifier. "
            f"If regulatory or audit requirements demand an explainable model, "
            f"Logistic Regression offers interpretable odds-ratios per predictor. "
            f"F1 Score should be used as the primary metric if class imbalance is present."
        )


def build_report_docx(model_results, task, qoi, train_pct, file_name):
    """Serialize results to JSON → call Node.js → return docx bytes."""
    payload = {
        "qoi":          qoi,
        "task":         task,
        "train_pct":    train_pct,
        "file_name":    file_name,
        "generated_at": datetime.datetime.now().strftime("%d %B %Y, %H:%M"),
        "recommendation": _auto_recommendation(model_results, task),
        "models": {
            k: {
                "name":       model_results[k]["name"],
                "params":     model_results[k].get("params", {}),
                "train_mse":  model_results[k].get("train_mse"),
                "test_mse":   model_results[k].get("test_mse"),
                "train_rmse": model_results[k].get("train_rmse"),
                "test_rmse":  model_results[k].get("test_rmse"),
                "train_acc":  model_results[k].get("train_acc"),
                "test_acc":   model_results[k].get("test_acc"),
                "train_f1":   model_results[k].get("train_f1"),
                "test_f1":    model_results[k].get("test_f1"),
            }
            for k in ["m1","m2","m3"] if k in model_results
        },
    }

    # Write JSON to temp file, call Node, capture stdout bytes
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        json.dump(payload, tf)
        tf_path = tf.name

    try:
        result = subprocess.run(
            ["node", _JS_SCRIPT, tf_path],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode())
        return BytesIO(result.stdout)
    finally:
        os.unlink(tf_path)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — REPORT
# ══════════════════════════════════════════════════════════════════════════════
def step4():
    st.markdown("## Step 4 — Model Report")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    model_results = st.session_state.model_results
    task          = st.session_state.last_task
    qoi           = st.session_state.last_qoi
    train_pct     = st.session_state.last_train_pct
    file_name     = st.session_state.file_name
    run_keys      = [k for k in ["m1","m2","m3"] if k in model_results]

    if len(run_keys) < 3:
        st.warning("Please run all three models in Step 3 before generating the report.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── Preview ───────────────────────────────────────────────────────────────
    st.markdown("### Report Contents")
    st.markdown(
        f"**QOI:** `{qoi}`  &nbsp;|&nbsp;  "
        f"**Task:** {'Regression' if task=='regression' else 'Binary Classification'}  &nbsp;|&nbsp;  "
        f"**Split:** {train_pct} / {100-train_pct}"
    )

    st.markdown("""
The report will contain:
- **Problem setup** — QOI, task type, train/test split, encoding approach
- **Model descriptions** — plain-English explanation of each method, key assumptions, parameters used
- **Performance table** — MSE/RMSE (regression) or Accuracy/F1 (classification) for all three models, with the best result highlighted
- **Recommendation** — auto-generated narrative identifying the best-performing model and guidance on when to prefer alternatives
- **Methodology notes** — reproducibility, encoding, and interpretation guidance
""")

    st.markdown("---")
    st.markdown("**Performance summary:**")
    _show_summary(model_results, run_keys, task)

    st.markdown("---")
    st.markdown("### Auto-generated Recommendation")
    rec = _auto_recommendation(model_results, task)
    st.info(rec)

    st.markdown("---")

    # ── Generate & download ───────────────────────────────────────────────────
    if st.button("Generate & Download Report (.docx)"):
        with st.spinner("Building Word document..."):
            try:
                docx_buf = build_report_docx(model_results, task, qoi, train_pct, file_name)
                ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                fname = f"ML_Report_{qoi}_{ts}.docx"
                st.download_button(
                    label=f"Download {fname}",
                    data=docx_buf,
                    file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
                st.success(f"Ready — click above to download **{fname}**.")
            except Exception as e:
                st.error(f"Report generation failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("Back to Step 3"):
        st.session_state.step = 3
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    sidebar_steps()
    step = st.session_state.step
    if step == 1:
        step1()
    elif step == 2:
        step2() if st.session_state.step1_done  else st.warning("Complete Step 1 first.")
    elif step == 3:
        step3() if st.session_state.step2_done  else st.warning("Complete Step 2 first.")
    elif step == 4:
        step4() if st.session_state.step3_done  else st.warning("Complete Step 3 first.")

if __name__ == "__main__":
    main()
