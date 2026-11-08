import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SAMPLE_CSV = os.path.join(BASE_DIR, "test_data.csv")

TARGET = "price_range"
CLASSES = [0, 1, 2, 3]
TIER_NAMES = {0: "Low", 1: "Medium", 2: "High", 3: "Very high"}
TIER_COLORS = {0: "#2a9d8f", 1: "#e9c46a", 2: "#f4a261", 3: "#d1495b"}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_NOTES = {
    "Accuracy": "share of phones put in the right bracket",
    "AUC": "one-vs-rest, weighted by class size",
    "Precision": "weighted over the 4 brackets",
    "Recall": "weighted over the 4 brackets",
    "F1": "weighted over the 4 brackets",
    "MCC": "1 = perfect, 0 = no better than chance",
}

st.set_page_config(page_title="PriceBand | mobile price classifier", page_icon="📱", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Space Grotesk', sans-serif; }
    .block-container { padding-top: 2rem; max-width: 1200px; }

    .hero {
        background: #1d2433; color: #f6f3ee; border-radius: 4px;
        padding: 1.6rem 2rem 1.4rem 2rem; margin-bottom: 1.2rem;
        border-left: 10px solid #d1495b; position: relative; overflow: hidden;
    }
    .hero h1 { font-size: 2.3rem; margin: 0; letter-spacing: -1px; color: #f6f3ee; }
    .hero p { margin: .35rem 0 .9rem 0; color: #c9cfdb; font-size: 1.02rem; }
    .hero .stripe { position: absolute; right: 0; top: 0; bottom: 0; width: 120px; display: flex; }
    .hero .stripe div { flex: 1; }
    .chip {
        display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: .78rem;
        border: 1px solid #4a5468; border-radius: 2px; padding: 2px 8px; margin-right: 6px; color: #e6e9ef;
    }

    .section-label {
        font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: 2px;
        font-size: .75rem; color: #7a6f60; margin: 1.2rem 0 .4rem 0;
    }

    .tile {
        background: #ffffff; border: 1px solid #ddd5c7; border-radius: 4px;
        padding: .9rem 1rem .8rem 1rem; height: 100%;
    }
    .tile .name { font-family: 'IBM Plex Mono', monospace; font-size: .78rem; color: #7a6f60; text-transform: uppercase; letter-spacing: 1px; }
    .tile .value { font-size: 2rem; font-weight: 700; color: #1d2433; line-height: 1.2; }
    .tile .note { font-size: .74rem; color: #8d8577; min-height: 2.1em; }
    .tile .bar { height: 6px; background: #ebe5da; border-radius: 3px; margin-top: .5rem; }
    .tile .bar span { display: block; height: 6px; border-radius: 3px; background: #d1495b; }

    .tier-row { display: flex; gap: 8px; margin-bottom: .4rem; }
    .tier {
        flex: 1; background: #fff; border: 1px solid #ddd5c7; border-radius: 4px; padding: .55rem .8rem;
        border-top: 5px solid var(--c);
    }
    .tier b { font-size: 1.3rem; }
    .tier small { display: block; color: #7a6f60; font-family: 'IBM Plex Mono', monospace; font-size: .72rem; }

    .empty {
        border: 2px dashed #c9bfae; border-radius: 6px; padding: 2rem; background: #fbf9f5; color: #4a5468;
    }
    .empty code { background: #ebe5da; }

    section[data-testid="stSidebar"] { border-right: 1px solid #ddd5c7; }
    .step { font-family: 'IBM Plex Mono', monospace; font-size: .75rem; color: #d1495b; letter-spacing: 1px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    return {name: joblib.load(os.path.join(MODEL_DIR, fn)) for name, fn in MODEL_FILES.items()}


def score(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)
    try:
        auc = roc_auc_score(y, prob, multi_class="ovr", average="weighted", labels=CLASSES)
    except ValueError:
        # happens when the uploaded file doesn't contain all 4 brackets
        auc = np.nan
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pred


def metric_tile(name, value):
    shown = "n/a" if pd.isna(value) else f"{value:.4f}"
    # MCC can go below 0, clip the bar so it still draws
    width = 0 if pd.isna(value) else max(0.0, min(1.0, value)) * 100
    return f"""
    <div class="tile">
        <div class="name">{name}</div>
        <div class="value">{shown}</div>
        <div class="note">{METRIC_NOTES[name]}</div>
        <div class="bar"><span style="width:{width:.1f}%"></span></div>
    </div>"""


def confusion_plot(y, pred):
    cm = confusion_matrix(y, pred, labels=CLASSES)
    cmap = LinearSegmentedColormap.from_list("paper", ["#fbf9f5", "#f4a261", "#d1495b"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm, cmap=cmap)
    names = [TIER_NAMES[c] for c in CLASSES]
    ax.set_xticks(range(4), names)
    ax.set_yticks(range(4), names)
    ax.set_xlabel("predicted bracket")
    ax.set_ylabel("actual bracket")
    for i in range(4):
        for j in range(4):
            colour = "white" if cm[i, j] > cm.max() * 0.6 else "#1d2433"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=colour, fontsize=13, fontweight="bold")
    for side in ax.spines.values():
        side.set_visible(False)
    fig.patch.set_facecolor("#f6f3ee")
    fig.tight_layout()
    return fig, cm


# ---------------------------------------------------------------- header
stripe = "".join(f'<div style="background:{TIER_COLORS[c]}"></div>' for c in CLASSES)
st.markdown(
    f"""
    <div class="hero">
        <div class="stripe">{stripe}</div>
        <h1>PriceBand</h1>
        <p>Upload labelled phone specs and see how five pre-trained classifiers sort them into price brackets.</p>
        <span class="chip">Kaggle: Mobile Price Classification</span>
        <span class="chip">20 features</span>
        <span class="chip">4 brackets</span>
        <span class="chip">5 saved models · no retraining</span>
    </div>
    """,
    unsafe_allow_html=True,
)

models = load_models()
feature_cols = list(next(iter(models.values())).feature_names_in_)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="step">STEP 1</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv",
                                help=f"Needs the 20 feature columns and the `{TARGET}` column.")
    use_sample = st.toggle("Use the bundled test_data.csv instead", value=False)
    with open(SAMPLE_CSV, "rb") as f:
        st.download_button("Download test_data.csv", f, file_name="test_data.csv", mime="text/csv",
                           width="stretch")

    st.markdown('<div class="step">STEP 2</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))

    st.divider()
    st.caption("Models were trained and tuned offline in `model_training.ipynb` and are only loaded here. "
               "Nothing is refit on the uploaded file.")

# ---------------------------------------------------------------- data
if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    data = pd.read_csv(SAMPLE_CSV)
    source = "test_data.csv (bundled)"
else:
    st.markdown(
        f"""
        <div class="empty">
            <h3 style="margin-top:0">No data yet</h3>
            Upload a CSV in the sidebar, or switch on <b>Use the bundled test_data.csv</b>.<br><br>
            The file needs these columns (extra columns are ignored):<br>
            <code>{'</code> <code>'.join(feature_cols)}</code> and the target <code>{TARGET}</code>
            (0 = low, 1 = medium, 2 = high, 3 = very high).
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

missing = [c for c in feature_cols + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The file is missing {len(missing)} required column(s): {', '.join(missing)}")
    st.stop()

data = data.dropna(subset=feature_cols + [TARGET])
bad_labels = sorted(set(data[TARGET].unique()) - set(CLASSES))
if bad_labels:
    st.error(f"`{TARGET}` should only contain 0, 1, 2 or 3. Found: {bad_labels}")
    st.stop()
if data.empty:
    st.error("No usable rows left after dropping rows with missing values.")
    st.stop()

X = data[feature_cols]
y = data[TARGET].astype(int)

st.markdown(f'<div class="section-label">Data · {source} · {len(data)} phones</div>', unsafe_allow_html=True)
counts = y.value_counts()
tiers = "".join(
    f'<div class="tier" style="--c:{TIER_COLORS[c]}"><b>{counts.get(c, 0)}</b>'
    f'<small>{c} · {TIER_NAMES[c].upper()}</small></div>'
    for c in CLASSES
)
st.markdown(f'<div class="tier-row">{tiers}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- results
metrics, pred = score(models[model_name], X, y)
if pd.isna(metrics["AUC"]):
    st.warning("AUC can't be computed because the file doesn't contain all four price brackets.")

tab_score, tab_cm, tab_report, tab_all, tab_pred = st.tabs(
    ["Scorecard", "Confusion matrix", "Classification report", "All models", "Predictions"]
)

with tab_score:
    st.markdown(f'<div class="section-label">{model_name}</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for col, (name, value) in zip(cols, metrics.items()):
        col.markdown(metric_tile(name, value), unsafe_allow_html=True)
    wrong = int((pred != y.values).sum())
    st.markdown(" ")
    st.write(f"**{len(y) - wrong}** of **{len(y)}** phones placed in the right bracket, **{wrong}** missed.")

with tab_cm:
    left, right = st.columns([3, 2])
    fig, cm = confusion_plot(y, pred)
    left.pyplot(fig)
    plt.close(fig)
    off_by_one = sum(cm[i, j] for i in range(4) for j in range(4) if abs(i - j) == 1)
    far = sum(cm[i, j] for i in range(4) for j in range(4) if abs(i - j) > 1)
    right.markdown('<div class="section-label">Reading it</div>', unsafe_allow_html=True)
    right.write("Rows are the real bracket, columns are what the model said. The diagonal is correct.")
    right.write(f"- misses into the **next** bracket: **{off_by_one}**\n- misses by **two or more** brackets: **{far}**")

with tab_report:
    report = classification_report(y, pred, labels=CLASSES, target_names=[TIER_NAMES[c] for c in CLASSES],
                                   output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    report_df["support"] = report_df["support"].astype(int)
    st.dataframe(report_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}"}),
                 width="stretch")

with tab_all:
    rows = []
    for name, m in models.items():
        res, _ = score(m, X, y)
        rows.append({"Model": name, **res})
    table = pd.DataFrame(rows).set_index("Model")
    st.markdown('<div class="section-label">Same file, all five models · best per column highlighted</div>',
                unsafe_allow_html=True)
    st.dataframe(
        table.style.format("{:.4f}", na_rep="n/a")
             .highlight_max(axis=0, props="background-color:#f4d6d1; font-weight:700;"),
        width="stretch",
    )

    fig, ax = plt.subplots(figsize=(9, 3.2))
    order = table["Accuracy"].sort_values()
    colours = ["#d1495b" if n == model_name else "#b7ad9c" for n in order.index]
    ax.barh(order.index, order.values, color=colours)
    for i, v in enumerate(order.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("accuracy")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.patch.set_facecolor("#f6f3ee")
    ax.set_facecolor("#f6f3ee")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The model picked in the sidebar is shown in red.")

with tab_pred:
    out = data.copy()
    out["predicted"] = pred
    out["correct"] = out["predicted"] == out[TARGET]
    only_wrong = st.checkbox("Show only the misses")
    view = out[~out["correct"]] if only_wrong else out
    st.dataframe(view[[TARGET, "predicted", "correct"] + feature_cols], width="stretch", height=380)
    st.download_button("Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv", mime="text/csv")
