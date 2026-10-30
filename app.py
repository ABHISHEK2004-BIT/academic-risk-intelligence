from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Circle
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


DATA_PATH = Path(__file__).parent / "student_academic_risk_1000.csv"
TARGET_COLUMN = "Academic_Risk"
RISK_LABEL = "At Risk"
CRITERIA = ("entropy", "gini")

COLORS = {
    "background": "#0b1220",
    "surface": "#111c2c",
    "surface_alt": "#172538",
    "border": "#25364a",
    "text": "#e8eef5",
    "muted": "#9aabbd",
    "blue": "#67a9ff",
    "teal": "#54d2c2",
    "lime": "#c5e879",
}


st.set_page_config(
    page_title="Academic Risk | Analytics",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --app-bg: #0b1220;
            --surface: #111c2c;
            --surface-alt: #172538;
            --border: #25364a;
            --text: #e8eef5;
            --muted: #9aabbd;
            --accent: #54d2c2;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 12% 0%, #14253a 0, transparent 34%),
                var(--app-bg);
            color: var(--text);
        }
        [data-testid="stHeader"] { background: rgba(11, 18, 32, .88); }
        [data-testid="stSidebar"] {
            background: #0e1928;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
        .block-container {
            max-width: 1500px;
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 { color: var(--text) !important; letter-spacing: -.025em; }
        p, label, [data-testid="stMarkdownContainer"] { color: #d3deea; }
        [data-testid="stCaptionContainer"] { color: var(--muted); }
        [data-testid="stMetric"] {
            height: 100%;
            padding: 1rem 1.1rem;
            background: linear-gradient(145deg, #162438, #111c2c);
            border: 1px solid var(--border);
            border-radius: 12px;
        }
        [data-testid="stMetricLabel"] { color: var(--muted) !important; }
        [data-testid="stMetricValue"] { color: var(--text) !important; }
        [data-testid="stMetricDelta"] { font-size: .82rem; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(17, 28, 44, .78);
            border-color: var(--border);
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, .12);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 10px;
        }
        [data-baseweb="tab-list"] { gap: .45rem; }
        button[data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
        .stButton > button, .stFormSubmitButton > button {
            min-height: 2.7rem;
            color: #071821;
            background: linear-gradient(100deg, #54d2c2, #67a9ff);
            border: 0;
            border-radius: 9px;
            font-weight: 700;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            color: #071821;
            border: 0;
            filter: brightness(1.08);
        }
        hr { border-color: var(--border); }
        .hero {
            padding: 1.4rem 1.6rem;
            margin: 0 0 1.3rem;
            background: linear-gradient(115deg, #162a40 0%, #122132 60%, #102b32 100%);
            border: 1px solid #2b4358;
            border-radius: 16px;
        }
        .hero-eyebrow, .section-eyebrow {
            color: #76d9cb;
            font-size: .72rem;
            font-weight: 750;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .hero-title {
            margin: .25rem 0 .35rem;
            color: #f2f6fb;
            font-size: clamp(1.5rem, 3vw, 2.15rem);
            font-weight: 760;
            letter-spacing: -.04em;
        }
        .hero-copy {
            max-width: 760px;
            margin: 0;
            color: #b6c4d2;
            font-size: .96rem;
            line-height: 1.55;
        }
        .section-heading {
            margin: 0 0 .25rem;
            color: #eff5fb;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: -.015em;
        }
        .section-copy {
            margin: 0 0 .8rem;
            color: var(--muted);
            font-size: .83rem;
            line-height: 1.45;
        }
        .sidebar-brand {
            color: #eff5fb;
            font-size: 1.05rem;
            font-weight: 750;
            letter-spacing: -.02em;
        }
        .sidebar-mark {
            display: inline-block;
            margin-right: .4rem;
            color: #54d2c2;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()

    data = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in data.columns:
        st.error(f"Dataset is missing the required target column: {TARGET_COLUMN}")
        st.stop()
    return data


@st.cache_resource
def train_models(data: pd.DataFrame):
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    models = {}
    for criterion in CRITERIA:
        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        criterion=criterion,
                        max_depth=5,
                        min_samples_leaf=10,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        models[criterion] = model
    return X_test, y_test, models


def section_heading(title: str, description: str | None = None) -> None:
    st.markdown(
        f'<div class="section-heading">{title}</div>'
        + (
            f'<div class="section-copy">{description}</div>'
            if description
            else ""
        ),
        unsafe_allow_html=True,
    )


def page_heading(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">{eyebrow}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-copy">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig: plt.Figure) -> plt.Figure:
    fig.patch.set_facecolor(COLORS["surface"])
    for ax in fig.axes:
        ax.set_facecolor(COLORS["surface"])
        ax.tick_params(colors="#c4d0dc", labelsize=8)
        ax.xaxis.label.set_color("#c4d0dc")
        ax.yaxis.label.set_color("#c4d0dc")
        ax.title.set_color(COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])
        if ax.axison:
            ax.grid(color=COLORS["border"], alpha=0.55, linewidth=0.6)
            ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def show_figure(fig: plt.Figure) -> None:
    st.pyplot(style_figure(fig), use_container_width=True)
    plt.close(fig)


def draw_network() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    layers = [
        (1.0, [1.0, 2.0, 3.0, 4.0], COLORS["blue"], "INPUT"),
        (3.5, [0.7, 1.7, 2.7, 3.7, 4.7], COLORS["teal"], "FEATURES"),
        (6.2, [1.0, 2.0, 3.0, 4.0], COLORS["teal"], "TREE NODES"),
        (8.8, [1.5, 3.0, 4.5], COLORS["lime"], "OUTPUT"),
    ]
    for index in range(len(layers) - 1):
        x1, ys1, _, _ = layers[index]
        x2, ys2, _, _ = layers[index + 1]
        for y1 in ys1:
            for y2 in ys2:
                ax.plot(
                    [x1 + 0.15, x2 - 0.15],
                    [y1, y2],
                    color="#52748b",
                    alpha=0.28,
                    linewidth=0.7,
                    zorder=1,
                )
    for x, positions, color, label in layers:
        for y in positions:
            ax.add_patch(
                Circle(
                    (x, y),
                    0.14,
                    facecolor=color,
                    edgecolor="#e2edf5",
                    linewidth=0.8,
                    zorder=3,
                )
            )
        ax.text(
            x,
            5.55,
            label,
            ha="center",
            color="#d8e5ee",
            fontsize=8,
            fontweight="bold",
        )
    ax.text(
        9.35,
        3.0,
        "Risk\nclass",
        color="#d8e5ee",
        ha="left",
        va="center",
        fontsize=9,
    )
    ax.text(
        5,
        0.12,
        "Illustrative workflow — the classifier is a decision tree, not a neural network.",
        color=COLORS["muted"],
        ha="center",
        fontsize=8,
    )
    return fig


def render_overview(
    data: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict,
    tree: DecisionTreeClassifier,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    risk_index: int,
    criterion: str,
) -> None:
    left, right = st.columns([1.55, 1])
    with left:
        with st.container(border=True):
            section_heading(
                "How the model works",
                "Student attributes are encoded, evaluated through decision rules, and mapped to a risk class.",
            )
            show_figure(draw_network())
    with right:
        with st.container(border=True):
            section_heading(
                "Held-out performance",
                "Precision and recall on the stratified test split.",
            )
            report = pd.DataFrame(
                classification_report(
                    y_test, y_pred, output_dict=True, zero_division=0
                )
            ).T
            class_rows = report.loc[
                [label for label in tree.classes_ if label in report.index],
                ["precision", "recall"],
            ]
            fig, ax = plt.subplots(figsize=(5, 3.8))
            positions = np.arange(len(class_rows))
            width = 0.32
            ax.bar(
                positions - width / 2,
                class_rows["precision"] * 100,
                width,
                label="Precision",
                color=COLORS["blue"],
            )
            ax.bar(
                positions + width / 2,
                class_rows["recall"] * 100,
                width,
                label="Recall",
                color=COLORS["teal"],
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(class_rows.index)
            ax.set_ylim(0, 105)
            ax.set_ylabel("Score (%)")
            ax.legend(frameon=False)
            show_figure(fig)

    lower_left, lower_middle, lower_right = st.columns([1.35, 1, 1])
    with lower_left:
        with st.container(border=True):
            section_heading("Cross-validation", "Five-fold accuracy across the full dataset.")
            features = data.drop(columns=[TARGET_COLUMN])
            target = data[TARGET_COLUMN]
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_validate(
                models[criterion], features, target, cv=folds, scoring=["accuracy"]
            )["test_accuracy"]
            fig, ax = plt.subplots(figsize=(5.3, 2.5))
            ax.plot(
                range(1, 6),
                scores * 100,
                marker="o",
                linewidth=2.5,
                color=COLORS["teal"],
            )
            ax.axhline(
                scores.mean() * 100,
                color=COLORS["lime"],
                linestyle="--",
                linewidth=1,
                label=f"Mean {scores.mean():.1%}",
            )
            ax.set_xticks(range(1, 6))
            ax.set_xlabel("Fold")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 105)
            ax.legend(frameon=False)
            show_figure(fig)
            st.caption(f"Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    with lower_middle:
        with st.container(border=True):
            section_heading(
                "Prediction mix", "Predicted classes in the held-out test set."
            )
            counts = pd.Series(y_pred).value_counts()
            fig, ax = plt.subplots(figsize=(3.4, 2.5))
            ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.0f%%",
                colors=[COLORS["blue"], COLORS["teal"]],
                textprops={"color": COLORS["text"], "fontsize": 8},
                wedgeprops={"width": 0.42, "edgecolor": COLORS["surface"]},
            )
            show_figure(fig)
    with lower_right:
        with st.container(border=True):
            section_heading("Data quality", "Basic checks on the supplied CSV.")
            st.metric("Missing values", f"{int(data.isna().sum().sum()):,}")
            st.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")
            st.metric("Input features", f"{data.shape[1] - 1}")

    with st.container(border=True):
        section_heading(
            "Test-set examples",
            "Sample predictions for held-out records. No live student records are connected.",
        )
        preview = X_test.head(8).copy()
        preview["Actual risk"] = y_test.head(8).values
        preview["Predicted risk"] = y_pred[:8]
        preview["At-risk probability"] = [
            f"{value:.1%}" for value in probabilities[:8, risk_index]
        ]
        st.dataframe(
            preview.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )


def render_exploration(data: pd.DataFrame) -> None:
    page_heading(
        "Dataset",
        "Explore the data",
        "Review the synthetic teaching dataset, inspect its quality, and explore feature distributions.",
    )
    columns = st.columns(4)
    columns[0].metric("Records", f"{len(data):,}")
    columns[1].metric("Columns", f"{len(data.columns)}")
    columns[2].metric("Missing values", f"{int(data.isna().sum().sum()):,}")
    columns[3].metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")

    with st.container(border=True):
        section_heading("Dataset preview", "First 20 records from the supplied CSV.")
        st.dataframe(data.head(20), use_container_width=True, hide_index=True)

    first, second = st.columns(2)
    with first:
        with st.container(border=True):
            section_heading("Risk class balance")
            counts = data[TARGET_COLUMN].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(counts.index, counts.values, color=[COLORS["blue"], COLORS["teal"]])
            ax.set_ylabel("Records")
            show_figure(fig)
    with second:
        with st.container(border=True):
            section_heading("Records by program")
            counts = data["Program"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(counts.index, counts.values, color=COLORS["teal"])
            ax.set_xlabel("Records")
            show_figure(fig)

    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    with st.container(border=True):
        section_heading("Feature distribution", "Select a numeric field to inspect its spread.")
        selected = st.selectbox("Numeric feature", numeric_columns)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(
            data[selected],
            bins=25,
            color=COLORS["blue"],
            edgecolor="#dce8f2",
            alpha=0.9,
        )
        ax.set_xlabel(selected)
        ax.set_ylabel("Frequency")
        show_figure(fig)

    with st.container(border=True):
        section_heading(
            "Numeric feature correlations",
            "Pairwise correlation among numeric fields; the categorical target is excluded.",
        )
        correlation = data[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(9, 5))
        image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(correlation.columns)))
        ax.set_xticklabels(correlation.columns, rotation=65, ha="right", fontsize=8)
        ax.set_yticks(range(len(correlation.index)))
        ax.set_yticklabels(correlation.index, fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.03)
        show_figure(fig)


def render_performance(
    y_test: pd.Series,
    y_pred: np.ndarray,
    tree: DecisionTreeClassifier,
    models: dict,
    X_test: pd.DataFrame,
    auc: float,
) -> None:
    page_heading(
        "Model evaluation",
        "Measure model performance",
        "Inspect held-out classification results and compare entropy and Gini decision trees under the same split.",
    )
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            section_heading("Confusion matrix", "Predicted versus observed classes.")
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_pred,
                labels=tree.classes_,
                ax=ax,
                cmap="Blues",
                colorbar=False,
            )
            show_figure(fig)
    with right:
        with st.container(border=True):
            section_heading("Classification report", "Per-class precision, recall, and F1.")
            report = pd.DataFrame(
                classification_report(
                    y_test, y_pred, output_dict=True, zero_division=0
                )
            ).T.round(3)
            st.dataframe(report, use_container_width=True)

    with st.container(border=True):
        section_heading("Criterion comparison", "Both models use the same test split.")
        rows = []
        for name, candidate in models.items():
            predictions = candidate.predict(X_test)
            rows.append(
                {
                    "Criterion": name.title(),
                    "Accuracy": accuracy_score(y_test, predictions),
                    "Macro precision": precision_score(
                        y_test, predictions, average="macro", zero_division=0
                    ),
                    "Macro recall": recall_score(
                        y_test, predictions, average="macro", zero_division=0
                    ),
                    "Macro F1": f1_score(
                        y_test, predictions, average="macro", zero_division=0
                    ),
                }
            )
        comparison, auc_column = st.columns([2, 1])
        comparison.dataframe(
            pd.DataFrame(rows).set_index("Criterion").round(3),
            use_container_width=True,
        )
        auc_column.metric(
            "ROC-AUC · At Risk",
            f"{auc:.3f}" if np.isfinite(auc) else "Not available",
        )
        st.caption(
            "Configuration: stratified 80/20 split, max_depth=5, "
            "min_samples_leaf=10, random_state=42."
        )


def render_explainability(
    tree: DecisionTreeClassifier, feature_names: np.ndarray
) -> None:
    page_heading(
        "Model interpretation",
        "Understand the decision tree",
        "Review which encoded features influence the fitted model and inspect its learned split rules.",
    )
    importances = pd.Series(
        tree.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    first, second = st.columns([1, 1.4])
    with first:
        with st.container(border=True):
            section_heading(
                "Most influential features",
                "Feature importance is descriptive, not causal.",
            )
            st.bar_chart(importances.head(12), color=COLORS["teal"])
    with second:
        with st.container(border=True):
            section_heading("Tree visualization", "Displayed to a maximum depth of four.")
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_tree(
                tree,
                feature_names=feature_names,
                class_names=tree.classes_,
                filled=True,
                rounded=True,
                max_depth=4,
                fontsize=7,
                ax=ax,
            )
            show_figure(fig)
    with st.container(border=True):
        section_heading("Extracted decision rules")
        st.code(
            export_text(tree, feature_names=list(feature_names), max_depth=5),
            language="text",
        )
    st.caption(
        "Feature importance describes this fitted synthetic-data model; it does not establish causation."
    )


def render_prediction(
    data: pd.DataFrame,
    model: Pipeline,
    tree: DecisionTreeClassifier,
    feature_names: np.ndarray,
    risk_index: int,
) -> None:
    page_heading(
        "Interactive prediction",
        "Try a hypothetical profile",
        "Explore how the teaching model responds to an example profile. Do not enter identifiable personal information.",
    )
    with st.container(border=True):
        section_heading("Profile inputs", "All values are illustrative.")
        with st.form("prediction_form"):
            first, second, third = st.columns(3)
            with first:
                age = st.number_input("Age", 18, 25, 20)
                attendance = st.slider("Attendance (%)", 35.0, 100.0, 70.0)
                study_hours = st.slider("Weekly study hours", 0.0, 12.0, 4.0)
            with second:
                previous_score = st.slider("Previous exam score", 15.0, 100.0, 60.0)
                assignments = st.slider("Assignments completed (out of 10)", 0, 10, 6)
                sleep_hours = st.slider("Average sleep hours", 3.0, 10.0, 7.0)
            with third:
                internet = st.selectbox(
                    "Internet access", ["Good", "Average", "Poor"]
                )
                participation = st.selectbox(
                    "Class participation", ["High", "Medium", "Low"]
                )
                program = st.selectbox("Program", sorted(data["Program"].unique()))
            submitted = st.form_submit_button(
                "Generate prediction", type="primary", use_container_width=True
            )

    if not submitted:
        return

    sample = pd.DataFrame(
        [
            {
                "Age": age,
                "Attendance_Percent": attendance,
                "Weekly_Study_Hours": study_hours,
                "Previous_Exam_Score": previous_score,
                "Assignments_Completed_Out_of_10": assignments,
                "Average_Sleep_Hours": sleep_hours,
                "Internet_Access": internet,
                "Class_Participation": participation,
                "Program": program,
            }
        ]
    )
    started = perf_counter()
    label = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    elapsed_ms = (perf_counter() - started) * 1000

    st.markdown("#### Prediction result")
    metrics = st.columns(3)
    metrics[0].metric("Predicted class", label)
    metrics[1].metric("At-risk probability", f"{probabilities[risk_index]:.1%}")
    metrics[2].metric("Inference time", f"{elapsed_ms:.2f} ms")

    left, right = st.columns([1, 1.2])
    with left:
        with st.container(border=True):
            section_heading("Class probabilities")
            st.bar_chart(
                pd.Series(probabilities, index=tree.classes_),
                color=COLORS["blue"],
            )
    with right:
        with st.container(border=True):
            section_heading("Decision path", "The split taken for this profile at each node.")
            transformed = model.named_steps["preprocessor"].transform(sample)
            path = tree.decision_path(transformed)
            node_ids = path.indices[path.indptr[0] : path.indptr[1]]
            children_left = tree.tree_.children_left
            for index, node in enumerate(node_ids[:-1]):
                feature_id = tree.tree_.feature[node]
                threshold = tree.tree_.threshold[node]
                branch = "≤" if node_ids[index + 1] == children_left[node] else ">"
                feature_name = (
                    feature_names[feature_id] if feature_id >= 0 else "leaf"
                )
                st.write(f"Node {node}: `{feature_name}` {branch} `{threshold:.3f}`")
    st.caption(
        "This is a model output from synthetic teaching data, not a validated academic-risk assessment."
    )


def main() -> None:
    apply_theme()
    data = load_data()
    X_test, y_test, models = train_models(data)

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><span class="sidebar-mark">◆</span>ACADEMIC RISK</div>',
            unsafe_allow_html=True,
        )
        st.caption("Explainable analytics · Teaching project")
        st.divider()
        view = st.radio(
            "Workspace",
            [
                "Executive Overview",
                "Data Exploration",
                "Model Performance",
                "Explainability",
                "Live Prediction",
            ],
        )
        st.divider()
        criterion = st.selectbox(
            "Decision-tree criterion",
            CRITERIA,
            format_func=str.title,
            key="criterion",
        )
        st.caption("Stratified 80/20 train-test split · random_state=42")
        st.warning(
            "Synthetic teaching data only. Do not use this model to make decisions about real students."
        )

    model = models[criterion]
    tree = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    risk_index = list(tree.classes_).index(RISK_LABEL)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label=RISK_LABEL, zero_division=0
    )
    recall = recall_score(y_test, predictions, pos_label=RISK_LABEL, zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label=RISK_LABEL, zero_division=0)
    try:
        auc = roc_auc_score(
            (y_test == RISK_LABEL).astype(int), probabilities[:, risk_index]
        )
    except ValueError:
        auc = float("nan")

    if view == "Executive Overview":
        page_heading(
            "IALM · ENGINEERING LAB",
            "Academic risk analytics",
            "An explainable decision-tree dashboard for exploring a synthetic academic-risk dataset and its model behavior.",
        )
    metrics = st.columns(5)
    metrics[0].metric("Dataset records", f"{len(data):,}")
    metrics[1].metric("Test accuracy", f"{accuracy:.1%}", criterion.title())
    metrics[2].metric("Precision · At Risk", f"{precision:.1%}")
    metrics[3].metric("Recall · At Risk", f"{recall:.1%}")
    metrics[4].metric("F1 · At Risk", f"{f1:.1%}")
    st.write("")

    if view == "Executive Overview":
        render_overview(
            data,
            X_test,
            y_test,
            models,
            tree,
            predictions,
            probabilities,
            risk_index,
            criterion,
        )
    elif view == "Data Exploration":
        render_exploration(data)
    elif view == "Model Performance":
        render_performance(y_test, predictions, tree, models, X_test, auc)
    elif view == "Explainability":
        render_explainability(tree, feature_names)
    else:
        render_prediction(data, model, tree, feature_names, risk_index)

    st.divider()
    st.caption(
        "IALM Lab · M.Tech Engineering Project · Reproducible random_state=42 · "
        "Synthetic dataset for learning only."
    )


if __name__ == "__main__":
    main()
