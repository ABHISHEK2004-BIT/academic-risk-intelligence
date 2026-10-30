
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay, classification_report
)

# ---------------------------------------------------------------------
# Page setup and visual theme
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Academic Risk | Predictive Insights",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root { --bg:#0b1420; --panel:#172433; --panel2:#1d2c3b; --line:#314253;
        --muted:#a9b8c7; --cyan:#28c7c2; --blue:#2997f3; --lime:#b9e85a; }
.stApp { background: radial-gradient(circle at 15% 0%, #172a3a 0, #0b1420 42%, #09121c 100%); color:#edf4fb; }
[data-testid="stHeader"] { background:rgba(8,17,27,.88); }
[data-testid="stSidebar"] { background:#101d2a; border-right:1px solid #2b3d4d; }
.block-container { padding-top:1.1rem; padding-bottom:2rem; max-width:1600px; }
h1,h2,h3 { color:#f2f7fb !important; }
p, label, .stMarkdown { color:#d5e0e9; }
div[data-testid="stMetric"] {
 background:linear-gradient(145deg,#1b2b3b,#142231);
 border:1px solid #34485a; border-radius:13px; padding:14px 16px;
 box-shadow:0 5px 18px rgba(0,0,0,.18);
}
div[data-testid="stMetricLabel"] { color:#aebdca !important; }
div[data-testid="stMetricValue"] { color:#f5f9fc !important; }
div[data-testid="stMetricDelta"] { font-size:.82rem; }
.panel {
 background:linear-gradient(145deg,rgba(28,43,58,.98),rgba(18,31,44,.98));
 border:1px solid #34495b; border-radius:14px; padding:15px 17px 12px 17px;
 margin:0 0 14px 0; box-shadow:0 8px 24px rgba(0,0,0,.16);
}
.panel-title { font-size:1rem; font-weight:650; color:#f1f6fa; margin-bottom:10px; }
.small-muted { color:#9fb0bf; font-size:.82rem; }
div[data-baseweb="tab-list"] { gap:8px; }
button[data-baseweb="tab"] { background:#152433; border-radius:8px 8px 0 0; }
.stButton>button, .stFormSubmitButton>button {
 background:linear-gradient(90deg,#167fc7,#20b9b2); color:white; border:0; border-radius:9px;
 font-weight:650;
}
[data-testid="stDataFrame"] { border:1px solid #34495b; border-radius:10px; }
hr { border-color:#34485a; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "student_academic_risk_1000.csv"

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_models(data):
    X = data.drop(columns=["Academic_Risk"])
    y = data["Academic_Risk"]
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )
    models = {}
    for criterion in ["entropy", "gini"]:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                criterion=criterion, max_depth=5, min_samples_leaf=10,
                random_state=42
            ))
        ])
        pipe.fit(X_train, y_train)
        models[criterion] = pipe
    return X_train, X_test, y_train, y_test, models

def panel_title(title, note=None):
    st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>' +
                (f'<div class="small-muted">{note}</div>' if note else ''),
                unsafe_allow_html=True)

def panel_end():
    st.markdown("</div>", unsafe_allow_html=True)

def dark_fig(fig):
    fig.patch.set_facecolor("#172433")
    for ax in fig.axes:
        ax.set_facecolor("#172433")
        ax.tick_params(colors="#c7d3dd")
        ax.xaxis.label.set_color("#c7d3dd")
        ax.yaxis.label.set_color("#c7d3dd")
        ax.title.set_color("#f0f5f9")
        for spine in ax.spines.values():
            spine.set_color("#405466")
        ax.grid(color="#34485a", alpha=.45, linewidth=.6)
    fig.tight_layout()
    return fig

def draw_network():
    # Illustrative view of the fitted decision tree's preprocessing + tree stages.
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    layers = [
        (1.0, [1.0, 2.0, 3.0, 4.0], "#2997f3", "INPUT"),
        (3.5, [0.7, 1.7, 2.7, 3.7, 4.7], "#28c7c2", "FEATURE SPLITS"),
        (6.2, [1.0, 2.0, 3.0, 4.0], "#28c7c2", "TREE NODES"),
        (8.8, [1.5, 3.0, 4.5], "#b9e85a", "OUTPUT")
    ]
    for li in range(len(layers)-1):
        x1, ys1, _, _ = layers[li]
        x2, ys2, _, _ = layers[li+1]
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1+.15,x2-.15],[y1,y2],color="#5b91a7",alpha=.32,lw=.7,zorder=1)
    for x, ys, color, label in layers:
        for y in ys:
            ax.add_patch(Circle((x,y),.14,facecolor=color,edgecolor="#d7f4ff",lw=.8,zorder=3))
        ax.text(x,5.55,label,ha="center",color="#d8e5ee",fontsize=8,fontweight="bold")
    ax.text(9.35,3.0,"Risk\nclass",color="#d8e5ee",ha="left",va="center",fontsize=9)
    ax.text(5,.12,"Illustrative processing flow — the classifier is a decision tree, not a neural network.",
            color="#9fb0bf",ha="center",fontsize=8)
    return dark_fig(fig)

# ---------------------------------------------------------------------
# Data and model
# ---------------------------------------------------------------------
df = load_data()
X_train, X_test, y_train, y_test, models = train_models(df)
criterion = st.session_state.get("criterion", "entropy")
model = models[criterion]
tree = model.named_steps["classifier"]
X_all = df.drop(columns=["Academic_Risk"])
y_all = df["Academic_Risk"]
y_pred = model.predict(X_test)
prob = model.predict_proba(X_test)
risk_idx = list(tree.classes_).index("At Risk")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="At Risk", zero_division=0)
recall = recall_score(y_test, y_pred, pos_label="At Risk", zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label="At Risk", zero_division=0)
try:
    auc = roc_auc_score((y_test == "At Risk").astype(int), prob[:, risk_idx])
except ValueError:
    auc = float("nan")

# ---------------------------------------------------------------------
# Header and sidebar controls
# ---------------------------------------------------------------------
header_l, header_r = st.columns([5, 2])
with header_l:
    st.markdown("## 🎓  ACADEMIC RISK ANALYTICS  |  PREDICTIVE INSIGHTS")
    st.caption("IALM • Explainable Decision Tree Classification • M.Tech Engineering Lab")
with header_r:
    st.selectbox("Model criterion", ["entropy", "gini"], key="criterion")

with st.sidebar:
    st.markdown("## ◈ ANALYTICS PLATFORM")
    st.caption("Academic Risk Intelligence")
    view = st.radio("Dashboard section", [
        "Executive Overview", "Data Exploration", "Model Performance",
        "Explainability", "Live Prediction"
    ])
    st.divider()
    st.markdown("### Dataset controls")
    source = st.selectbox("Data view", ["All records", "Training split", "Test split"])
    st.caption("Dataset: student_academic_risk_1000.csv")
    st.caption("Rows: 1,000 • Target: Academic_Risk")
    st.divider()
    st.warning("Synthetic educational dataset. Not suitable for real student decisions.")

st.markdown(
    '<div class="small-muted">MODEL: Decision Tree  •  DATA SOURCE: Synthetic academic dataset  •  '
    'VALIDATION: Stratified 80/20 hold-out split</div>', unsafe_allow_html=True
)

# KPI strip
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("TOTAL RECORDS", f"{len(df):,}", "1,000-row dataset")
k2.metric("TEST ACCURACY", f"{accuracy*100:.1f}%", f"{criterion.title()} criterion")
k3.metric("PRECISION · AT RISK", f"{precision*100:.1f}%")
k4.metric("RECALL · AT RISK", f"{recall*100:.1f}%")
k5.metric("F1 · AT RISK", f"{f1*100:.1f}%")

st.divider()

# ---------------------------------------------------------------------
# Dashboard sections
# ---------------------------------------------------------------------
if view == "Executive Overview":
    left, right = st.columns([1.8, 1])
    with left:
        panel_title("DECISION TREE — MODEL PIPELINE", "Input features → encoded variables → decision rules → predicted academic risk")
        st.pyplot(draw_network(), use_container_width=True)
        panel_end()
    with right:
        panel_title("TEST PERFORMANCE BY CLASS", "Precision and recall calculated on the held-out test set")
        report = pd.DataFrame(classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )).T
        class_rows = report.loc[[c for c in tree.classes_ if c in report.index], ["precision","recall"]]
        fig, ax = plt.subplots(figsize=(5,3.8))
        x=np.arange(len(class_rows)); width=.32
        ax.bar(x-width/2,class_rows["precision"]*100,width,label="Precision",color="#2997f3")
        ax.bar(x+width/2,class_rows["recall"]*100,width,label="Recall",color="#28c7c2")
        ax.set_xticks(x); ax.set_xticklabels(class_rows.index,rotation=0)
        ax.set_ylim(0,105); ax.set_ylabel("Score (%)"); ax.legend(frameon=False)
        st.pyplot(dark_fig(fig), use_container_width=True)
        panel_end()

    lower1, lower2, lower3 = st.columns([1.35,1,1])
    with lower1:
        panel_title("MODEL ACCURACY SNAPSHOT", "Five-fold stratified cross-validation")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_validate(
            models[criterion], X_all, y_all, cv=cv,
            scoring=["accuracy"], n_jobs=None
        )["test_accuracy"]
        fig, ax = plt.subplots(figsize=(5.3,2.5))
        ax.plot(range(1,6),cv_scores*100,marker="o",lw=2.5,color="#28c7c2")
        ax.axhline(cv_scores.mean()*100,color="#b9e85a",ls="--",lw=1,label=f"Mean {cv_scores.mean()*100:.1f}%")
        ax.set_xticks(range(1,6)); ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0,105); ax.legend(frameon=False)
        st.pyplot(dark_fig(fig), use_container_width=True)
        st.caption(f"Mean CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        panel_end()
    with lower2:
        panel_title("PREDICTION DISTRIBUTION", "Predicted labels for held-out test records")
        counts=pd.Series(y_pred).value_counts()
        fig, ax=plt.subplots(figsize=(3.4,2.5))
        ax.pie(counts.values,labels=counts.index,autopct="%1.0f%%",
               colors=["#2997f3","#28c7c2"],textprops={"color":"#e6eef5","fontsize":8},
               wedgeprops={"width":.42,"edgecolor":"#172433"})
        st.pyplot(dark_fig(fig),use_container_width=True)
        panel_end()
    with lower3:
        panel_title("DATA QUALITY MONITOR", "Checks on the supplied CSV")
        st.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        st.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        st.metric("Feature columns", f"{df.shape[1]-1}")
        panel_end()

    panel_title("RECENT TEST-SET PREDICTIONS", "Sample rows from the held-out set; no real-time student records are connected")
    preview = X_test.head(8).copy()
    preview["Actual risk"] = y_test.head(8).values
    preview["Predicted risk"] = y_pred[:8]
    preview["At-risk probability"] = model.predict_proba(X_test.head(8))[:,risk_idx]
    preview["At-risk probability"] = preview["At-risk probability"].map(lambda v:f"{v:.1%}")
    st.dataframe(preview.reset_index(drop=True), use_container_width=True, hide_index=True)
    panel_end()

elif view == "Data Exploration":
    st.subheader("DATA EXPLORATION & QUALITY")
    q1,q2,q3,q4=st.columns(4)
    q1.metric("Rows",f"{len(df):,}"); q2.metric("Columns",len(df.columns))
    q3.metric("Missing cells",int(df.isna().sum().sum()))
    q4.metric("Duplicate rows",int(df.duplicated().sum()))
    st.markdown("### Dataset preview")
    st.dataframe(df.head(20),use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("### Target class balance")
        vc=df["Academic_Risk"].value_counts()
        fig,ax=plt.subplots(figsize=(5,3))
        ax.bar(vc.index,vc.values,color=["#2997f3","#28c7c2"])
        ax.set_ylabel("Records")
        st.pyplot(dark_fig(fig),use_container_width=True)
    with c2:
        st.markdown("### Program-wise record count")
        pc=df["Program"].value_counts()
        fig,ax=plt.subplots(figsize=(5,3))
        ax.barh(pc.index,pc.values,color="#28c7c2")
        ax.set_xlabel("Records")
        st.pyplot(dark_fig(fig),use_container_width=True)
    numeric=df.select_dtypes(include=np.number).columns.tolist()
    chosen=st.selectbox("Numerical feature distribution",numeric)
    fig,ax=plt.subplots(figsize=(8,3))
    ax.hist(df[chosen],bins=25,color="#2997f3",edgecolor="#dce8f2",alpha=.9)
    ax.set_xlabel(chosen); ax.set_ylabel("Frequency")
    st.pyplot(dark_fig(fig),use_container_width=True)
    st.markdown("### Correlation heatmap")
    corr=df[numeric].corr()
    fig,ax=plt.subplots(figsize=(9,5))
    im=ax.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns,rotation=65,ha="right",fontsize=8)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index,fontsize=8)
    fig.colorbar(im,ax=ax,fraction=.03)
    st.pyplot(dark_fig(fig),use_container_width=True)

elif view == "Model Performance":
    st.subheader("MODEL PERFORMANCE & VALIDATION")
    a,b=st.columns(2)
    with a:
        st.markdown("### Confusion matrix")
        fig,ax=plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay.from_predictions(y_test,y_pred,labels=tree.classes_,ax=ax,cmap="Blues",colorbar=False)
        st.pyplot(dark_fig(fig),use_container_width=True)
    with b:
        st.markdown("### Classification report")
        st.dataframe(pd.DataFrame(classification_report(
            y_test,y_pred,output_dict=True,zero_division=0
        )).T.round(3),use_container_width=True)
    st.markdown("### Entropy vs Gini comparison")
    rows=[]
    for name,pipe in models.items():
        p=pipe.predict(X_test)
        rows.append({"Criterion":name,"Accuracy":accuracy_score(y_test,p),
                     "Macro precision":precision_score(y_test,p,average="macro",zero_division=0),
                     "Macro recall":recall_score(y_test,p,average="macro",zero_division=0),
                     "Macro F1":f1_score(y_test,p,average="macro",zero_division=0)})
    st.dataframe(pd.DataFrame(rows).set_index("Criterion").round(3),use_container_width=True)
    st.caption("Both models use the same stratified 80/20 split, max_depth=5, min_samples_leaf=10 and random_state=42.")
    st.markdown("### ROC-AUC")
    st.metric("ROC-AUC · At Risk",f"{auc:.3f}" if np.isfinite(auc) else "Not available")

elif view == "Explainability":
    st.subheader("MODEL EXPLAINABILITY")
    names=model.named_steps["preprocessor"].get_feature_names_out()
    importance=pd.Series(tree.feature_importances_,index=names).sort_values(ascending=False)
    c1,c2=st.columns([1,1.4])
    with c1:
        st.markdown("### Top feature importance")
        st.bar_chart(importance.head(12),color="#28c7c2")
    with c2:
        st.markdown("### Decision tree visualization")
        fig,ax=plt.subplots(figsize=(10,6))
        plot_tree(tree,feature_names=names,class_names=tree.classes_,filled=True,
                  rounded=True,max_depth=4,fontsize=7,ax=ax)
        st.pyplot(dark_fig(fig),use_container_width=True)
    st.markdown("### Extracted decision rules")
    st.code(export_text(tree,feature_names=list(names),max_depth=5),language="text")
    st.caption("Feature importance describes this fitted synthetic-data model; it does not establish causation.")

else:
    st.subheader("LIVE PREDICTION — HYPOTHETICAL STUDENT PROFILE")
    st.info("Enter an illustrative profile to see how the trained decision tree classifies it. Do not enter identifiable personal information.")
    with st.form("prediction_form"):
        c1,c2,c3=st.columns(3)
        with c1:
            age=st.number_input("Age",18,25,20)
            attendance=st.slider("Attendance (%)",35.0,100.0,70.0)
            study=st.slider("Weekly study hours",0.0,12.0,4.0)
        with c2:
            previous=st.slider("Previous exam score",15.0,100.0,60.0)
            assignments=st.slider("Assignments completed (out of 10)",0,10,6)
            sleep=st.slider("Average sleep hours",3.0,10.0,7.0)
        with c3:
            internet=st.selectbox("Internet access",["Good","Average","Poor"])
            participation=st.selectbox("Class participation",["High","Medium","Low"])
            program=st.selectbox("Program",sorted(df["Program"].unique()))
        submit=st.form_submit_button("GENERATE PREDICTION",type="primary")
    if submit:
        sample=pd.DataFrame([{
            "Age":age,"Attendance_Percent":attendance,"Weekly_Study_Hours":study,
            "Previous_Exam_Score":previous,"Assignments_Completed_Out_of_10":assignments,
            "Average_Sleep_Hours":sleep,"Internet_Access":internet,
            "Class_Participation":participation,"Program":program
        }])
        start=perf_counter()
        label=model.predict(sample)[0]
        probabilities=model.predict_proba(sample)[0]
        elapsed=(perf_counter()-start)*1000
        p1,p2,p3=st.columns(3)
        p1.metric("PREDICTED CLASS",label)
        p2.metric("AT-RISK PROBABILITY",f"{probabilities[risk_idx]:.1%}")
        p3.metric("INFERENCE TIME",f"{elapsed:.2f} ms")
        st.markdown("### Class probabilities")
        st.bar_chart(pd.Series(probabilities,index=tree.classes_),color="#2997f3")
        transformed=model.named_steps["preprocessor"].transform(sample)
        path=tree.decision_path(transformed)
        node_ids=path.indices[path.indptr[0]:path.indptr[1]]
        st.markdown("### Decision path")
        for node in node_ids[:-1]:
            feature_id=tree.tree_.feature[node]
            threshold=tree.tree_.threshold[node]
            feature_name=names[feature_id] if feature_id>=0 else "leaf"
            st.write(f"Node {node}: `{feature_name}` ≤ `{threshold:.3f}`")
        st.caption("This is a model output from synthetic teaching data, not a validated academic-risk assessment.")

st.divider()
st.caption("IALM Lab • M.Tech Engineering Project • Reproducible random_state=42 • Synthetic dataset for learning only")
