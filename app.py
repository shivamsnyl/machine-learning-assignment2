import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import *

st.set_page_config(
    page_title="Income Classification App",
    layout="wide"
)

st.title("Income Classification Using ML Models")

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

selected_model = st.selectbox(
    "Select Model",
    [
        "logistic_regression",
        "decision_tree",
        "knn",
        "naive_bayes",
        "svm"
    ]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    X = df.drop("income", axis=1)

    y = df["income"]

    model = joblib.load(
        f"model/{selected_model}.pkl"
    )

    scaler = joblib.load(
        "model/scaler.pkl"
    )

    if selected_model in [
        "logistic_regression",
        "knn",
        "svm"
    ]:
        X = scaler.transform(X)

    pred = model.predict(X)

    prob = model.predict_proba(X)[:, 1]

    st.subheader("Metrics")

    st.write(
        "Accuracy:",
        round(
            accuracy_score(
                y,
                pred
            ),
            4
        )
    )

    st.write(
        "AUC:",
        round(
            roc_auc_score(
                y,
                prob
            ),
            4
        )
    )

    st.write(
        "Precision:",
        round(
            precision_score(
                y,
                pred
            ),
            4
        )
    )

    st.write(
        "Recall:",
        round(
            recall_score(
                y,
                pred
            ),
            4
        )
    )

    st.write(
        "F1:",
        round(
            f1_score(
                y,
                pred
            ),
            4
        )
    )

    st.write(
        "MCC:",
        round(
            matthews_corrcoef(
                y,
                pred
            ),
            4
        )
    )

    st.subheader(
        "Confusion Matrix"
    )

    st.write(
        confusion_matrix(
            y,
            pred
        )
    )

    st.subheader(
        "Classification Report"
    )

    st.text(
        classification_report(
            y,
            pred
        )
    )