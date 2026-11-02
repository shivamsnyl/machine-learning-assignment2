import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

os.makedirs("model", exist_ok=True)

df = pd.read_csv("adult.csv")

df.replace(" ?", pd.NA, inplace=True)
df.dropna(inplace=True)

encoders = {}

for col in df.select_dtypes(include="object").columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

test_df = X_test.copy()
test_df["income"] = y_test

test_df.to_csv("test_data.csv", index=False)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=3000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "Naive Bayes":
        GaussianNB(),

    "SVM":
        SVC(probability=True)
}

results = []

for name, model in models.items():

    use_scaling = name in [
        "Logistic Regression",
        "KNN",
        "SVM"
    ]

    Xtr = X_train_scaled if use_scaling else X_train
    Xte = X_test_scaled if use_scaling else X_test

    model.fit(Xtr, y_train)

    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]

    accuracy = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    mcc = matthews_corrcoef(y_test, pred)

    results.append([
        name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

    filename = (
        name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    joblib.dump(model, f"model/{filename}")

joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoders, "model/encoders.pkl")

result_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
)

print(result_df)

result_df.to_csv(
    "model_metrics.csv",
    index=False
)

print("\nSaved Successfully")