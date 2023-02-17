import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_validate
import lightgbm as lgb
import streamlit as st

st.title("AutoML")
uploaded_train_file = st.sidebar.file_uploader("学習データをアップロード", type="csv")

if uploaded_train_file is not None:
    df = pd.read_csv(uploaded_train_file)
    cols = df.columns.tolist()
    explanatory_cols = st.sidebar.multiselect("説明変数を選択", cols, default=cols[:-1])
    target_col = st.sidebar.selectbox("目的変数を選択", [cols[-1]] + cols[:-1])
    cv = st.sidebar.number_input('fold数', min_value=2, max_value=len(df), value=5, step=1)
    task = st.sidebar.selectbox("タスクを選択", ["", "回帰", "分類"])

    # 学習データ
    df = df[list(set(explanatory_cols+[target_col]))]
    st.write("### データ")
    st.write(f"レコード数 : {len(df)}  カラム数 : {df.shape[1]}")
    df
    
    if task != "" and st.sidebar.button("実行"):
        if task == "回帰":
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "KNN": KNeighborsRegressor(),
                "SVM": SVR(),
                "RandomForest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "LightGBM": lgb.LGBMRegressor(),
            }
            metrics = {
                "r2": "R^2",
                "neg_mean_squared_error": "MSE",
                "neg_mean_absolute_error": "MAE",
            }

        elif task == "分類":
            models = {
                "LogisticRegression": LogisticRegression(),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(),
                "RandomForest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "LightGBM": lgb.LGBMClassifier(),
            }
            metrics = {
                "accuracy": "Accuracy",
                "f1": "F1",
                "roc_auc": "AUC",
                "recall": "Recall",
                "precision": "Precision",
            }

        # データ
        df = df[list(set(explanatory_cols+[target_col]))]
        df = df.fillna(df.mean())
        X = df[explanatory_cols]
        y = df[target_col]

        # 交差検証
        table = []
        bar = st.progress(0)
        latest_iteration = st.empty()
        for i, (model_name, model) in enumerate(models.items()):
            latest_iteration.text(f"{i+1}/{len(models)} : {model_name}")
            bar.progress(int(100/len(models) * (i+1)))
            result = cross_validate(model, X, y, cv=cv, scoring=list(metrics.keys()))
            scores = []
            for k in metrics:
                scores.append(result["test_" + k].mean())
            table.append(scores)

        latest_iteration.text("")
        st.write("### 学習結果")
        table = pd.DataFrame(table)
        table.columns = metrics.values()
        table.index = models.keys()
        table = table.sort_values("R^2" if task == "回帰" else "Accuracy", ascending=False)
        table