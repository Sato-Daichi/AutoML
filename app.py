import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
import lightgbm as lgb
import streamlit as st

st.set_page_config(layout="wide")

st.title("AutoML")
col1, col2 = st.columns([1, 1], gap="medium")

col1.write("### 設定")
uploaded_train_file = col1.file_uploader("学習データをアップロード", type="csv")

if uploaded_train_file is not None:
    df = pd.read_csv(uploaded_train_file)
    cols = df.columns

    # 説明変数
    exp_select = col1.expander("説明変数を選択")
    selected = [True for _ in cols]
    for i, col in enumerate(cols):
        value = True if i < len(cols)-1 else False
        selected[i] = exp_select.checkbox(col, value=value)
    explanatory_cols = cols[selected].tolist()

    # 目的変数
    target_col = col1.selectbox("目的変数を選択", cols.tolist(), index=len(cols)-1)

    if target_col in explanatory_cols:
        col1.error(f"{target_col}は説明変数に含まれています")

    # 目的変数のタスク推定
    unique_values = np.sort(df[target_col].unique())
    if unique_values.tolist() == list(range(len(unique_values))):
        task = col1.selectbox("タスクを選択", ["2値分類", "回帰"])
    else:
        task = col1.selectbox("タスクを選択", ["回帰", "2値分類"])
    
    # その他詳細設定
    cv = col1.number_input('fold数', min_value=2, max_value=10, value=5, step=1)

    # 学習データ
    df = df[list(set(explanatory_cols+[target_col]))]
    col2.write("### データ")
    exp = col2.expander("データを見る")
    exp.write(f"レコード数 : {len(df)}  カラム数 : {df.shape[1]}")
    exp.write(df)
    
    if task != "" and col1.button("実行"):
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

        elif task == "2値分類":
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
        bar = col2.progress(0)
        latest_iteration = col2.empty()
        kf = KFold(cv, shuffle=True) if task == "回帰" else StratifiedKFold(cv, shuffle=True)
        for i, (model_name, model) in enumerate(models.items()):
            result = cross_validate(model, X, y, cv=kf, scoring=list(metrics.keys()))
            scores = []
            for k in metrics:
                scores.append(result["test_" + k].mean())
            table.append(scores)
            latest_iteration.text(f"{i+1}/{len(models)} : {model_name}")
            bar.progress(int(100/len(models) * (i+1)))

        latest_iteration.text("")
        col2.write("### 学習結果")
        table = pd.DataFrame(table)
        table.columns = metrics.values()
        table.index = models.keys()
        table = table.sort_values("R^2" if task == "回帰" else "Accuracy", ascending=False)
        col2.write(table)