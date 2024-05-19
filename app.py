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

st.title("AutoML for Tabular Data")

# 2カラムレイアウト
col1, col2 = st.columns([1, 1], gap="medium")

col1.write("### 設定")
uploaded_csv_file = col1.file_uploader("CSVファイルをアップロード", type="csv")

if uploaded_csv_file is not None:
    df = pd.read_csv(uploaded_csv_file)
    cols = df.columns

    # 説明変数
    explanatory_cols_selector = col1.expander("説明変数を選択")
    is_selected = [True for _ in cols]
    
    # デフォルトでは末尾の変数を目的変数とする
    last_col_index = len(cols) - 1
    for i, col in enumerate(cols):
        value = False if i == last_col_index else True
        is_selected[i] = explanatory_cols_selector.checkbox(col, value=value)
    explanatory_cols: list = cols[is_selected].tolist()

    # 目的変数
    target_col = col1.selectbox("目的変数を選択", cols.tolist(), index=last_col_index)
    if target_col in explanatory_cols:
        col1.error(f"{target_col}は説明変数に含まれています")

    # 使用する質的変数ごとにエンコーディングを指定
    encoding_methods_selector = col1.expander("エンコーディングを選択")
    use_cols = explanatory_cols + [target_col]
    categorical_cols = df[use_cols].select_dtypes(include="object").columns.tolist()
    encoding_methods = {}
    for col in categorical_cols:
        encoding_methods[col] = encoding_methods_selector.selectbox(f"{col}のエンコーディングを選択", ["LabelEncoder", "OneHotEncoder"])
    
    # 目的変数のタスク推定
    task = col1.selectbox("目的変数のタスクを選択", ["回帰", "分類"])

    # cv数
    num_folds = col1.number_input('fold数', min_value=2, max_value=10, value=5, step=1)

    # 使用データ
    df = df[use_cols]
    col2.write("### データ")
    exp = col2.expander("データを見る")
    exp.write(f"レコード数 : {len(df)}  カラム数 : {df.shape[1]}")
    exp.write(df)

    match task:
        case "回帰":
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
            kf = KFold(num_folds, shuffle=True)

        case "分類":
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
            kf = StratifiedKFold(num_folds, shuffle=True)
    
    # モデルを選択
    selected_models = col1.multiselect("使用するモデルを選択", list(models.keys()))
    models = {k: v for k, v in models.items() if k in selected_models}

    if task != "" and col1.button("実行"):

        # データ
        df = df.dropna()

        # エンコーディング
        for col, method in encoding_methods.items():
            if method == "LabelEncoder":
                df[col] = df[col].astype("category").cat.codes
            elif method == "OneHotEncoder":
                df = pd.get_dummies(df, columns=[col])
        
        # 標準化
        df = (df - df.mean()) / df.std()
        
        X = df[explanatory_cols]
        y = df[target_col]

        # 交差検証
        table = []
        bar = col2.progress(0)
        latest_iteration = col2.empty()
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
