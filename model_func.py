import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    PowerTransformer, RobustScaler, StandardScaler,
    OneHotEncoder, OrdinalEncoder, FunctionTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectFromModel


# ------------------------------------------------------------
# Drop Correlated Features
# ------------------------------------------------------------
# def drop_correlated_features(model_name, col_dic=column_dic):
drop_cols = ["days_birth", "amt_income_total", "years_employed", "days_employed", "age_binned"]
    # if model_name in ["SVC", "KNN"]:
    #     drop_cols.extend(["cnt_children", "cnt_fam_members"])
    #     drop_cols.extend
    # return drop_cols

# ------------------------------------------------------------
# Feature Selection Method
# ------------------------------------------------------------
def build_feature_selector(model_name):
    if model_name=="SVC":
        return RFE(SVC(kernel='linear'), n_features_to_select=None, step=0.2, importance_getter='feature_importances_')
    elif model_name=="XGB":
        return SelectFromModel(XGBClassifier(eval_metric="logloss", random_state=42), threshold='median')
    elif model_name=="KNN":
        return SelectKBest(score_func=mutual_info_classif, k=10)
# ------------------------------------------------------------
# Build Model
# ------------------------------------------------------------
def build_model(model_name):
    if model_name == "SVC":
        return "SVM (Linear)", SVC(kernel='linear', random_state=42)
    elif model_name == "XGB":
        return "XGB Classifier", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    elif model_name == "KNN":
        return "KNN", KNeighborsClassifier()
    else:
        raise ValueError("Unsupported model name")
    


# ------------------------------------------------------------
# 5. Model Training Pipeline
# ------------------------------------------------------------
def model_pipeline(model_name, train_df, test_df, target_col="label", random_state=42):
    # Drop correlated columns
    drop_cols = ["days_birth", "amt_income_total", "years_employed", "flag_mobil", "code_gender", "flag_own_realty", "flag_own_car", "cnt_fam_members"]
    train_df = train_df.drop(columns=drop_cols, errors='ignore')
    test_df = test_df.drop(columns=drop_cols, errors='ignore')

    # Split features and target
    X_train_full = train_df.drop(columns=[target_col])
    y_train_full = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Preprocessor
    # preprocessor = build_transformer()
    
    # Model
    name, model = build_model(model_name)
    print(f"\nTraining model: {name} using StratifiedKFold...")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    acc_scores, f1_scores, roc_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        pipeline = Pipeline([
            # ("preprocess", preprocessor),
            ("feature_selector", build_feature_selector(model_name)),
            ("classifier", model)
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        

        acc_scores.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        roc_scores.append(roc_auc_score(y_val, y_proba))

        print(f"Fold {fold}: Accuracy={acc_scores[-1]:.3f}, F1={f1_scores[-1]:.3f}, ROC-AUC={roc_scores[-1]:.3f}")

    results = {
        "model": name,
        "accuracy": np.mean(acc_scores),
        "f1_score": np.mean(f1_scores),
        "roc_auc": np.nanmean(roc_scores)
    }

    print(f"\nFinished training {name} across all folds.")
    print(f"Average Accuracy: {results['accuracy']:.3f}, F1: {results['f1_score']:.3f}, ROC-AUC: {results['roc_auc']:.3f}")

    return results, X_train_full, y_train_full, X_test, y_test
