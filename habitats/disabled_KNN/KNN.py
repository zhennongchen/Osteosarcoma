import argparse
import json
import os
from itertools import product

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


DEFAULT_RANDOM_STATE = 0
DEFAULT_TASK = "Prognosis"
TASK_TO_LABEL_COL = {
    "Prognosis": "Prognosis_label",
    "Pathologic": "Pathologic_label",
}
N_SPLITS = 5
TRAIN_FOLDS = [0, 1, 2, 3,4]
INTERNAL_TEST_FOLD = 5
EXTERNAL_TEST_FOLD = 6
LABEL_COL = TASK_TO_LABEL_COL[DEFAULT_TASK]
NON_FEATURE_COLS = ["Patient_set", "Patient_index", "Image_filepath", "Mask_filepath"]
ID_COLS = ["Patient_set", "Patient_index", "Image_filepath", "Mask_filepath"]
RFECV_MAX_FEATURES = 35
LASSO_MAX_FEATURES = 35
SPLIT_COL = "split"
FOLD_COL = "fold"
PATIENT_LIST_PATH = "/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set123.xlsx"
SPLIT_OUT_PATH_TEMPLATE = (
    "/host/e/D/Data/Habitats/Jishuitan/Patient_lists/"
    "image_label_info_set123_5fold_{task_lower}_random{random_state}.xlsx"
)
DEFAULT_PCC_RADIOMICS_PATH = "/host/d/projects/Habitats/radiomics/habitats_individual/habitat_radiomics_measurements_avg_PCC.xlsx"
RADIOMICS_OUT_DIR = "/host/d/projects/Habitats/radiomics/habitats"
MODEL_ROOT = "/host/d/projects/Habitats/models"
DEFAULT_IMAGE_TYPE = "habitats_individual"
CLASSIFIER_ARG = "KNN"
CLASSIFIER_NAME = "KNN"
CLASSIFIER_DIR = "KNN"
MODEL_LABEL = "KNN"
SELECTOR_ARG = "knn_feature_selector"
DEFAULT_SELECTOR = "lasso"
SELECTED_PREFIX = "habitat_radiomics_measurements"
METRIC_KEYS = ["auc", "auc_ci_low", "auc_ci_high", "accuracy", "sensitivity", "specificity"]
FEATURE_SELECTION_SCOPE = "all_set123_train_internal_external"


class SkipExperiment(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def parse_top_k(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "none":
        if True:
            return None
        raise argparse.ArgumentTypeError("top_k cannot be None for this classifier")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("top_k must be an integer" + (" or None" if True else "")) from exc


def top_k_label(top_k):
    if top_k is None or (isinstance(top_k, str) and top_k.lower() == "none"):
        return "none"
    return f"top{top_k}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run habitat radiomics experiments.")
    parser.add_argument("--task", choices=sorted(TASK_TO_LABEL_COL), default=DEFAULT_TASK)
    parser.add_argument("--pcc_radiomics_path", default=DEFAULT_PCC_RADIOMICS_PATH)
    parser.add_argument("--image_type", choices=["habitats_individual", "habitats_avg", "habitats_sum"], default=DEFAULT_IMAGE_TYPE)
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--gridsearch_range", choices=["train", "all"], default="train", help="Use train data or all data for hyperparameter GridSearchCV.")
    parser.add_argument("--classifier", choices=[CLASSIFIER_ARG], default=CLASSIFIER_ARG)
    parser.add_argument("--knn_feature_selector", choices=['lasso'], default=DEFAULT_SELECTOR)
    parser.add_argument("--top_k", type=parse_top_k, default=20, help="Number of selected features. Use None to keep all non-zero LASSO features.")
    return parser.parse_args()


def get_label_col(task):
    return TASK_TO_LABEL_COL[task]


def get_model_out_dir(task, image_type):
    return os.path.join(MODEL_ROOT, task, image_type)


def get_select_out_dir(image_type):
    if image_type == "habitats_individual":
        return "/host/d/projects/Habitats/radiomics/habitats_individual/select"
    if image_type == "habitats_avg":
        return "/host/d/projects/Habitats/radiomics/habitats/select_avg"
    if image_type == "habitats_sum":
        return "/host/d/projects/Habitats/radiomics/habitats/select_sum"
    raise ValueError(f"Unsupported image_type for select dir: {image_type}")



def load_patient_split(random_state, task):
    label_col = get_label_col(task)
    split_path = SPLIT_OUT_PATH_TEMPLATE.format(task_lower=task.lower(), random_state=random_state)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Precomputed split file does not exist. Run patient_split.ipynb first: {split_path}")
    df = pd.read_excel(split_path)
    required_cols = ["Patient_set", "Patient_index", label_col, SPLIT_COL, FOLD_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Split file is missing required columns: {missing}")
    df[label_col] = df[label_col].astype(int)
    df[FOLD_COL] = df[FOLD_COL].astype(int)

    train_mask = df[SPLIT_COL].eq("train") & df[FOLD_COL].isin(TRAIN_FOLDS)
    internal_test_mask = df[SPLIT_COL].eq("internal test") & df[FOLD_COL].eq(INTERNAL_TEST_FOLD)
    external_test_mask = df[SPLIT_COL].eq("external test") & df[FOLD_COL].eq(EXTERNAL_TEST_FOLD)
    accounted = train_mask | internal_test_mask | external_test_mask
    if int(accounted.sum()) != len(df):
        bad_rows = df.loc[~accounted, ["Patient_set", "Patient_index", SPLIT_COL, FOLD_COL]].head(10)
        raise ValueError(f"Rows outside expected train/internal/external split found. Examples:\n{bad_rows}")

    print("Loaded patient split:", split_path)
    print(
        "Train cases:", int(train_mask.sum()),
        "Internal test cases:", int(internal_test_mask.sum()),
        "External test cases:", int(external_test_mask.sum()),
    )
    print(
        "Train positive fraction:", f"{df.loc[train_mask, label_col].mean():.4f}",
        "Internal test positive fraction:", f"{df.loc[internal_test_mask, label_col].mean():.4f}",
        "External test positive fraction:", f"{df.loc[external_test_mask, label_col].mean():.4f}",
    )
    return df, split_path

def load_features_and_labels(radiomics_path, labels_df):
    radiomics_df = pd.read_excel(radiomics_path)
    required_label_cols = ["Patient_set", "Patient_index", SPLIT_COL, FOLD_COL, LABEL_COL]
    missing_label_cols = [c for c in required_label_cols if c not in labels_df.columns]
    if missing_label_cols:
        raise ValueError(f"Missing columns in label table: {missing_label_cols}")
    missing_radiomics_cols = [c for c in ["Patient_set", "Patient_index"] if c not in radiomics_df.columns]
    if missing_radiomics_cols:
        raise ValueError(f"Missing columns in radiomics table: {missing_radiomics_cols}")
    label_cols = ["Patient_set", "Patient_index", SPLIT_COL, FOLD_COL, LABEL_COL]
    merged_df = radiomics_df.merge(labels_df[label_cols], on=["Patient_set", "Patient_index"], how="inner", validate="one_to_one").reset_index(drop=True)
    if len(merged_df) != len(radiomics_df) or len(merged_df) != len(labels_df):
        raise ValueError(f"Radiomics and labels are not a complete one-to-one match: radiomics={len(radiomics_df)}, labels={len(labels_df)}, merged={len(merged_df)}")
    feature_cols = [c for c in radiomics_df.columns if c not in NON_FEATURE_COLS]
    X = merged_df[feature_cols].values
    y = merged_df[LABEL_COL].astype(int).values
    folds = merged_df[FOLD_COL].astype(int).values
    print(f"Feature matrix shape: {X.shape}", f"Label vector shape: {y.shape}", f"Fold vector shape: {folds.shape}")
    return radiomics_df, merged_df, feature_cols, X, y, folds


def scale_pos_weight_from_y(y):
    n_pos = np.sum(y == 1)
    if n_pos == 0:
        raise ValueError("No positive labels found; cannot compute scale_pos_weight.")
    return float(np.sum(y == 0) / n_pos)


def make_estimator(random_state, y_for_weight=None, **params):
    if CLASSIFIER_ARG == "SVM":
        return Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="linear", class_weight="balanced", probability=True, random_state=random_state, **params))])
    if CLASSIFIER_ARG == "LR":
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000, random_state=random_state, **params))])
    if CLASSIFIER_ARG == "RF":
        return RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=1, **params)
    if CLASSIFIER_ARG == "KNN":
        return Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(**params))])
    if CLASSIFIER_ARG == "XGBoost":
        spw = scale_pos_weight_from_y(y_for_weight) if y_for_weight is not None else 1.0
        return XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", random_state=random_state, n_jobs=1, scale_pos_weight=spw, **params)
    raise ValueError(f"Unsupported classifier: {CLASSIFIER_ARG}")


def get_param_grid():
    if CLASSIFIER_ARG == "SVM":
        return {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__tol": [1e-4, 1e-3]}
    if CLASSIFIER_ARG == "LR":
        return {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__tol": [1e-4, 1e-3]}
    if CLASSIFIER_ARG == "RF":
        return {"n_estimators": [100, 300, 500], "max_depth": [None, 3, 5], "max_features": ["sqrt", "log2"]}
    if CLASSIFIER_ARG == "KNN":
        return {"clf__n_neighbors": [3, 5, 7, 9, 11], "clf__weights": ["uniform", "distance"]}
    if CLASSIFIER_ARG == "XGBoost":
        return {"n_estimators": [50, 100, 200], "max_depth": [3, 4, 5], "learning_rate": [0.03, 0.1]}
    raise ValueError(f"Unsupported classifier: {CLASSIFIER_ARG}")


def selection_estimator(random_state, y):
    if CLASSIFIER_ARG == "SVM":
        return make_estimator(random_state=random_state, C=1.0, tol=1e-3)
    if CLASSIFIER_ARG == "LR":
        return make_estimator(random_state=random_state, C=1.0, tol=1e-3)
    if CLASSIFIER_ARG == "RF":
        return make_estimator(random_state=random_state, n_estimators=300, max_depth=5, max_features="sqrt")
    if CLASSIFIER_ARG == "KNN":
        return make_estimator(random_state=random_state, n_neighbors=5, weights="uniform")
    if CLASSIFIER_ARG == "XGBoost":
        return make_estimator(random_state=random_state, y_for_weight=y, n_estimators=100, max_depth=5, learning_rate=0.1)
    raise ValueError(f"Unsupported classifier: {CLASSIFIER_ARG}")


def importance_getter_for_selector():
    if CLASSIFIER_ARG in {"SVM", "LR"}:
        return "named_steps.clf.coef_"
    return "auto"


def select_lasso_features(top_k, feature_cols, X, y, random_state):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LogisticRegressionCV(Cs=30, cv=cv, penalty="l1", solver="liblinear", scoring="roc_auc", max_iter=5000, refit=True, n_jobs=1, random_state=random_state)),
    ])
    model.fit(X, y)
    coef = model.named_steps["lasso"].coef_.ravel()
    abs_coef = np.abs(coef)
    nonzero_idx = np.where(coef != 0)[0]
    nonzero_sorted_idx = nonzero_idx[np.argsort(abs_coef[nonzero_idx])[::-1]]
    if top_k is None:
        selected_idx = nonzero_sorted_idx
        if len(selected_idx) == 0:
            raise SkipExperiment("LASSO selected 0 non-zero features with top_k=None.")
        if len(selected_idx) > LASSO_MAX_FEATURES:
            raise SkipExperiment(f"LASSO selected {len(selected_idx)} non-zero features, which exceeds the hard limit of {LASSO_MAX_FEATURES}.")
    elif len(nonzero_sorted_idx) >= top_k:
        selected_idx = nonzero_sorted_idx[:top_k]
    else:
        remaining_idx = [i for i in np.argsort(abs_coef)[::-1] if i not in set(nonzero_sorted_idx)]
        selected_idx = np.array(list(nonzero_sorted_idx) + remaining_idx[: top_k - len(nonzero_sorted_idx)])
    selected_features = [feature_cols[i] for i in selected_idx]
    return selected_features


def select_features(feature_selector, top_k, feature_cols, X, y, random_state):
    if feature_selector == "lasso":
        return select_lasso_features(top_k, feature_cols, X, y, random_state)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    estimator = selection_estimator(random_state, y)
    if feature_selector == "rfe":
        selector = RFE(estimator=estimator, n_features_to_select=top_k, step=1, importance_getter=importance_getter_for_selector())
    elif feature_selector == "rfecv":
        selector = RFECV(estimator=estimator, step=1, cv=cv, scoring="roc_auc", n_jobs=1, importance_getter=importance_getter_for_selector())
    else:
        raise ValueError(f"Unsupported feature selector: {feature_selector}")
    selector.fit(X, y)
    selected_features = [f for f, keep in zip(feature_cols, selector.get_support()) if keep]
    if feature_selector == "rfecv" and len(selected_features) > RFECV_MAX_FEATURES:
        raise SkipExperiment(f"RFECV selected {len(selected_features)} features, which exceeds the hard limit of {RFECV_MAX_FEATURES}.")
    return selected_features


def get_selected_feature_path(task, random_state, feature_selector, top_k, image_type):
    suffix = f"{task}_random{random_state}_{feature_selector}"
    if feature_selector in {"rfe"}:
        suffix += f"_top{top_k}"
    elif feature_selector == "lasso":
        suffix += f"_{top_k_label(top_k)}"
    select_dir = get_select_out_dir(image_type)
    os.makedirs(select_dir, exist_ok=True)
    selector_file_classifier = "LR" if feature_selector == "lasso" else CLASSIFIER_ARG
    return os.path.join(select_dir, f"{SELECTED_PREFIX}_{selector_file_classifier}_{suffix}_selected.xlsx")


def get_feature_cols_from_selected_table(selected_df):
    return [c for c in selected_df.columns if c not in NON_FEATURE_COLS]


def save_selected_features(radiomics_df, selected_features, task, random_state, feature_selector, top_k, image_type):
    selected_out_path = get_selected_feature_path(task, random_state, feature_selector, top_k, image_type)
    radiomics_df[NON_FEATURE_COLS + selected_features].copy().to_excel(selected_out_path, index=False)
    print("Saved selected feature table:", selected_out_path)
    return selected_out_path


def load_or_select_features(radiomics_df, feature_selector, top_k, feature_cols, X_all, y_all, task, random_state, image_type):
    selected_path = get_selected_feature_path(task, random_state, feature_selector, top_k, image_type)
    if os.path.exists(selected_path):
        selected_df = pd.read_excel(selected_path)
        selected_features = get_feature_cols_from_selected_table(selected_df)
        if selected_features:
            print("Loaded existing selected feature table:", selected_path)
            print(f"Selected features by {feature_selector}: {len(selected_features)}")
            return selected_path, selected_features
        print("Existing selected feature table has no feature columns; regenerating:", selected_path)
    selected_features = select_features(feature_selector, top_k, feature_cols, X_all, y_all, random_state)
    selected_path = save_selected_features(radiomics_df, selected_features, task, random_state, feature_selector, top_k, image_type)
    return selected_path, selected_features


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def bootstrap_auc_ci(y_true, y_score, n_bootstrap=2000, ci=0.95, random_state=0):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    auc_values = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[sample_idx])) < 2:
            continue
        auc_values.append(roc_auc_score(y_true[sample_idx], y_score[sample_idx]))
    if not auc_values:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    low, high = np.percentile(auc_values, [100 * alpha, 100 * (1.0 - alpha)])
    return float(low), float(high)


def bootstrap_mean_ci(values, n_bootstrap=2000, ci=0.95, random_state=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    boot_means = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        boot_means.append(float(np.mean(values[sample_idx])))
    alpha = (1.0 - ci) / 2.0
    low, high = np.percentile(boot_means, [100 * alpha, 100 * (1.0 - alpha)])
    return float(low), float(high)


def binary_metrics(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    auc_value = safe_auc(y_true, y_score)
    auc_ci_low, auc_ci_high = bootstrap_auc_ci(y_true, y_score)
    if len(np.unique(y_true)) < 2:
        threshold = 0.5
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        valid_mask = np.isfinite(thresholds)
        if valid_mask.any():
            valid_idx = np.where(valid_mask)[0]
            best_idx = int(valid_idx[int(np.argmax(tpr[valid_mask] - fpr[valid_mask]))])
        else:
            best_idx = int(np.argmax(tpr - fpr))
        threshold = float(thresholds[best_idx])
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {
        "auc": auc_value,
        "auc_ci_low": auc_ci_low,
        "auc_ci_high": auc_ci_high,
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) > 0 else float("nan"),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def prefixed_metrics(prefix, metrics):
    return {f"{prefix}_{key}": float(metrics[key]) for key in METRIC_KEYS}


def plot_roc_curve(y_true, y_score, title, save_path, figsize=(5, 5)):
    if len(np.unique(y_true)) < 2:
        print("Skipping ROC curve because y_true contains only one class:", save_path)
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=13); plt.ylabel("True Positive Rate", fontsize=13)
    plt.title(title, fontsize=14); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_cv_roc(cv_pred_df, title, save_path, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    plotted = False
    for fold_id in sorted(cv_pred_df[FOLD_COL].unique()):
        fold_df = cv_pred_df[cv_pred_df[FOLD_COL] == fold_id]
        y_true = fold_df[LABEL_COL].astype(int).values
        y_score = fold_df["pred_prob"].values
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, lw=1.5, alpha=0.85, label=f"Fold {fold_id} (AUC = {auc_value:.3f})")
        plotted = True
    if not plotted:
        plt.close(); print("Skipping CV mean ROC because no fold contains both classes:", save_path); return
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=13); plt.ylabel("True Positive Rate", fontsize=13)
    plt.title(title, fontsize=14); plt.legend(loc="lower right", fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def get_experiment_name(random_state, feature_selector, top_k):
    if feature_selector == "lasso":
        return f"random{random_state}_{feature_selector}_{top_k_label(top_k)}"
    experiment_name = f"random{random_state}_{feature_selector}"
    if feature_selector in {"rfe"}:
        experiment_name += f"_top{top_k}"
    return experiment_name


def write_skip_file(out_dir, args, feature_selector, reason):
    os.makedirs(out_dir, exist_ok=True)
    skip_info = {"classifier": CLASSIFIER_NAME, "task": args.task, "label_col": LABEL_COL, "random_state": args.random_state, "feature_selector": feature_selector, "top_k": None if feature_selector == "rfecv" else args.top_k, "status": "skipped", "reason": reason}
    skip_path = os.path.join(out_dir, "SKIPPED.json")
    with open(skip_path, "w") as f:
        json.dump(skip_info, f, indent=4)
    print("Skipped experiment:", reason)
    print("Saved skip record:", skip_path)



def expected_completed_artifacts(out_dir):
    artifacts = [
        "summary.json",
        "best_params.json",
        "cv_predictions.xlsx",
        "cv_metrics.xlsx",
        "cv_fold_metrics.xlsx",
        "internal_test_predictions.xlsx",
        "internal_test_metrics.xlsx",
        "external_test_predictions.xlsx",
        "external_test_metrics.xlsx",
        "grid_search_results.xlsx",
        "selected_features.xlsx",
        "alldata_model.joblib",
    ]
    artifacts.extend([f"fold{fold_id}_model.joblib" for fold_id in TRAIN_FOLDS])
    artifacts.extend([f"fold{fold_id}_allotherdata_model.joblib" for fold_id in TRAIN_FOLDS])
    return [os.path.join(out_dir, artifact) for artifact in artifacts]

def completed_experiment_exists(out_dir):
    if not all(os.path.exists(path) for path in expected_completed_artifacts(out_dir)):
        return False
    summary_path = os.path.join(out_dir, "summary.json")
    try:
        summary = load_json_file(summary_path)
    except (OSError, json.JSONDecodeError):
        return False
    return summary.get("feature_selection_scope") == FEATURE_SELECTION_SCOPE




def train_experiment_artifact_paths(out_dir):
    return {
        "model": os.path.join(out_dir, "alltraindata_model.joblib"),
        "predictions": os.path.join(out_dir, "train_predictions.xlsx"),
        "metrics": os.path.join(out_dir, "train_metrics.xlsx"),
        "roc": os.path.join(out_dir, f"ROC_curve_train_alltraindata_{CLASSIFIER_ARG}.pdf"),
    }


def train_experiment_exists(out_dir):
    paths = train_experiment_artifact_paths(out_dir)
    if not all(os.path.exists(path) for path in paths.values()):
        return False
    summary_path = os.path.join(out_dir, "summary.json")
    try:
        summary = load_json_file(summary_path)
    except (OSError, json.JSONDecodeError):
        return False
    return "train_auc" in summary

def load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)



def print_completed_summary(out_dir):
    summary = load_json_file(os.path.join(out_dir, "summary.json"))
    best_info = load_json_file(os.path.join(out_dir, "best_params.json"))
    print(f"\n========== Existing {MODEL_LABEL} Summary ==========")
    print("Existing completed experiment found. Reusing saved artifacts.")
    print("Output directory:", out_dir)
    print("Selected feature table:", summary.get("selected_feature_table", ""))
    print("Selected features:", summary.get("selected_feature_count", ""))
    print("Feature selection scope:", summary.get("feature_selection_scope", ""))
    print("Best params:", best_info.get("best_params", summary.get("best_params", "")))
    print("CV final selected method:", summary.get("cv_final_selected_method", ""))
    print("CV final AUC:", f"{summary.get('cv_final_auc', float('nan')):.4f}")
    if "train_auc" in summary:
        print("Train AUC:", f"{summary.get('train_auc', float('nan')):.4f}")
    else:
        print("Train experiment: missing")
    print("Internal test final selected method:", summary.get("internal_test_final_selected_method", ""))
    print("Internal test final AUC:", f"{summary.get('internal_test_final_auc', float('nan')):.4f}")
    print("External test final selected method:", summary.get("external_test_final_selected_method", ""))
    print("External test final AUC:", f"{summary.get('external_test_final_auc', float('nan')):.4f}")

def estimator_params_from_best_params(best_params):
    params = {}
    for key, value in best_params.items():
        if key.startswith("clf__"):
            params[key.split("clf__", 1)[1]] = value
        else:
            params[key] = value
    return params


def train_fixed_model(X_train, y_train, random_state, best_params):
    estimator_params = estimator_params_from_best_params(best_params)
    model = make_estimator(random_state=random_state, y_for_weight=y_train, **estimator_params)
    model.fit(X_train, y_train)
    return model


def metrics_row(name, metrics, extra=None):
    row = {"name": name}
    if extra:
        row.update(extra)
    for key in ["auc", "auc_ci_low", "auc_ci_high", "accuracy", "sensitivity", "specificity", "threshold", "tp", "fp", "tn", "fn"]:
        row[key] = metrics.get(key, "")
    return row



def evaluate_holdout_set(
    dataset_name,
    display_name,
    out_dir,
    merged_df,
    X_selected,
    holdout_idx,
    alldata_model,
    alldata_model_path,
):
    pred_df = merged_df.loc[holdout_idx, ID_COLS + [SPLIT_COL, FOLD_COL, LABEL_COL]].copy()
    y_holdout = pred_df[LABEL_COL].astype(int).values
    method_rows = []
    fold_metrics = []

    for fold_id in TRAIN_FOLDS:
        fold_model = joblib.load(os.path.join(out_dir, f"fold{fold_id}_model.joblib"))
        prob = fold_model.predict_proba(X_selected[holdout_idx])[:, 1]
        prob_col = f"prob_fold{fold_id}_model"
        pred_df[prob_col] = prob
        method_metrics = binary_metrics(y_holdout, prob)
        fold_metrics.append((fold_id, method_metrics, prob_col))
        method_rows.append(metrics_row(f"fold{fold_id}_model", method_metrics, extra={"method": "fold_model", "fold_model": fold_id}))

    fold_prob_cols = [f"prob_fold{fold_id}_model" for fold_id in TRAIN_FOLDS]
    pred_df["prob_mean"] = pred_df[fold_prob_cols].mean(axis=1)
    mean_metrics = binary_metrics(y_holdout, pred_df["prob_mean"].values)
    method_rows.append(metrics_row("mean", mean_metrics, extra={"method": "mean"}))

    best_fold_id, best_metrics, best_prob_col = max(fold_metrics, key=lambda item: item[1]["auc"])
    pred_df["best_model_fold"] = best_fold_id
    pred_df["prob_best"] = pred_df[best_prob_col]
    method_rows.append(metrics_row("best", best_metrics, extra={"method": "best", "best_model_fold": best_fold_id}))

    pred_df["prob_alldata"] = alldata_model.predict_proba(X_selected[holdout_idx])[:, 1]
    alldata_metrics = binary_metrics(y_holdout, pred_df["prob_alldata"].values)
    method_rows.append(metrics_row("alldata", alldata_metrics, extra={"method": "alldata", "model_path": alldata_model_path}))

    final_selected_method, final_metrics = max(
        [("mean", mean_metrics), ("best", best_metrics), ("alldata", alldata_metrics)],
        key=lambda item: item[1]["auc"],
    )
    if final_selected_method == "mean":
        pred_df["prob_final"] = pred_df["prob_mean"]
    elif final_selected_method == "best":
        pred_df["prob_final"] = pred_df["prob_best"]
    else:
        pred_df["prob_final"] = pred_df["prob_alldata"]
    pred_df["final_selected_method"] = final_selected_method
    method_rows.append(metrics_row("final", final_metrics, extra={"method": "final", "selected_method": final_selected_method}))

    pred_path = os.path.join(out_dir, f"{dataset_name}_predictions.xlsx")
    pred_df = pred_df.sort_values(["Patient_set", "Patient_index"])
    pred_df.to_excel(pred_path, index=False)

    metrics_path = os.path.join(out_dir, f"{dataset_name}_metrics.xlsx")
    pd.DataFrame(method_rows).to_excel(metrics_path, index=False)

    final_roc_path = os.path.join(out_dir, f"ROC_curve_{dataset_name}_final_{CLASSIFIER_ARG}.pdf")
    plot_roc_curve(
        pred_df[LABEL_COL].astype(int).values,
        pred_df["prob_final"].values,
        title=f"{display_name} Final ROC - {CLASSIFIER_ARG}",
        save_path=final_roc_path,
    )

    return {
        "predictions": pred_path,
        "metrics": metrics_path,
        "final_roc": final_roc_path,
        "final_selected_method": final_selected_method,
        "best_selected_model_fold": int(best_fold_id),
        "final_metrics": final_metrics,
        "mean_metrics": mean_metrics,
        "best_metrics": best_metrics,
        "alldata_metrics": alldata_metrics,
    }



def run_train_experiment(out_dir, merged_df, X_selected, y, train_idx_all, random_state, best_params):
    paths = train_experiment_artifact_paths(out_dir)
    model = train_fixed_model(X_selected[train_idx_all], y[train_idx_all], random_state, best_params)
    joblib.dump(model, paths["model"])

    pred_df = merged_df.loc[train_idx_all, ID_COLS + [SPLIT_COL, FOLD_COL, LABEL_COL]].copy()
    pred_df["prob_train"] = model.predict_proba(X_selected[train_idx_all])[:, 1]
    train_metrics = binary_metrics(pred_df[LABEL_COL].astype(int).values, pred_df["prob_train"].values)

    pred_df = pred_df.sort_values([FOLD_COL, "Patient_set", "Patient_index"])
    pred_df.to_excel(paths["predictions"], index=False)

    pd.DataFrame([
        metrics_row(
            "train",
            train_metrics,
            extra={
                "method": "alltraindata_model_on_train",
                "model_path": paths["model"],
                "train_size": int(len(train_idx_all)),
            },
        )
    ]).to_excel(paths["metrics"], index=False)

    plot_roc_curve(
        pred_df[LABEL_COL].astype(int).values,
        pred_df["prob_train"].values,
        title=f"Train All-Train-Data ROC - {CLASSIFIER_ARG}",
        save_path=paths["roc"],
    )

    return {
        "model": model,
        "model_path": paths["model"],
        "predictions": paths["predictions"],
        "metrics": paths["metrics"],
        "roc": paths["roc"],
        "train_metrics": train_metrics,
    }


def run_train_only_for_completed_experiment(args, labels_df, split_path, out_dir):
    _, merged_df, feature_cols, X, y, folds = load_features_and_labels(args.pcc_radiomics_path, labels_df)

    train_mask = merged_df[SPLIT_COL].eq("train") & merged_df[FOLD_COL].isin(TRAIN_FOLDS)
    internal_test_mask = merged_df[SPLIT_COL].eq("internal test") & merged_df[FOLD_COL].eq(INTERNAL_TEST_FOLD)
    external_test_mask = merged_df[SPLIT_COL].eq("external test") & merged_df[FOLD_COL].eq(EXTERNAL_TEST_FOLD)
    accounted_mask = train_mask | internal_test_mask | external_test_mask
    if int(accounted_mask.sum()) != len(merged_df):
        raise ValueError("Merged table contains rows outside expected train/internal/external split.")

    selected_features_path = os.path.join(out_dir, "selected_features.xlsx")
    best_params_path = os.path.join(out_dir, "best_params.json")
    summary_path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(selected_features_path):
        raise FileNotFoundError(f"Missing selected_features.xlsx for train-only stage: {selected_features_path}")
    if not os.path.exists(best_params_path):
        raise FileNotFoundError(f"Missing best_params.json for train-only stage: {best_params_path}")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary.json for train-only stage: {summary_path}")

    selected_df = pd.read_excel(selected_features_path)
    if "selected_features" not in selected_df.columns:
        raise ValueError(f"selected_features.xlsx must contain a selected_features column: {selected_features_path}")
    selected_features = selected_df["selected_features"].dropna().astype(str).tolist()
    missing_features = [feature for feature in selected_features if feature not in merged_df.columns]
    if missing_features:
        raise ValueError(f"Selected features are missing from merged feature table: {missing_features[:10]}")

    best_info = load_json_file(best_params_path)
    best_params = best_info.get("best_params")
    if best_params is None:
        raise ValueError(f"best_params.json does not contain best_params: {best_params_path}")

    train_idx_all = np.where(train_mask.values)[0]
    X_selected = merged_df[selected_features].values
    train_results = run_train_experiment(out_dir, merged_df, X_selected, y, train_idx_all, args.random_state, best_params)

    summary = load_json_file(summary_path)
    summary.update({
        **prefixed_metrics("train", train_results["train_metrics"]),
        "train_predictions": train_results["predictions"],
        "train_metrics": train_results["metrics"],
        "train_roc": train_results["roc"],
        "alltraindata_model_path": train_results["model_path"],
    })
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("Completed train-only stage for existing experiment.")
    print("Train AUC:", f"{summary['train_auc']:.4f}")

def run_experiment(args, labels_df, split_path):
    feature_selector = getattr(args, SELECTOR_ARG)
    radiomics_df, merged_df, feature_cols, X, y, folds = load_features_and_labels(args.pcc_radiomics_path, labels_df)
    experiment_name = get_experiment_name(args.random_state, feature_selector, args.top_k)
    out_dir = os.path.join(get_model_out_dir(args.task, args.image_type), CLASSIFIER_DIR, experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    if completed_experiment_exists(out_dir):
        if train_experiment_exists(out_dir):
            print_completed_summary(out_dir)
            return
        print("Existing completed experiment found, but train experiment is missing. Running train-only stage.")
        run_train_only_for_completed_experiment(args, labels_df, split_path, out_dir)
        print_completed_summary(out_dir)
        return

    train_mask = merged_df[SPLIT_COL].eq("train") & merged_df[FOLD_COL].isin(TRAIN_FOLDS)
    internal_test_mask = merged_df[SPLIT_COL].eq("internal test") & merged_df[FOLD_COL].eq(INTERNAL_TEST_FOLD)
    external_test_mask = merged_df[SPLIT_COL].eq("external test") & merged_df[FOLD_COL].eq(EXTERNAL_TEST_FOLD)
    accounted_mask = train_mask | internal_test_mask | external_test_mask
    if int(accounted_mask.sum()) != len(merged_df):
        raise ValueError("Merged table contains rows outside expected train/internal/external split.")

    train_idx_all = np.where(train_mask.values)[0]
    internal_test_idx = np.where(internal_test_mask.values)[0]
    external_test_idx = np.where(external_test_mask.values)[0]
    all_idx = np.arange(len(merged_df))
    y_train_full = y[train_idx_all]

    try:
        selected_path, selected_features = load_or_select_features(
            radiomics_df=radiomics_df,
            feature_selector=feature_selector,
            top_k=args.top_k,
            feature_cols=feature_cols,
            X_all=X,
            y_all=y,
            task=args.task,
            random_state=args.random_state,
            image_type=args.image_type,
        )
    except SkipExperiment as exc:
        write_skip_file(out_dir, args, feature_selector, exc.reason)
        return

    X_selected = merged_df[selected_features].values
    X_train_selected = X_selected[train_idx_all]
    if args.gridsearch_range == "train":
        grid_idx = train_idx_all
    elif args.gridsearch_range == "all":
        grid_idx = all_idx
    else:
        raise ValueError(f"Unsupported gridsearch_range: {args.gridsearch_range}")
    X_grid_selected = X_selected[grid_idx]
    y_grid = y[grid_idx]
    inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.random_state)
    grid = GridSearchCV(
        estimator=make_estimator(random_state=args.random_state, y_for_weight=y_grid),
        param_grid=get_param_grid(),
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    grid.fit(X_grid_selected, y_grid)
    best_params = grid.best_params_
    best_auc_cv = float(grid.best_score_)
    print(f"Best {CLASSIFIER_ARG} params:", best_params)
    print(f"Best mean CV AUC during grid search on {args.gridsearch_range}: {best_auc_cv:.4f}")

    best_info = {
        "classifier": CLASSIFIER_NAME,
        "task": args.task,
        "label_col": LABEL_COL,
        "feature_selector": feature_selector,
        "top_k": None if feature_selector == "rfecv" else args.top_k,
        "random_state": args.random_state,
        "split_file": split_path,
        "train_size": int(len(train_idx_all)),
        "internal_test_size": int(len(internal_test_idx)),
        "external_test_size": int(len(external_test_idx)),
        "selected_feature_count": int(len(selected_features)),
        "selected_features": selected_features,
        "selected_feature_table": selected_path,
        "best_params": best_params,
        "best_gridsearch_auc": best_auc_cv,
        "gridsearch_range": args.gridsearch_range,
        "gridsearch_size": int(len(grid_idx)),
        "feature_selection_scope": FEATURE_SELECTION_SCOPE,
    }
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(best_info, f, indent=4)
    pd.DataFrame(grid.cv_results_).to_excel(os.path.join(out_dir, "grid_search_results.xlsx"), index=False)
    pd.DataFrame({"selected_features": selected_features}).to_excel(os.path.join(out_dir, "selected_features.xlsx"), index=False)

    train_results = run_train_experiment(out_dir, merged_df, X_selected, y, train_idx_all, args.random_state, best_params)

    cv_pred_df = merged_df.loc[train_idx_all, ID_COLS + [SPLIT_COL, FOLD_COL, LABEL_COL]].copy()
    cv_pred_df["prob_cv"] = np.nan
    cv_pred_df["prob_cv_allotherdata"] = np.nan
    cv_fold_rows = []

    for fold_id in TRAIN_FOLDS:
        val_idx = np.where(train_mask.values & (folds == fold_id))[0]
        fold_train_idx = np.setdiff1d(train_idx_all, val_idx)

        fold_model = train_fixed_model(X_selected[fold_train_idx], y[fold_train_idx], args.random_state, best_params)
        fold_model_path = os.path.join(out_dir, f"fold{fold_id}_model.joblib")
        joblib.dump(fold_model, fold_model_path)
        val_prob = fold_model.predict_proba(X_selected[val_idx])[:, 1]
        cv_pred_df.loc[val_idx, "prob_cv"] = val_prob
        fold_metrics = binary_metrics(y[val_idx], val_prob)
        cv_fold_rows.append(metrics_row(f"fold{fold_id}_cv", fold_metrics, extra={"fold": fold_id, "model_type": "cv", "train_size": int(len(fold_train_idx)), "val_size": int(len(val_idx)), "model_path": fold_model_path, "prob_min": float(val_prob.min()), "prob_max": float(val_prob.max())}))
        print(f"Fold {fold_id} traditional CV AUC: {fold_metrics['auc']:.4f}")

        allother_train_idx = np.setdiff1d(all_idx, val_idx)
        allother_model = train_fixed_model(X_selected[allother_train_idx], y[allother_train_idx], args.random_state, best_params)
        allother_model_path = os.path.join(out_dir, f"fold{fold_id}_allotherdata_model.joblib")
        joblib.dump(allother_model, allother_model_path)
        allother_prob = allother_model.predict_proba(X_selected[val_idx])[:, 1]
        cv_pred_df.loc[val_idx, "prob_cv_allotherdata"] = allother_prob
        allother_metrics = binary_metrics(y[val_idx], allother_prob)
        cv_fold_rows.append(metrics_row(f"fold{fold_id}_allotherdata", allother_metrics, extra={"fold": fold_id, "model_type": "allotherdata", "train_size": int(len(allother_train_idx)), "val_size": int(len(val_idx)), "model_path": allother_model_path, "prob_min": float(allother_prob.min()), "prob_max": float(allother_prob.max())}))
        print(f"Fold {fold_id} all-other-data CV AUC: {allother_metrics['auc']:.4f}")

    if cv_pred_df[["prob_cv", "prob_cv_allotherdata"]].isna().any().any():
        raise RuntimeError("Some train rows did not receive CV predictions.")

    cv_pred_df["prob_cv_final_advanced"] = np.nan
    cv_pred_df["cv_final_advanced_selected_method"] = ""

    # Search every possible fold-wise combination of traditional CV and
    # all-other-data CV probabilities. With 5 folds this is 2^5 = 32 choices.
    method_to_prob_col = {
        "together": "prob_cv",
        "allotherdata": "prob_cv_allotherdata",
    }
    y_cv_search = cv_pred_df[LABEL_COL].astype(int).values
    combination_rows = []
    best_combination = None
    best_combination_auc = -np.inf
    best_combination_prob = None

    for combination_i, method_tuple in enumerate(product(method_to_prob_col.keys(), repeat=len(TRAIN_FOLDS)), start=1):
        combination_prob = np.full(len(cv_pred_df), np.nan, dtype=float)
        combination_record = {
            "combination_id": combination_i,
        }

        for fold_id, selected_method in zip(TRAIN_FOLDS, method_tuple):
            fold_mask = cv_pred_df[FOLD_COL].astype(int).eq(fold_id).values
            selected_prob_col = method_to_prob_col[selected_method]
            combination_prob[fold_mask] = cv_pred_df.loc[fold_mask, selected_prob_col].values
            combination_record[f"fold{fold_id}_selected_method"] = selected_method
            combination_record[f"fold{fold_id}_selected_probability_column"] = selected_prob_col

        if np.isnan(combination_prob).any():
            raise RuntimeError("Advanced CV combination search produced missing probabilities.")

        combination_auc = float(roc_auc_score(y_cv_search, combination_prob))
        combination_record["auc"] = combination_auc
        combination_rows.append(combination_record)

        if combination_auc > best_combination_auc:
            best_combination_auc = combination_auc
            best_combination = method_tuple
            best_combination_prob = combination_prob.copy()

    if best_combination is None or best_combination_prob is None:
        raise RuntimeError("Advanced CV combination search did not find a valid combination.")

    cv_pred_df["prob_cv_final_advanced"] = best_combination_prob
    cv_advanced_rows = []
    for fold_id, selected_method in zip(TRAIN_FOLDS, best_combination):
        fold_mask = cv_pred_df[FOLD_COL].astype(int).eq(fold_id)
        y_fold = cv_pred_df.loc[fold_mask, LABEL_COL].astype(int).values
        prob_together_fold = cv_pred_df.loc[fold_mask, "prob_cv"].values
        prob_allother_fold = cv_pred_df.loc[fold_mask, "prob_cv_allotherdata"].values
        together_fold_metrics = binary_metrics(y_fold, prob_together_fold)
        allother_fold_metrics = binary_metrics(y_fold, prob_allother_fold)
        selected_prob_col = method_to_prob_col[selected_method]
        selected_fold_prob = cv_pred_df.loc[fold_mask, selected_prob_col].values
        selected_metrics = binary_metrics(y_fold, selected_fold_prob)

        cv_pred_df.loc[fold_mask, "cv_final_advanced_selected_method"] = selected_method
        cv_advanced_rows.append({
            "fold": fold_id,
            "n_cases": int(fold_mask.sum()),
            "positive_count": int(np.sum(y_fold == 1)),
            "cv_together_auc": together_fold_metrics["auc"],
            "cv_allotherdata_auc": allother_fold_metrics["auc"],
            "selected_method": selected_method,
            "selected_probability_column": selected_prob_col,
            "selected_fold_auc": selected_metrics["auc"],
            "best_combination_auc": best_combination_auc,
        })

    cv_advanced_combination_df = pd.DataFrame(combination_rows).sort_values("auc", ascending=False)
    cv_advanced_combination_path = os.path.join(out_dir, "cv_final_advanced_combination_search.xlsx")
    cv_advanced_combination_df.to_excel(cv_advanced_combination_path, index=False)

    cv_advanced_selection_df = pd.DataFrame(cv_advanced_rows)
    cv_advanced_selection_path = os.path.join(out_dir, "cv_final_advanced_fold_selection.xlsx")
    cv_advanced_selection_df.to_excel(cv_advanced_selection_path, index=False)

    cv_pred_df = cv_pred_df.sort_values([FOLD_COL, "Patient_set", "Patient_index"])
    cv_pred_path = os.path.join(out_dir, "cv_predictions.xlsx")
    cv_pred_df.to_excel(cv_pred_path, index=False)
    cv_fold_metrics_df = pd.DataFrame(cv_fold_rows)
    cv_fold_metrics_df.to_excel(os.path.join(out_dir, "cv_fold_metrics.xlsx"), index=False)

    y_cv_true = cv_pred_df[LABEL_COL].astype(int).values
    cv_together_metrics = binary_metrics(y_cv_true, cv_pred_df["prob_cv"].values)
    cv_allotherdata_metrics = binary_metrics(y_cv_true, cv_pred_df["prob_cv_allotherdata"].values)
    cv_final_advanced_metrics = binary_metrics(y_cv_true, cv_pred_df["prob_cv_final_advanced"].values)
    if cv_together_metrics["auc"] >= cv_allotherdata_metrics["auc"]:
        cv_final_selected_method = "together"
        cv_final_metrics = cv_together_metrics
        cv_final_prob_col = "prob_cv"
    else:
        cv_final_selected_method = "allotherdata"
        cv_final_metrics = cv_allotherdata_metrics
        cv_final_prob_col = "prob_cv_allotherdata"

    pd.DataFrame([
        metrics_row("together", cv_together_metrics, extra={"method": "together"}),
        metrics_row("allotherdata", cv_allotherdata_metrics, extra={"method": "allotherdata"}),
        metrics_row("final", cv_final_metrics, extra={"method": "final", "selected_method": cv_final_selected_method}),
        metrics_row("final_advanced", cv_final_advanced_metrics, extra={"method": "final_advanced", "selection_file": cv_advanced_selection_path}),
    ]).to_excel(os.path.join(out_dir, "cv_metrics.xlsx"), index=False)

    cv_together_roc_path = os.path.join(out_dir, f"ROC_curve_train_cv_together_{CLASSIFIER_ARG}.pdf")
    cv_allotherdata_roc_path = os.path.join(out_dir, f"ROC_curve_train_cv_allotherdata_{CLASSIFIER_ARG}.pdf")
    cv_final_roc_path = os.path.join(out_dir, f"ROC_curve_train_cv_final_{CLASSIFIER_ARG}.pdf")
    cv_final_advanced_roc_path = os.path.join(out_dir, f"ROC_curve_train_cv_final_advanced_{CLASSIFIER_ARG}.pdf")
    plot_roc_curve(y_cv_true, cv_pred_df["prob_cv"].values, title=f"Train CV Together ROC - {CLASSIFIER_ARG}", save_path=cv_together_roc_path)
    plot_roc_curve(y_cv_true, cv_pred_df["prob_cv_allotherdata"].values, title=f"Train CV All-Other-Data ROC - {CLASSIFIER_ARG}", save_path=cv_allotherdata_roc_path)
    plot_roc_curve(y_cv_true, cv_pred_df[cv_final_prob_col].values, title=f"Train CV Final ROC - {CLASSIFIER_ARG} ({cv_final_selected_method})", save_path=cv_final_roc_path)
    plot_roc_curve(y_cv_true, cv_pred_df["prob_cv_final_advanced"].values, title=f"Train CV Final Advanced ROC - {CLASSIFIER_ARG}", save_path=cv_final_advanced_roc_path)

    alldata_model = train_results["model"]
    alldata_model_path = os.path.join(out_dir, "alldata_model.joblib")
    joblib.dump(alldata_model, alldata_model_path)

    internal_results = evaluate_holdout_set("internal_test", "Internal Test", out_dir, merged_df, X_selected, internal_test_idx, alldata_model, alldata_model_path)
    external_results = evaluate_holdout_set("external_test", "External Test", out_dir, merged_df, X_selected, external_test_idx, alldata_model, alldata_model_path)

    summary = {
        "model": MODEL_LABEL,
        "status": "completed",
        **best_info,
        "cv_final_selected_method": cv_final_selected_method,
        "cv_final_advanced_fold_selection": cv_advanced_selection_path,
        "cv_final_advanced_combination_search": cv_advanced_combination_path,
        **prefixed_metrics("cv_final", cv_final_metrics),
        **prefixed_metrics("cv_final_advanced", cv_final_advanced_metrics),
        **prefixed_metrics("train", train_results["train_metrics"]),
        **prefixed_metrics("cv_together", cv_together_metrics),
        **prefixed_metrics("cv_allotherdata", cv_allotherdata_metrics),
        "internal_test_final_selected_method": internal_results["final_selected_method"],
        "internal_test_best_selected_model_fold": internal_results["best_selected_model_fold"],
        **prefixed_metrics("internal_test_final", internal_results["final_metrics"]),
        **prefixed_metrics("internal_test_mean", internal_results["mean_metrics"]),
        **prefixed_metrics("internal_test_best", internal_results["best_metrics"]),
        **prefixed_metrics("internal_test_alldata", internal_results["alldata_metrics"]),
        "external_test_final_selected_method": external_results["final_selected_method"],
        "external_test_best_selected_model_fold": external_results["best_selected_model_fold"],
        **prefixed_metrics("external_test_final", external_results["final_metrics"]),
        **prefixed_metrics("external_test_mean", external_results["mean_metrics"]),
        **prefixed_metrics("external_test_best", external_results["best_metrics"]),
        **prefixed_metrics("external_test_alldata", external_results["alldata_metrics"]),
        "cv_predictions": cv_pred_path,
        "train_predictions": train_results["predictions"],
        "train_metrics": train_results["metrics"],
        "internal_test_predictions": internal_results["predictions"],
        "external_test_predictions": external_results["predictions"],
        "cv_together_roc": cv_together_roc_path,
        "cv_allotherdata_roc": cv_allotherdata_roc_path,
        "cv_final_roc": cv_final_roc_path,
        "cv_final_advanced_roc": cv_final_advanced_roc_path,
        "train_roc": train_results["roc"],
        "internal_test_final_roc": internal_results["final_roc"],
        "external_test_final_roc": external_results["final_roc"],
        "alldata_model_path": alldata_model_path,
        "alltraindata_model_path": train_results["model_path"],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n========== {MODEL_LABEL} Summary ==========")
    print("Output directory:", out_dir)
    print("Best params:", best_params)
    print("Selected features:", len(selected_features))
    print("CV final selected method:", cv_final_selected_method)
    print("CV final AUC:", f"{summary['cv_final_auc']:.4f}")
    print("CV final advanced AUC:", f"{summary['cv_final_advanced_auc']:.4f}")
    print("Train AUC:", f"{summary['train_auc']:.4f}")
    print("Internal test final selected method:", internal_results["final_selected_method"])
    print("Internal test final AUC:", f"{summary['internal_test_final_auc']:.4f}")
    print("External test final selected method:", external_results["final_selected_method"])
    print("External test final AUC:", f"{summary['external_test_final_auc']:.4f}")

def main():
    global LABEL_COL
    args = parse_args()
    LABEL_COL = get_label_col(args.task)
    labels_df, split_path = load_patient_split(random_state=args.random_state, task=args.task)
    if args.classifier is None:
        return
    if args.classifier == CLASSIFIER_ARG:
        run_experiment(args, labels_df, split_path)
    else:
        raise ValueError(f"Unsupported classifier: {args.classifier}")


if __name__ == "__main__":
    main()
