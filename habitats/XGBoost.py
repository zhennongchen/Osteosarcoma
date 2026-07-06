import argparse
import json
import os

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
INTERNAL_TEST_FOLD = 5
LABEL_COL = TASK_TO_LABEL_COL[DEFAULT_TASK]
NON_FEATURE_COLS = ["Patient_set", "Patient_index", "Image_filepath", "Mask_filepath"]
ID_COLS = ["Patient_set", "Patient_index", "Image_filepath", "Mask_filepath"]
RFECV_MAX_FEATURES = 35
LASSO_MAX_FEATURES = 35
SPLIT_COL = "split"
FOLD_COL = "fold"
PATIENT_LIST_PATH = "/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12.xlsx"
SPLIT_OUT_PATH_TEMPLATE = (
    "/host/e/D/Data/Habitats/Jishuitan/Patient_lists/"
    "image_label_info_set12_5fold_{task_lower}_random{random_state}.xlsx"
)
DEFAULT_PCC_RADIOMICS_PATH = "/host/d/projects/Habitats/radiomics/habitats_individual/habitat_radiomics_measurements_avg_PCC.xlsx"
RADIOMICS_OUT_DIR = "/host/d/projects/Habitats/radiomics/habitats"
MODEL_ROOT = "/host/d/projects/Habitats/models"
DEFAULT_IMAGE_TYPE = "habitats_individual"
CLASSIFIER_ARG = "XGBoost"
CLASSIFIER_NAME = "XGBoost"
CLASSIFIER_DIR = "XGBoost"
MODEL_LABEL = "XGBoost"
SELECTOR_ARG = "xgb_feature_selector"
DEFAULT_SELECTOR = "rfe"
SELECTED_PREFIX = "habitat_radiomics_measurements"
METRIC_KEYS = ["auc", "auc_ci_low", "auc_ci_high", "accuracy", "sensitivity", "specificity"]
FEATURE_SELECTION_SCOPE = "all_330_train_plus_internal_test"


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
    parser.add_argument("--classifier", choices=[CLASSIFIER_ARG], default=CLASSIFIER_ARG)
    parser.add_argument("--xgb_feature_selector", choices=['rfe', 'sfs', 'rfecv', 'lasso'], default=DEFAULT_SELECTOR)
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
    train_mask = df[SPLIT_COL].eq("train") & df[FOLD_COL].isin(range(N_SPLITS))
    test_mask = df[SPLIT_COL].eq("internal test") & df[FOLD_COL].eq(INTERNAL_TEST_FOLD)
    if train_mask.sum() + test_mask.sum() != len(df):
        raise ValueError("Rows outside expected train/internal-test split found.")
    print("Loaded patient split:", split_path)
    print("Train cases:", int(train_mask.sum()), "Internal test cases:", int(test_mask.sum()))
    print("Train positive fraction:", f"{df.loc[train_mask, label_col].mean():.4f}", "Internal test positive fraction:", f"{df.loc[test_mask, label_col].mean():.4f}")
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
    elif feature_selector == "sfs":
        selector = SequentialFeatureSelector(estimator=estimator, n_features_to_select=top_k, direction="forward", scoring="roc_auc", cv=cv, n_jobs=1)
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
    if feature_selector in {"rfe", "sfs"}:
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


def plot_cv_mean_roc(cv_pred_df, title, save_path, figsize=(5, 5)):
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
    if feature_selector in {"rfe", "sfs"}:
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
    artifacts = ["summary.json", "best_params.json", "cv_predictions.xlsx", "test_predictions.xlsx", "cv_metrics.xlsx", "cv_fold_metrics.xlsx", "test_metrics.xlsx", "grid_search_results.xlsx", "selected_features.xlsx", "alldata_model.joblib"]
    artifacts.extend([f"fold{fold_id}_model.joblib" for fold_id in range(N_SPLITS)])
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
    print("CV selected mode:", summary.get("cv_selected_metric_mode", ""))
    print("CV better AUC:", f"{summary.get('cv_better_auc', float('nan')):.4f}")
    print("Test final selected method:", summary.get("test_final_selected_method", ""))
    print("Test final AUC:", f"{summary.get('test_final_auc', float('nan')):.4f}")


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


def run_experiment(args, labels_df, split_path):
    feature_selector = getattr(args, SELECTOR_ARG)
    radiomics_df, merged_df, feature_cols, X, y, folds = load_features_and_labels(args.pcc_radiomics_path, labels_df)
    experiment_name = get_experiment_name(args.random_state, feature_selector, args.top_k)
    out_dir = os.path.join(get_model_out_dir(args.task, args.image_type), CLASSIFIER_DIR, experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    if completed_experiment_exists(out_dir):
        print_completed_summary(out_dir)
        return

    train_mask = merged_df[SPLIT_COL].eq("train") & merged_df[FOLD_COL].isin(range(N_SPLITS))
    test_mask = merged_df[SPLIT_COL].eq("internal test") & merged_df[FOLD_COL].eq(INTERNAL_TEST_FOLD)
    train_idx_all = np.where(train_mask.values)[0]
    test_idx = np.where(test_mask.values)[0]
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
    inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.random_state)
    grid = GridSearchCV(estimator=make_estimator(random_state=args.random_state, y_for_weight=y_train_full), param_grid=get_param_grid(), scoring="roc_auc", cv=inner_cv, n_jobs=1, refit=True, verbose=1)
    grid.fit(X_train_selected, y_train_full)
    best_params = grid.best_params_
    best_auc_cv = float(grid.best_score_)
    print(f"Best {CLASSIFIER_ARG} params:", best_params)
    print(f"Best mean CV AUC during grid search on train: {best_auc_cv:.4f}")

    best_info = {
        "classifier": CLASSIFIER_NAME,
        "task": args.task,
        "label_col": LABEL_COL,
        "feature_selector": feature_selector,
        "top_k": None if feature_selector == "rfecv" else args.top_k,
        "random_state": args.random_state,
        "split_file": split_path,
        "train_size": int(len(train_idx_all)),
        "internal_test_size": int(len(test_idx)),
        "selected_feature_count": int(len(selected_features)),
        "selected_features": selected_features,
        "selected_feature_table": selected_path,
        "best_params": best_params,
        "best_gridsearch_auc": best_auc_cv,
        "feature_selection_scope": FEATURE_SELECTION_SCOPE,
        "image_type": args.image_type,
        "pcc_radiomics_path": args.pcc_radiomics_path,
    }
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(best_info, f, indent=4)
    pd.DataFrame(grid.cv_results_).to_excel(os.path.join(out_dir, "grid_search_results.xlsx"), index=False)
    pd.DataFrame({"selected_features": selected_features}).to_excel(os.path.join(out_dir, "selected_features.xlsx"), index=False)

    cv_pred_df = merged_df.loc[train_idx_all, ID_COLS + [SPLIT_COL, FOLD_COL, LABEL_COL]].copy()
    cv_pred_df["pred_prob"] = np.nan
    cv_fold_rows = []
    for fold_id in range(N_SPLITS):
        val_idx = np.where(train_mask.values & (folds == fold_id))[0]
        fold_train_idx = np.setdiff1d(train_idx_all, val_idx)
        fold_model = train_fixed_model(X_selected[fold_train_idx], y[fold_train_idx], args.random_state, best_params)
        fold_model_path = os.path.join(out_dir, f"fold{fold_id}_model.joblib")
        joblib.dump(fold_model, fold_model_path)
        val_prob = fold_model.predict_proba(X_selected[val_idx])[:, 1]
        cv_pred_df.loc[val_idx, "pred_prob"] = val_prob
        fold_metrics = binary_metrics(y[val_idx], val_prob)
        cv_fold_rows.append(metrics_row(f"fold{fold_id}", fold_metrics, extra={"fold": fold_id, "train_size": int(len(fold_train_idx)), "val_size": int(len(val_idx)), "model_path": fold_model_path, "prob_min": float(val_prob.min()), "prob_max": float(val_prob.max())}))
        print(f"Fold {fold_id} AUC: {fold_metrics['auc']:.4f}")

    if cv_pred_df["pred_prob"].isna().any():
        raise RuntimeError("Some train rows did not receive OOF predictions.")
    cv_pred_df = cv_pred_df.sort_values([FOLD_COL, "Patient_set", "Patient_index"])
    cv_pred_path = os.path.join(out_dir, "cv_predictions.xlsx")
    cv_pred_df.to_excel(cv_pred_path, index=False)
    cv_fold_metrics_df = pd.DataFrame(cv_fold_rows)
    cv_fold_metrics_df.to_excel(os.path.join(out_dir, "cv_fold_metrics.xlsx"), index=False)

    y_cv_true = cv_pred_df[LABEL_COL].astype(int).values
    y_cv_prob = cv_pred_df["pred_prob"].values
    cv_together_metrics = binary_metrics(y_cv_true, y_cv_prob)
    cv_mean_metrics = {key: float(cv_fold_metrics_df[key].mean()) for key in METRIC_KEYS}
    cv_mean_auc_ci_low, cv_mean_auc_ci_high = bootstrap_mean_ci(cv_fold_metrics_df["auc"].values)
    cv_mean_metrics["auc_ci_low"] = cv_mean_auc_ci_low
    cv_mean_metrics["auc_ci_high"] = cv_mean_auc_ci_high
    cv_mean_metrics["threshold"] = float(cv_fold_metrics_df["threshold"].mean())
    if cv_together_metrics["auc"] >= cv_mean_metrics["auc"]:
        cv_selected_metric_mode = "together"; cv_better_metrics = cv_together_metrics
    else:
        cv_selected_metric_mode = "mean"; cv_better_metrics = cv_mean_metrics
    pd.DataFrame([metrics_row("together", cv_together_metrics), metrics_row("mean", cv_mean_metrics), metrics_row("better", cv_better_metrics, extra={"selected_mode": cv_selected_metric_mode})]).to_excel(os.path.join(out_dir, "cv_metrics.xlsx"), index=False)

    cv_better_roc_path = os.path.join(out_dir, f"ROC_curve_train_cv_better_{CLASSIFIER_ARG}.pdf")
    if cv_selected_metric_mode == "together":
        plot_roc_curve(y_cv_true, y_cv_prob, title=f"Train CV Better ROC - {CLASSIFIER_ARG} (together)", save_path=cv_better_roc_path)
    else:
        plot_cv_mean_roc(cv_pred_df, title=f"Train CV Better ROC - {CLASSIFIER_ARG} (mean folds)", save_path=cv_better_roc_path)

    test_pred_df = merged_df.loc[test_idx, ID_COLS + [SPLIT_COL, FOLD_COL, LABEL_COL]].copy()
    y_test = test_pred_df[LABEL_COL].astype(int).values
    test_method_rows = []
    fold_test_metrics = []
    for fold_id in range(N_SPLITS):
        fold_model = joblib.load(os.path.join(out_dir, f"fold{fold_id}_model.joblib"))
        prob = fold_model.predict_proba(X_selected[test_idx])[:, 1]
        prob_col = f"prob_fold{fold_id}_model"
        test_pred_df[prob_col] = prob
        method_metrics = binary_metrics(y_test, prob)
        fold_test_metrics.append((fold_id, method_metrics, prob_col))
        test_method_rows.append(metrics_row(f"fold{fold_id}_model", method_metrics, extra={"method": "fold_model", "fold_model": fold_id}))
    fold_prob_cols = [f"prob_fold{fold_id}_model" for fold_id in range(N_SPLITS)]
    test_pred_df["prob_mean"] = test_pred_df[fold_prob_cols].mean(axis=1)
    test_mean_metrics = binary_metrics(y_test, test_pred_df["prob_mean"].values)
    test_method_rows.append(metrics_row("mean", test_mean_metrics, extra={"method": "mean"}))
    best_fold_id, test_best_metrics, best_prob_col = max(fold_test_metrics, key=lambda item: item[1]["auc"])
    test_pred_df["best_model_fold"] = best_fold_id
    test_pred_df["prob_best"] = test_pred_df[best_prob_col]
    test_method_rows.append(metrics_row("best", test_best_metrics, extra={"method": "best", "best_model_fold": best_fold_id}))
    alldata_model = train_fixed_model(X_train_selected, y_train_full, args.random_state, best_params)
    alldata_model_path = os.path.join(out_dir, "alldata_model.joblib")
    joblib.dump(alldata_model, alldata_model_path)
    test_pred_df["prob_alldata"] = alldata_model.predict_proba(X_selected[test_idx])[:, 1]
    test_alldata_metrics = binary_metrics(y_test, test_pred_df["prob_alldata"].values)
    test_method_rows.append(metrics_row("alldata", test_alldata_metrics, extra={"method": "alldata", "model_path": alldata_model_path}))
    test_final_selected_method, test_final_metrics = max([("mean", test_mean_metrics), ("best", test_best_metrics), ("alldata", test_alldata_metrics)], key=lambda item: item[1]["auc"])
    if test_final_selected_method == "mean":
        test_pred_df["prob_final"] = test_pred_df["prob_mean"]
    elif test_final_selected_method == "best":
        test_pred_df["prob_final"] = test_pred_df["prob_best"]
    else:
        test_pred_df["prob_final"] = test_pred_df["prob_alldata"]
    test_pred_df["final_selected_method"] = test_final_selected_method
    test_method_rows.append(metrics_row("final", test_final_metrics, extra={"method": "final", "selected_method": test_final_selected_method}))
    test_pred_path = os.path.join(out_dir, "test_predictions.xlsx")
    test_pred_df = test_pred_df.sort_values(["Patient_set", "Patient_index"])
    test_pred_df.to_excel(test_pred_path, index=False)
    pd.DataFrame(test_method_rows).to_excel(os.path.join(out_dir, "test_metrics.xlsx"), index=False)
    test_final_roc_path = os.path.join(out_dir, f"ROC_curve_internal_test_final_{CLASSIFIER_ARG}.pdf")
    plot_roc_curve(test_pred_df[LABEL_COL].astype(int).values, test_pred_df["prob_final"].values, title=f"Internal Test Final ROC - {CLASSIFIER_ARG}", save_path=test_final_roc_path)

    summary = {
        "model": MODEL_LABEL,
        "status": "completed",
        **best_info,
        "cv_selected_metric_mode": cv_selected_metric_mode,
        **prefixed_metrics("cv_better", cv_better_metrics),
        **prefixed_metrics("cv_together", cv_together_metrics),
        **prefixed_metrics("cv_mean", cv_mean_metrics),
        "test_final_selected_method": test_final_selected_method,
        "test_best_selected_model_fold": int(best_fold_id),
        **prefixed_metrics("test_final", test_final_metrics),
        **prefixed_metrics("test_mean", test_mean_metrics),
        **prefixed_metrics("test_best", test_best_metrics),
        **prefixed_metrics("test_alldata", test_alldata_metrics),
        "cv_predictions": cv_pred_path,
        "test_predictions": test_pred_path,
        "cv_better_roc": cv_better_roc_path,
        "test_final_roc": test_final_roc_path,
        "alldata_model_path": alldata_model_path,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\n========== {MODEL_LABEL} Summary ==========")
    print("Output directory:", out_dir)
    print("Best params:", best_params)
    print("Selected features:", len(selected_features))
    print("CV selected mode:", cv_selected_metric_mode)
    print("CV better AUC:", f"{summary['cv_better_auc']:.4f}")
    print("Test final selected method:", test_final_selected_method)
    print("Test final AUC:", f"{summary['test_final_auc']:.4f}")


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
