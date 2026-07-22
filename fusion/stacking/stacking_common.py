import argparse
import itertools
import json
import os
from copy import deepcopy

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TASK = "Prognosis"
LABEL_COL = "Prognosis_label"

FUSION_ROOT = "/host/d/projects/Habitats/models/Prognosis/fusion"
STACKING_ROOT = os.path.join(FUSION_ROOT, "stacking")
FUSION_PROBABILITY_PATH = os.path.join(
    FUSION_ROOT,
    "fusion_final_selection_probabilities.xlsx",
)

FEATURE_COLS = [
    "prob_clinical",
    "prob_whole_image",
    "prob_habitats_sum",
    "prob_dl_3d_ml_all",
]

ID_COLS = [
    "Patient_set",
    "Patient_index",
]

META_COLS = [
    "Patient_set",
    "Patient_index",
    LABEL_COL,
    "dataset",
    "split",
    "fold",
]


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=2000, random_state=0):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    aucs = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    if len(aucs) == 0:
        return np.nan, np.nan

    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def best_threshold_by_youden(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    return float(thresholds[int(np.argmax(youden))])


def evaluate_probability(y_true, y_prob, random_state=0):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    auc = safe_auc(y_true, y_prob)
    auc_ci_low, auc_ci_high = bootstrap_auc_ci(
        y_true,
        y_prob,
        random_state=random_state,
    )
    threshold = best_threshold_by_youden(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
    else:
        tn = fp = fn = tp = np.nan
        accuracy = sensitivity = specificity = np.nan

    return {
        "auc": auc,
        "auc_ci_low": auc_ci_low,
        "auc_ci_high": auc_ci_high,
        "threshold": threshold,
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n": int(len(y_true)),
        "positive_fraction": float(np.mean(y_true)) if len(y_true) else np.nan,
    }


def prefix_metrics(metrics, prefix):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def valid_json_file(path):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r") as f:
            json.load(f)
        return True
    except Exception:
        return False


def plot_roc(y_true, y_prob, title, save_path, label):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return

    auc = safe_auc(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"{label} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def probability_from_model(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]

    if hasattr(model, "decision_function"):
        decision = model.decision_function(x)
        decision = np.asarray(decision).astype(float)
        return 1.0 / (1.0 + np.exp(-decision))

    raise RuntimeError("Model does not support predict_proba or decision_function.")


def build_estimator_and_grid(classifier_name, random_state):
    classifier_name = classifier_name.upper()

    if classifier_name == "SVM":
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=random_state)),
        ])
        param_grid = {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto"],
            "clf__tol": [1e-3, 1e-4],
        }
    elif classifier_name == "LR":
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ])
        param_grid = {
            "clf__C": [0.01, 0.1, 1, 10, 100],
        }
    elif classifier_name == "RF":
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ])
        param_grid = {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [2, 3, 4, None],
            "clf__min_samples_leaf": [1, 2, 4],
        }
    elif classifier_name == "KNN":
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ])
        param_grid = {
            "clf__n_neighbors": [3, 5, 7, 9, 11],
            "clf__weights": ["uniform", "distance"],
        }
    elif classifier_name == "XGBOOST":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBoost stacking. Install xgboost or skip XGBoost."
            ) from exc

        estimator = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=1,
                    tree_method="hist",
                ),
            ),
        ])
        param_grid = {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [1, 2, 3],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        }
    else:
        raise ValueError(f"Unknown classifier_name: {classifier_name}")

    return estimator, param_grid


def load_data():
    if not os.path.isfile(FUSION_PROBABILITY_PATH):
        raise FileNotFoundError(
            "Fusion probability table not found. Run fusion/list_probs.ipynb first: "
            f"{FUSION_PROBABILITY_PATH}"
        )

    df = pd.read_excel(FUSION_PROBABILITY_PATH)
    missing_cols = [
        col for col in META_COLS + FEATURE_COLS
        if col not in df.columns
    ]
    if len(missing_cols) > 0:
        raise KeyError(f"Missing columns in fusion table: {missing_cols}")

    if df[FEATURE_COLS].isna().any().any():
        raise RuntimeError(
            "Missing stacking input probabilities:\n"
            f"{df[FEATURE_COLS].isna().sum()}"
        )

    df["dataset"] = df["dataset"].astype(str)
    df["fold"] = df["fold"].astype(int)
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    return df


def fit_grid_search(classifier_name, x_grid, y_grid, random_state):
    estimator, param_grid = build_estimator_and_grid(classifier_name, random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    grid.fit(x_grid, y_grid)
    return grid


def fit_best_model(base_estimator, x_train, y_train):
    model = clone(base_estimator)
    model.fit(x_train, y_train)
    return model


def choose_by_auc(metric_a, metric_b, name_a, name_b):
    auc_a = metric_a.get("auc", np.nan)
    auc_b = metric_b.get("auc", np.nan)
    if np.nan_to_num(auc_b, nan=-np.inf) > np.nan_to_num(auc_a, nan=-np.inf):
        return name_b
    return name_a


def build_cv_advanced(cv_pred_df, fold_values, random_state):
    y_true = cv_pred_df[LABEL_COL].astype(int).to_numpy()
    method_to_col = {
        "cv": "prob_cv",
        "allotherdata": "prob_cv_allotherdata",
    }

    best_auc = -np.inf
    best_combo = None
    best_prob = None

    for combo in itertools.product(["cv", "allotherdata"], repeat=len(fold_values)):
        prob = np.zeros(cv_pred_df.shape[0], dtype=float)
        for fold_value, method in zip(fold_values, combo):
            mask = cv_pred_df["fold"].astype(int).to_numpy() == int(fold_value)
            prob[mask] = cv_pred_df.loc[mask, method_to_col[method]].astype(float).to_numpy()
        auc = safe_auc(y_true, prob)
        if np.isnan(auc):
            continue
        if auc > best_auc:
            best_auc = auc
            best_combo = dict(zip(fold_values, combo))
            best_prob = prob.copy()

    if best_combo is None:
        raise RuntimeError("No valid cv_final_advanced combination was found.")

    selected_methods = []
    for _, row in cv_pred_df.iterrows():
        selected_methods.append(best_combo[int(row["fold"])])

    cv_pred_df["prob_cv_final_advanced"] = best_prob
    cv_pred_df["cv_final_advanced_selected_method"] = selected_methods

    advanced_metrics = evaluate_probability(y_true, best_prob, random_state=random_state)
    advanced_combo_df = pd.DataFrame({
        "fold": list(best_combo.keys()),
        "selected_method": list(best_combo.values()),
        "selected_probability_column": [
            method_to_col[method] for method in best_combo.values()
        ],
    })

    return cv_pred_df, advanced_metrics, advanced_combo_df


def evaluate_holdout(
    dataset_name,
    dataset_df,
    x_dataset,
    y_dataset,
    fold_models,
    alldata_model,
    out_dir,
    classifier_name,
    random_state,
):
    pred_df = dataset_df[META_COLS].copy()

    fold_metric_rows = []
    fold_prob_cols = []
    best_fold = None
    best_fold_auc = -np.inf

    for fold_value, model in fold_models.items():
        prob = probability_from_model(model, x_dataset)
        col = f"prob_fold{fold_value}_model"
        pred_df[col] = prob
        fold_prob_cols.append(col)

        metrics = evaluate_probability(y_dataset, prob, random_state=random_state)
        row = {
            "dataset": dataset_name,
            "mode": f"fold{fold_value}_model",
            "fold": int(fold_value),
            "probability_column": col,
        }
        row.update(metrics)
        fold_metric_rows.append(row)

        if np.nan_to_num(metrics["auc"], nan=-np.inf) > best_fold_auc:
            best_fold_auc = metrics["auc"]
            best_fold = fold_value

    pred_df["prob_mean"] = pred_df[fold_prob_cols].mean(axis=1)
    pred_df["best_model_fold"] = int(best_fold)
    pred_df["prob_best"] = pred_df[f"prob_fold{best_fold}_model"]
    pred_df["prob_alldata"] = probability_from_model(alldata_model, x_dataset)

    metric_rows = deepcopy(fold_metric_rows)
    mode_to_col = {
        "mean": "prob_mean",
        "best": "prob_best",
        "alldata": "prob_alldata",
    }
    mode_metrics = {}
    for mode, col in mode_to_col.items():
        metrics = evaluate_probability(y_dataset, pred_df[col].to_numpy(), random_state=random_state)
        mode_metrics[mode] = metrics
        row = {
            "dataset": dataset_name,
            "mode": mode,
            "probability_column": col,
            "is_final": False,
        }
        row.update(metrics)
        metric_rows.append(row)

    final_mode = "mean"
    for candidate in ["best", "alldata"]:
        final_mode = choose_by_auc(
            mode_metrics[final_mode],
            mode_metrics[candidate],
            final_mode,
            candidate,
        )

    pred_df["prob_final"] = pred_df[mode_to_col[final_mode]]
    pred_df["final_selected_method"] = final_mode

    final_metrics = evaluate_probability(y_dataset, pred_df["prob_final"].to_numpy(), random_state=random_state)
    row = {
        "dataset": dataset_name,
        "mode": "final",
        "final_selected_method": final_mode,
        "probability_column": "prob_final",
        "source_probability_column": mode_to_col[final_mode],
        "is_final": True,
    }
    row.update(final_metrics)
    metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)

    pred_path = os.path.join(out_dir, f"{dataset_name}_predictions.xlsx")
    metrics_path = os.path.join(out_dir, f"{dataset_name}_metrics.xlsx")
    roc_path = os.path.join(out_dir, f"ROC_curve_{dataset_name}_final_{classifier_name}.pdf")

    pred_df.to_excel(pred_path, index=False)
    metrics_df.to_excel(metrics_path, index=False)
    plot_roc(
        y_dataset,
        pred_df["prob_final"].to_numpy(),
        title=f"Stacking {classifier_name}: {dataset_name} final",
        save_path=roc_path,
        label=f"{dataset_name} final",
    )

    return pred_df, metrics_df, final_metrics, final_mode


def run_stacking_experiment(classifier_name, random_state, gridsearch_range):
    classifier_name = classifier_name.upper()
    gridsearch_range = gridsearch_range.lower()
    if gridsearch_range not in ["all", "train"]:
        raise ValueError("--gridsearch_range must be 'all' or 'train'.")

    experiment_name = f"random{random_state}"
    out_dir = os.path.join(STACKING_ROOT, classifier_name, experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "summary.json")
    required_done = [
        summary_path,
        os.path.join(out_dir, "cv_predictions.xlsx"),
        os.path.join(out_dir, "train_predictions.xlsx"),
        os.path.join(out_dir, "internal_test_predictions.xlsx"),
        os.path.join(out_dir, "external_test_predictions.xlsx"),
    ]
    if all(os.path.isfile(path) for path in required_done) and valid_json_file(summary_path):
        print("Existing completed stacking experiment found. Skipping:", out_dir)
        return

    print("\n============================================================")
    print("Stacking classifier:", classifier_name)
    print("Random state:", random_state)
    print("Grid-search range:", gridsearch_range)
    print("Output:", out_dir)

    df = load_data()
    x_all = df[FEATURE_COLS].astype(float).to_numpy()
    y_all = df[LABEL_COL].astype(int).to_numpy()

    cv_mask = df["dataset"].astype(str).to_numpy() == "cv"
    internal_mask = df["dataset"].astype(str).to_numpy() == "internal_test"
    external_mask = df["dataset"].astype(str).to_numpy() == "external_test"

    train_df = df.loc[cv_mask].copy()
    internal_df = df.loc[internal_mask].copy()
    external_df = df.loc[external_mask].copy()

    x_train = train_df[FEATURE_COLS].astype(float).to_numpy()
    y_train = train_df[LABEL_COL].astype(int).to_numpy()
    x_internal = internal_df[FEATURE_COLS].astype(float).to_numpy()
    y_internal = internal_df[LABEL_COL].astype(int).to_numpy()
    x_external = external_df[FEATURE_COLS].astype(float).to_numpy()
    y_external = external_df[LABEL_COL].astype(int).to_numpy()

    if gridsearch_range == "all":
        x_grid = x_all
        y_grid = y_all
    else:
        x_grid = x_train
        y_grid = y_train

    grid = fit_grid_search(classifier_name, x_grid, y_grid, random_state=random_state)
    best_estimator = grid.best_estimator_

    print("Best params:", grid.best_params_)
    print("Best grid-search AUC:", grid.best_score_)

    grid_results_df = pd.DataFrame(grid.cv_results_)
    grid_results_df.to_excel(os.path.join(out_dir, "grid_search_results.xlsx"), index=False)
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(grid.best_params_, f, indent=2)

    fold_values = sorted(train_df["fold"].astype(int).unique().tolist())

    cv_pred_df = train_df[META_COLS].copy()
    cv_pred_df["prob_cv"] = np.nan
    cv_pred_df["prob_cv_allotherdata"] = np.nan

    fold_models = {}
    allotherdata_models = {}

    for fold_value in fold_values:
        fold_mask_train = train_df["fold"].astype(int).to_numpy() != int(fold_value)
        fold_mask_val = train_df["fold"].astype(int).to_numpy() == int(fold_value)

        model = fit_best_model(
            best_estimator,
            x_train[fold_mask_train],
            y_train[fold_mask_train],
        )
        fold_models[fold_value] = model
        joblib.dump(model, os.path.join(out_dir, f"fold{fold_value}_model.joblib"))

        cv_pred_df.loc[fold_mask_val, "prob_cv"] = probability_from_model(
            model,
            x_train[fold_mask_val],
        )

        allother_mask = ~(
            (df["dataset"].astype(str).to_numpy() == "cv")
            & (df["fold"].astype(int).to_numpy() == int(fold_value))
        )
        allother_model = fit_best_model(
            best_estimator,
            x_all[allother_mask],
            y_all[allother_mask],
        )
        allotherdata_models[fold_value] = allother_model
        joblib.dump(
            allother_model,
            os.path.join(out_dir, f"fold{fold_value}_allotherdata_model.joblib"),
        )

        cv_pred_df.loc[fold_mask_val, "prob_cv_allotherdata"] = probability_from_model(
            allother_model,
            x_train[fold_mask_val],
        )

    y_cv = cv_pred_df[LABEL_COL].astype(int).to_numpy()
    cv_together_metrics = evaluate_probability(y_cv, cv_pred_df["prob_cv"].to_numpy(), random_state=random_state)
    cv_allotherdata_metrics = evaluate_probability(
        y_cv,
        cv_pred_df["prob_cv_allotherdata"].to_numpy(),
        random_state=random_state,
    )

    cv_final_method = choose_by_auc(
        cv_together_metrics,
        cv_allotherdata_metrics,
        "cv",
        "allotherdata",
    )
    cv_final_col = "prob_cv" if cv_final_method == "cv" else "prob_cv_allotherdata"
    cv_pred_df["prob_cv_final"] = cv_pred_df[cv_final_col].astype(float)
    cv_pred_df["cv_final_selected_method"] = cv_final_method

    cv_final_metrics = evaluate_probability(
        y_cv,
        cv_pred_df["prob_cv_final"].to_numpy(),
        random_state=random_state,
    )

    cv_pred_df, cv_final_advanced_metrics, cv_advanced_combo_df = build_cv_advanced(
        cv_pred_df,
        fold_values=fold_values,
        random_state=random_state,
    )

    cv_metric_rows = []
    for mode, prob_col, metrics in [
        ("cv_together", "prob_cv", cv_together_metrics),
        ("cv_allotherdata", "prob_cv_allotherdata", cv_allotherdata_metrics),
        ("cv_final", "prob_cv_final", cv_final_metrics),
        ("cv_final_advanced", "prob_cv_final_advanced", cv_final_advanced_metrics),
    ]:
        row = {
            "dataset": "cv",
            "mode": mode,
            "probability_column": prob_col,
        }
        if mode == "cv_final":
            row["selected_method"] = cv_final_method
        row.update(metrics)
        cv_metric_rows.append(row)

    cv_metrics_df = pd.DataFrame(cv_metric_rows)
    cv_pred_df.to_excel(os.path.join(out_dir, "cv_predictions.xlsx"), index=False)
    cv_metrics_df.to_excel(os.path.join(out_dir, "cv_metrics.xlsx"), index=False)
    cv_advanced_combo_df.to_excel(os.path.join(out_dir, "cv_final_advanced_combo.xlsx"), index=False)
    plot_roc(
        y_cv,
        cv_pred_df["prob_cv_final"].to_numpy(),
        title=f"Stacking {classifier_name}: CV final",
        save_path=os.path.join(out_dir, f"ROC_curve_cv_final_{classifier_name}.pdf"),
        label="CV final",
    )
    plot_roc(
        y_cv,
        cv_pred_df["prob_cv_final_advanced"].to_numpy(),
        title=f"Stacking {classifier_name}: CV final advanced",
        save_path=os.path.join(out_dir, f"ROC_curve_cv_final_advanced_{classifier_name}.pdf"),
        label="CV final advanced",
    )

    alldata_model = fit_best_model(best_estimator, x_train, y_train)
    joblib.dump(alldata_model, os.path.join(out_dir, "alldata_model.joblib"))
    joblib.dump(alldata_model, os.path.join(out_dir, "alltraindata_model.joblib"))

    train_pred_df = train_df[META_COLS].copy()
    train_pred_df["prob_train"] = probability_from_model(alldata_model, x_train)
    train_metrics = evaluate_probability(
        y_train,
        train_pred_df["prob_train"].to_numpy(),
        random_state=random_state,
    )
    train_pred_df.to_excel(os.path.join(out_dir, "train_predictions.xlsx"), index=False)
    pd.DataFrame([{**{"dataset": "train", "mode": "train"}, **train_metrics}]).to_excel(
        os.path.join(out_dir, "train_metrics.xlsx"),
        index=False,
    )
    plot_roc(
        y_train,
        train_pred_df["prob_train"].to_numpy(),
        title=f"Stacking {classifier_name}: train",
        save_path=os.path.join(out_dir, f"ROC_curve_train_{classifier_name}.pdf"),
        label="train",
    )

    internal_pred_df, internal_metrics_df, internal_final_metrics, internal_final_method = evaluate_holdout(
        "internal_test",
        internal_df,
        x_internal,
        y_internal,
        fold_models,
        alldata_model,
        out_dir,
        classifier_name,
        random_state,
    )
    external_pred_df, external_metrics_df, external_final_metrics, external_final_method = evaluate_holdout(
        "external_test",
        external_df,
        x_external,
        y_external,
        fold_models,
        alldata_model,
        out_dir,
        classifier_name,
        random_state,
    )

    summary = {
        "task": TASK,
        "classifier": classifier_name,
        "experiment": experiment_name,
        "random_state": random_state,
        "gridsearch_range": gridsearch_range,
        "feature_columns": FEATURE_COLS,
        "feature_selection": "none",
        "fusion_probability_path": FUSION_PROBABILITY_PATH,
        "output_dir": out_dir,
        "best_params": grid.best_params_,
        "best_grid_search_auc": float(grid.best_score_),
        "n_total": int(df.shape[0]),
        "n_cv": int(train_df.shape[0]),
        "n_internal_test": int(internal_df.shape[0]),
        "n_external_test": int(external_df.shape[0]),
        "cv_final_selected_method": cv_final_method,
        "internal_test_final_selected_method": internal_final_method,
        "external_test_final_selected_method": external_final_method,
        "train_metrics": train_metrics,
        "cv_together_metrics": cv_together_metrics,
        "cv_allotherdata_metrics": cv_allotherdata_metrics,
        "cv_final_metrics": cv_final_metrics,
        "cv_final_advanced_metrics": cv_final_advanced_metrics,
        "internal_test_final_metrics": internal_final_metrics,
        "external_test_final_metrics": external_final_metrics,
        "artifacts": {
            "best_params": os.path.join(out_dir, "best_params.json"),
            "grid_search_results": os.path.join(out_dir, "grid_search_results.xlsx"),
            "cv_predictions": os.path.join(out_dir, "cv_predictions.xlsx"),
            "cv_metrics": os.path.join(out_dir, "cv_metrics.xlsx"),
            "train_predictions": os.path.join(out_dir, "train_predictions.xlsx"),
            "train_metrics": os.path.join(out_dir, "train_metrics.xlsx"),
            "internal_test_predictions": os.path.join(out_dir, "internal_test_predictions.xlsx"),
            "internal_test_metrics": os.path.join(out_dir, "internal_test_metrics.xlsx"),
            "external_test_predictions": os.path.join(out_dir, "external_test_predictions.xlsx"),
            "external_test_metrics": os.path.join(out_dir, "external_test_metrics.xlsx"),
        },
    }

    with open(summary_path, "w") as f:
        json.dump(make_json_safe(summary), f, indent=2)

    print("\nFinished stacking experiment:", out_dir)
    print("Train AUC:", train_metrics["auc"])
    print("CV final AUC:", cv_final_metrics["auc"])
    print("CV final advanced AUC:", cv_final_advanced_metrics["auc"])
    print("Internal test final AUC:", internal_final_metrics["auc"])
    print("External test final AUC:", external_final_metrics["auc"])


def parse_args(classifier_name):
    parser = argparse.ArgumentParser(description=f"Run {classifier_name} stacking.")
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--gridsearch_range", default="all", choices=["all", "train"])
    args = parser.parse_args()
    if args.task != TASK:
        raise ValueError(f"Only TASK={TASK} is currently configured. Got {args.task}.")
    return args


def main_for_classifier(classifier_name):
    args = parse_args(classifier_name)
    run_stacking_experiment(
        classifier_name=classifier_name,
        random_state=args.random_state,
        gridsearch_range=args.gridsearch_range,
    )
