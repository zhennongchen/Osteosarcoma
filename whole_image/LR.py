import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_RANDOM_STATE = 0
N_SPLITS = 5
LABEL_COL = "Prognosis_label"
NON_FEATURE_COLS = ["Patient_set", "Patient_index", "Image_filepath", "Mask_filepath"]
LASSO_MAX_FEATURES = 50
PATIENT_LIST_PATH = (
    "/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12.xlsx"
)
SPLIT_OUT_PATH_TEMPLATE = (
    "/host/e/D/Data/Habitats/Jishuitan/Patient_lists/"
    "image_label_info_set12_5fold_prognosis_random{random_state}.xlsx"
)
PCC_RADIOMICS_PATH = (
    "/host/d/projects/Habitats/radiomics/whole_image/radiomics_measurements_PCC.xlsx"
)
WHOLE_IMAGE_RADIOMICS_OUT_DIR = "/host/d/projects/Habitats/radiomics/whole_image"
WHOLE_IMAGE_MODEL_OUT_DIR = "/host/d/projects/Habitats/models/whole_image"



def parse_top_k(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "none":
        return None
    try:
        top_k = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("top_k must be 15, 20, 25, or None") from exc
    if top_k not in {15, 20, 25}:
        raise argparse.ArgumentTypeError("top_k must be 15, 20, 25, or None")
    return top_k


def top_k_label(top_k):
    if top_k is None or (isinstance(top_k, str) and top_k.lower() == "none"):
        return "none"
    return f"top{top_k}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run whole-image radiomics A-line Logistic Regression experiments."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for stratified 5-fold patient split.",
    )
    parser.add_argument(
        "--classifier",
        choices=["LR"],
        default="LR",
        help="Classifier branch to run.",
    )
    parser.add_argument(
        "--lr_feature_selector",
        choices=["lasso"],
        default="lasso",
        help="LR-specific supervised feature-selection method.",
    )
    parser.add_argument(
        "--top_k",
        type=parse_top_k,
        default=20,
        help="Number of selected features for LASSO ranking, or None for all non-zero features.",
    )
    return parser.parse_args()


def make_patient_split(random_state):
    split_out_path = SPLIT_OUT_PATH_TEMPLATE.format(random_state=random_state)

    if os.path.exists(split_out_path):
        df = pd.read_excel(split_out_path)
        if "fold" in df.columns:
            print("Loaded existing patient split:", split_out_path)
        else:
            print("Existing split file has no fold column; regenerating:", split_out_path)
            df = None
    else:
        df = None

    if df is None:
        df = pd.read_excel(PATIENT_LIST_PATH)

        if LABEL_COL not in df.columns:
            raise ValueError(f"Missing label column: {LABEL_COL}")

        y = df[LABEL_COL].astype(int).values
        skf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=random_state,
        )

        df["fold"] = -1
        for fold_id, (_, val_idx) in enumerate(skf.split(df, y)):
            df.loc[val_idx, "fold"] = fold_id

        os.makedirs(os.path.dirname(split_out_path), exist_ok=True)
        df.to_excel(split_out_path, index=False)
        print("Saved patient split:", split_out_path)

    return df, split_out_path


def load_features_and_labels(radiomics_path, labels_df):
    radiomics_df = pd.read_excel(radiomics_path)

    missing_label_cols = [
        c for c in ["Patient_set", "Patient_index", "fold", LABEL_COL] if c not in labels_df.columns
    ]
    if missing_label_cols:
        raise ValueError(f"Missing columns in label table: {missing_label_cols}")

    missing_radiomics_cols = [
        c for c in ["Patient_set", "Patient_index"] if c not in radiomics_df.columns
    ]
    if missing_radiomics_cols:
        raise ValueError(f"Missing columns in radiomics table: {missing_radiomics_cols}")

    label_cols = ["Patient_set", "Patient_index", "fold", LABEL_COL]
    merged_df = radiomics_df.merge(
        labels_df[label_cols],
        on=["Patient_set", "Patient_index"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged_df) != len(radiomics_df) or len(merged_df) != len(labels_df):
        raise ValueError(
            "Radiomics and label tables are not a complete one-to-one match: "
            f"radiomics={len(radiomics_df)}, labels={len(labels_df)}, merged={len(merged_df)}"
        )

    feature_cols = [c for c in radiomics_df.columns if c not in NON_FEATURE_COLS]
    X = merged_df[feature_cols].values
    y = merged_df[LABEL_COL].astype(int).values
    folds = merged_df["fold"].astype(int).values

    print(
        f"Feature matrix shape: {X.shape}",
        f"Label vector shape: {y.shape}",
        f"Fold vector shape: {folds.shape}",
    )

    return radiomics_df, merged_df, feature_cols, X, y, folds


def get_feature_cols_from_selected_table(selected_df):
    return [c for c in selected_df.columns if c not in NON_FEATURE_COLS]


def get_lr_selected_feature_path(random_state, feature_selector, top_k):
    suffix = f"random{random_state}_{feature_selector}_{top_k_label(top_k)}"
    return os.path.join(
        WHOLE_IMAGE_RADIOMICS_OUT_DIR,
        f"radiomics_measurements_LR_{suffix}_selected.xlsx",
    )


def select_lasso_features(top_k, feature_cols, X, y, random_state):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=random_state,
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lasso",
                LogisticRegressionCV(
                    Cs=30,
                    cv=cv,
                    penalty="l1",
                    solver="liblinear",
                    scoring="roc_auc",
                    max_iter=5000,
                    refit=True,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X, y)
    lasso_cv = model.named_steps["lasso"]

    coef = lasso_cv.coef_.ravel()
    abs_coef = np.abs(coef)
    nonzero_idx = np.where(coef != 0)[0]
    nonzero_sorted_idx = nonzero_idx[np.argsort(abs_coef[nonzero_idx])[::-1]]

    if top_k is None:
    
        selected_idx = nonzero_sorted_idx
        if len(selected_idx) > LASSO_MAX_FEATURES:
            raise ValueError(
                f"LASSO selected {len(selected_idx)} non-zero features, "
                f"which exceeds the hard limit of {LASSO_MAX_FEATURES}."
            )
    elif len(nonzero_sorted_idx) >= top_k:
        selected_idx = nonzero_sorted_idx[:top_k]
    else:
        remaining_idx = [
            i for i in np.argsort(abs_coef)[::-1] if i not in set(nonzero_sorted_idx)
        ]
        selected_idx = np.array(
            list(nonzero_sorted_idx) + remaining_idx[: top_k - len(nonzero_sorted_idx)]
        )

    selected_features = [feature_cols[i] for i in selected_idx]
    selected_coef = coef[selected_idx]

    # print("Best LASSO C:", float(lasso_cv.C_[0]))
    # print("Best LASSO mean CV AUC:", float(lasso_cv.scores_[1].mean(axis=0).max()))
    # print("Total features:", len(feature_cols))
    # print("Selected by LASSO non-zero:", len(nonzero_idx))
    # print("Selected features:", len(selected_features))
    # for feature, coefficient in zip(selected_features, selected_coef):
    #     print(f"  {feature}: coef={coefficient:.6f}")

    metadata = {
        "lasso_best_C": float(lasso_cv.C_[0]),
        "lasso_best_mean_cv_auc": float(lasso_cv.scores_[1].mean(axis=0).max()),
        "lasso_nonzero_feature_count": int(len(nonzero_idx)),
    }
    return selected_features, metadata


def save_selected_lr_features(radiomics_df, selected_features, random_state, feature_selector, top_k):
    os.makedirs(WHOLE_IMAGE_RADIOMICS_OUT_DIR, exist_ok=True)
    selected_out_path = get_lr_selected_feature_path(
        random_state=random_state,
        feature_selector=feature_selector,
        top_k=top_k,
    )
    df_selected = radiomics_df[NON_FEATURE_COLS + selected_features].copy()
    df_selected.to_excel(selected_out_path, index=False)
    print("Saved selected feature table:", selected_out_path)
    return selected_out_path


def load_or_select_lr_features(
    radiomics_df,
    feature_selector,
    top_k,
    feature_cols,
    X,
    y,
    random_state,
):
    selected_path = get_lr_selected_feature_path(
        random_state=random_state,
        feature_selector=feature_selector,
        top_k=top_k,
    )

    if os.path.exists(selected_path):
        selected_df = pd.read_excel(selected_path)
        selected_features = get_feature_cols_from_selected_table(selected_df)
        if selected_features:
            print("Loaded existing selected feature table:", selected_path)
            print(f"Selected features by {feature_selector}: {len(selected_features)}")
            return selected_path, selected_features, {}
        print("Existing selected feature table has no feature columns; regenerating:", selected_path)

    if feature_selector != "lasso":
        raise ValueError(f"Unsupported LR feature selector: {feature_selector}")

    selected_features, metadata = select_lasso_features(
        top_k=top_k,
        feature_cols=feature_cols,
        X=X,
        y=y,
        random_state=random_state,
    )
    selected_path = save_selected_lr_features(
        radiomics_df=radiomics_df,
        selected_features=selected_features,
        random_state=random_state,
        feature_selector=feature_selector,
        top_k=top_k,
    )
    return selected_path, selected_features, metadata


def make_lr_pipeline(random_state, C=1.0, tol=1e-4):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    C=C,
                    class_weight="balanced",
                    tol=tol,
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def best_threshold_metrics(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_mask = np.isfinite(thresholds)
    if valid_mask.any():
        valid_idx = np.where(valid_mask)[0]
        best_local = int(np.argmax(tpr[valid_mask] - fpr[valid_mask]))
        best_idx = int(valid_idx[best_local])
    else:
        best_idx = int(np.argmax(tpr - fpr))

    best_threshold = float(thresholds[best_idx])
    y_pred = (y_prob >= best_threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = float((tp + tn) / len(y_true)) if len(y_true) > 0 else float("nan")
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    f1 = (
        float((2 * precision * sensitivity) / (precision + sensitivity))
        if (precision + sensitivity) > 0
        else float("nan")
    )

    return {
        "best_threshold_max_se_plus_sp": best_threshold,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def run_lr_experiment(args, labels_df):
    radiomics_df, merged_df, feature_cols, X, y, folds = load_features_and_labels(
        PCC_RADIOMICS_PATH,
        labels_df,
    )
    selected_path, selected_features, selection_metadata = load_or_select_lr_features(
        radiomics_df=radiomics_df,
        feature_selector=args.lr_feature_selector,
        top_k=args.top_k,
        feature_cols=feature_cols,
        X=X,
        y=y,
        random_state=args.random_state,
    )

    selected_df = pd.read_excel(selected_path)
    selected_feature_cols = get_feature_cols_from_selected_table(selected_df)
    X_selected = merged_df[selected_feature_cols].values

    experiment_name = f"random{args.random_state}_{args.lr_feature_selector}_{top_k_label(args.top_k)}"
    out_dir = os.path.join(WHOLE_IMAGE_MODEL_OUT_DIR, "LR", experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    lr_pipe = make_lr_pipeline(random_state=args.random_state)
    param_grid_lr = {
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "clf__tol": [1e-4, 1e-3],
    }
    inner_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=args.random_state,
    )

    grid = GridSearchCV(
        estimator=lr_pipe,
        param_grid=param_grid_lr,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid.fit(X_selected, y)

    best_params = grid.best_params_
    best_auc_cv = grid.best_score_
    print("Best LR params:", best_params)
    print(f"Best mean CV AUC during grid search: {best_auc_cv:.4f}")

    best_info = {
        "classifier": "LR",
        "feature_selector": args.lr_feature_selector,
        "top_k": args.top_k,
        "random_state": args.random_state,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "selected_feature_table": selected_path,
        **selection_metadata,
        "best_params": best_params,
        "best_gridsearch_auc": float(best_auc_cv),
    }
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(best_info, f, indent=4)

    grid_results_df = pd.DataFrame(grid.cv_results_)
    grid_results_df.to_excel(os.path.join(out_dir, "grid_search_results.xlsx"), index=False)

    df_preds = merged_df[["Patient_set", "Patient_index", "fold", LABEL_COL]].copy()
    df_preds["pred_prob"] = np.nan
    fold_rows = []
    fold_aucs = []

    for fold_id in range(N_SPLITS):
        train_idx = np.where(folds != fold_id)[0]
        val_idx = np.where(folds == fold_id)[0]

        X_train, y_train = X_selected[train_idx], y[train_idx]
        X_val, y_val = X_selected[val_idx], y[val_idx]

        lr_fixed = make_lr_pipeline(
            random_state=args.random_state,
            C=best_params["clf__C"],
            tol=best_params["clf__tol"],
        )
        lr_fixed.fit(X_train, y_train)
        prob = lr_fixed.predict_proba(X_val)[:, 1]

        fold_auc = roc_auc_score(y_val, prob)
        fold_aucs.append(fold_auc)
        df_preds.loc[val_idx, "pred_prob"] = prob

        print(f"Fold {fold_id} AUC: {fold_auc:.4f}")
        fold_rows.append(
            {
                "fold": fold_id,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "auc": fold_auc,
                "prob_min": float(prob.min()),
                "prob_max": float(prob.max()),
            }
        )

    pred_path = os.path.join(out_dir, "predictions.xlsx")
    df_preds.to_excel(pred_path, index=False)
    pd.DataFrame(fold_rows).to_excel(os.path.join(out_dir, "fold_metrics.xlsx"), index=False)

    y_true = df_preds[LABEL_COL].values
    y_prob = df_preds["pred_prob"].values
    overall_auc = roc_auc_score(y_true, y_prob)
    threshold_metrics = best_threshold_metrics(y_true, y_prob)

    summary = {
        "model": "Logistic Regression",
        **best_info,
        "fold_aucs": [float(x) for x in fold_aucs],
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs)),
        "overall_oof_auc": float(overall_auc),
        **threshold_metrics,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\n========== Logistic Regression Summary ==========")
    print("Output directory:", out_dir)
    print("Best params:", best_params)
    print(f"Fold AUCs: {[round(x, 4) for x in fold_aucs]}")
    print(f"Mean fold AUC: {np.mean(fold_aucs):.4f}")
    print(f"Std fold AUC: {np.std(fold_aucs):.4f}")
    print(f"Overall out-of-fold AUC: {overall_auc:.4f}")


def main():
    args = parse_args()
    labels_df, _ = make_patient_split(random_state=args.random_state)

    if args.classifier == "LR":
        run_lr_experiment(args, labels_df)
    else:
        raise ValueError(f"Unsupported classifier: {args.classifier}")


if __name__ == "__main__":
    main()
