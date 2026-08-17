import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


TASK = "Prognosis"
LABEL_COL = "Prognosis_label"

FUSION_ROOT = "/host/d/projects/Habitats/models/Prognosis/fusion"
FUSION_PROBABILITY_PATH = os.path.join(
    FUSION_ROOT,
    "fusion_final_selection_probabilities.xlsx",
)

PROBABILITY_COLS = [
    "prob_clinical",
    "prob_whole_image",
    "prob_habitats_avg",
    "prob_dl_3d_ml_all",
]

SOFT_VOTE_PROB_COL = "prob_soft_vote"

PREDICTION_SAVE_PATH = os.path.join(
    FUSION_ROOT,
    "soft_vote_predictions.xlsx",
)
METRICS_SAVE_PATH = os.path.join(
    FUSION_ROOT,
    "soft_vote_metrics.xlsx",
)
MANIFEST_SAVE_PATH = os.path.join(
    FUSION_ROOT,
    "soft_vote_manifest.json",
)
ROC_SAVE_PATH_TEMPLATE = os.path.join(
    FUSION_ROOT,
    "ROC_curve_soft_vote_{dataset}.pdf",
)


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


def evaluate_probability(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    auc = safe_auc(y_true, y_prob)
    auc_ci_low, auc_ci_high = bootstrap_auc_ci(y_true, y_prob)
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


def plot_roc(y_true, y_prob, title, save_path):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        print("  ROC skipped because only one class is present:", save_path)
        return

    auc = safe_auc(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"Soft vote AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("  Saved ROC:", save_path)


def main():
    if not os.path.isfile(FUSION_PROBABILITY_PATH):
        raise FileNotFoundError(
            "Fusion probability table not found. Run list_probs.ipynb first: "
            f"{FUSION_PROBABILITY_PATH}"
        )

    os.makedirs(FUSION_ROOT, exist_ok=True)

    df = pd.read_excel(FUSION_PROBABILITY_PATH)

    missing_cols = [
        col for col in [LABEL_COL, "dataset"] + PROBABILITY_COLS
        if col not in df.columns
    ]
    if len(missing_cols) > 0:
        raise KeyError(f"Missing required columns: {missing_cols}")

    if df[PROBABILITY_COLS].isna().any().any():
        missing_counts = df[PROBABILITY_COLS].isna().sum()
        raise RuntimeError(
            "Missing probability values detected. Please inspect fusion table:\n"
            f"{missing_counts}"
        )

    df[SOFT_VOTE_PROB_COL] = df[PROBABILITY_COLS].mean(axis=1)

    metric_rows = []
    dataset_order = ["cv", "internal_test", "external_test"]
    available_datasets = df["dataset"].astype(str).unique().tolist()
    dataset_order += [
        dataset for dataset in available_datasets
        if dataset not in dataset_order
    ]

    for dataset in dataset_order:
        dataset_df = df[df["dataset"].astype(str) == dataset].copy()
        if dataset_df.shape[0] == 0:
            continue

        print("\n============================================================")
        print("Evaluating soft vote:", dataset)
        print("  n =", dataset_df.shape[0])

        y_true = dataset_df[LABEL_COL].astype(int).to_numpy()
        y_prob = dataset_df[SOFT_VOTE_PROB_COL].astype(float).to_numpy()

        metrics = evaluate_probability(y_true, y_prob)
        row = {
            "dataset": dataset,
            "method": "soft_vote",
            "probability_column": SOFT_VOTE_PROB_COL,
            "base_probability_columns": ", ".join(PROBABILITY_COLS),
        }
        row.update(metrics)
        metric_rows.append(row)

        print(
            "  AUC = {auc:.4f}, ACC = {accuracy:.4f}, "
            "SEN = {sensitivity:.4f}, SPE = {specificity:.4f}".format(**metrics)
        )

        plot_roc(
            y_true,
            y_prob,
            title=f"Fusion soft vote: {dataset}",
            save_path=ROC_SAVE_PATH_TEMPLATE.format(dataset=dataset),
        )

    metrics_df = pd.DataFrame(metric_rows)

    df.to_excel(PREDICTION_SAVE_PATH, index=False)
    metrics_df.to_excel(METRICS_SAVE_PATH, index=False)

    manifest = {
        "task": TASK,
        "label_col": LABEL_COL,
        "method": "soft_vote",
        "definition": "Arithmetic mean of the four final-selection probabilities.",
        "fusion_probability_path": FUSION_PROBABILITY_PATH,
        "base_probability_columns": PROBABILITY_COLS,
        "soft_vote_probability_column": SOFT_VOTE_PROB_COL,
        "outputs": {
            "predictions": PREDICTION_SAVE_PATH,
            "metrics": METRICS_SAVE_PATH,
            "manifest": MANIFEST_SAVE_PATH,
            "roc_template": ROC_SAVE_PATH_TEMPLATE,
        },
    }
    with open(MANIFEST_SAVE_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n============================================================")
    print("Saved soft-vote predictions:", PREDICTION_SAVE_PATH)
    print("Saved soft-vote metrics:", METRICS_SAVE_PATH)
    print("Saved soft-vote manifest:", MANIFEST_SAVE_PATH)


if __name__ == "__main__":
    main()
