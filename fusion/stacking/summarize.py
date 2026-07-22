import argparse
import json
import os

import pandas as pd


TASK = "Prognosis"
STACKING_ROOT_TEMPLATE = "/host/d/projects/Habitats/models/{task}/fusion/stacking"
SUMMARY_FILENAME = "stacking_model_summary.xlsx"


METRIC_GROUPS = [
    "train_metrics",
    "cv_final_metrics",
    "cv_final_advanced_metrics",
    "internal_test_final_metrics",
    "external_test_final_metrics",
]


def flatten_metrics(summary, metric_group):
    metrics = summary.get(metric_group, {})
    prefix = metric_group.replace("_metrics", "")
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def read_all_summaries(stacking_root):
    rows = []
    if not os.path.isdir(stacking_root):
        raise FileNotFoundError(stacking_root)

    for classifier in sorted(os.listdir(stacking_root)):
        classifier_dir = os.path.join(stacking_root, classifier)
        if not os.path.isdir(classifier_dir):
            continue

        for experiment in sorted(os.listdir(classifier_dir)):
            experiment_dir = os.path.join(classifier_dir, experiment)
            if not os.path.isdir(experiment_dir):
                continue

            summary_path = os.path.join(experiment_dir, "summary.json")
            if not os.path.isfile(summary_path):
                rows.append({
                    "classifier": classifier,
                    "experiment": experiment,
                    "status": "missing_summary",
                    "output_dir": experiment_dir,
                })
                continue

            with open(summary_path, "r") as f:
                summary = json.load(f)

            row = {
                "classifier": summary.get("classifier", classifier),
                "experiment": summary.get("experiment", experiment),
                "random_state": summary.get("random_state"),
                "gridsearch_range": summary.get("gridsearch_range"),
                "feature_selection": summary.get("feature_selection"),
                "feature_columns": ", ".join(summary.get("feature_columns", [])),
                "best_params": json.dumps(summary.get("best_params", {}), sort_keys=True),
                "best_grid_search_auc": summary.get("best_grid_search_auc"),
                "cv_final_selected_method": summary.get("cv_final_selected_method"),
                "internal_test_final_selected_method": summary.get("internal_test_final_selected_method"),
                "external_test_final_selected_method": summary.get("external_test_final_selected_method"),
                "n_total": summary.get("n_total"),
                "n_cv": summary.get("n_cv"),
                "n_internal_test": summary.get("n_internal_test"),
                "n_external_test": summary.get("n_external_test"),
                "status": "completed",
                "output_dir": summary.get("output_dir", experiment_dir),
            }

            for metric_group in METRIC_GROUPS:
                row.update(flatten_metrics(summary, metric_group))

            rows.append(row)

    return pd.DataFrame(rows)


def make_compact_table(full_df):
    compact_cols = [
        "classifier",
        "experiment",
        "random_state",
        "gridsearch_range",
        "best_params",
        "train_auc",
        "train_auc_ci_low",
        "train_auc_ci_high",
        "train_accuracy",
        "train_sensitivity",
        "train_specificity",
        "cv_final_auc",
        "cv_final_auc_ci_low",
        "cv_final_auc_ci_high",
        "cv_final_accuracy",
        "cv_final_sensitivity",
        "cv_final_specificity",
        "cv_final_advanced_auc",
        "cv_final_advanced_auc_ci_low",
        "cv_final_advanced_auc_ci_high",
        "cv_final_advanced_accuracy",
        "cv_final_advanced_sensitivity",
        "cv_final_advanced_specificity",
        "internal_test_final_auc",
        "internal_test_final_auc_ci_low",
        "internal_test_final_auc_ci_high",
        "internal_test_final_accuracy",
        "internal_test_final_sensitivity",
        "internal_test_final_specificity",
        "external_test_final_auc",
        "external_test_final_auc_ci_low",
        "external_test_final_auc_ci_high",
        "external_test_final_accuracy",
        "external_test_final_sensitivity",
        "external_test_final_specificity",
        "cv_final_selected_method",
        "internal_test_final_selected_method",
        "external_test_final_selected_method",
        "status",
        "output_dir",
    ]
    existing_cols = [col for col in compact_cols if col in full_df.columns]
    return full_df[existing_cols].copy()


def main():
    parser = argparse.ArgumentParser(description="Summarize stacking experiments.")
    parser.add_argument("--task", default=TASK)
    args = parser.parse_args()

    stacking_root = STACKING_ROOT_TEMPLATE.format(task=args.task)
    save_path = os.path.join(stacking_root, SUMMARY_FILENAME)

    full_df = read_all_summaries(stacking_root)
    compact_df = make_compact_table(full_df)

    if "status" in full_df.columns:
        status_df = (
            full_df
            .groupby(["classifier", "status"], dropna=False)
            .size()
            .reset_index(name="n")
        )
    else:
        status_df = pd.DataFrame()

    with pd.ExcelWriter(save_path) as writer:
        full_df.to_excel(writer, sheet_name="All_full", index=False)
        compact_df.to_excel(writer, sheet_name="All_compact", index=False)
        status_df.to_excel(writer, sheet_name="Status", index=False)

    print("Saved stacking summary:", save_path)
    print("Rows:", full_df.shape[0])


if __name__ == "__main__":
    main()
