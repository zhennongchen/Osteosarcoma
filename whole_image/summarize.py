#!/usr/bin/env python3
"""Summarize whole-image ML experiment outputs into full and compact Excel sheets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TASK = "Prognosis"
MODEL_ROOT = Path("/host/d/projects/Habitats/models")
IMAGE_TYPE = "whole_image"
DEFAULT_OUTPUT_NAME = "whole_image_model_summary.xlsx"
PREFERRED_CLASSIFIER_ORDER = ["SVM", "LR", "XGBoost", "RandomForest"]
EXPERIMENT_RE = re.compile(
    r"^random(?P<random_state>\d+)_(?P<feature_selector>[^_]+)(?:_(?P<top_label>none|top\d+)|_top(?P<top_k>.+))?$",
    re.IGNORECASE,
)
METRIC_PREFIXES = [
    "train",
    "cv_final",
    "cv_final_advanced",
    "cv_together",
    "cv_allotherdata",
    "internal_test_final",
    "internal_test_mean",
    "internal_test_best",
    "internal_test_alldata",
    "external_test_final",
    "external_test_mean",
    "external_test_best",
    "external_test_alldata",
]
METRICS = ["auc", "auc_ci_low", "auc_ci_high", "accuracy", "sensitivity", "specificity"]
BASE_COLUMNS = [
    "experiment",
    "status",
    "classifier",
    "model",
    "task",
    "random_state",
    "feature_selector",
    "top_k",
    "feature_selection_scope",
    "selected_feature_count",
    "best_gridsearch_auc",
    "gridsearch_range",
    "gridsearch_size",
    "best_params",
    "train_size",
    "internal_test_size",
    "external_test_size",
    "cv_final_selected_method",
    "internal_test_final_selected_method",
    "internal_test_best_selected_model_fold",
    "external_test_final_selected_method",
    "external_test_best_selected_model_fold",
]
FULL_COLUMNS = [
    *BASE_COLUMNS,
    *[f"{prefix}_{metric}" for prefix in METRIC_PREFIXES for metric in METRICS],
    "skip_reason",
    "selected_feature_table",
    "cv_predictions",
    "train_predictions",
    "train_metrics",
    "cv_final_advanced_fold_selection",
    "cv_final_advanced_combination_search",
    "internal_test_predictions",
    "external_test_predictions",
    "cv_together_roc",
    "cv_allotherdata_roc",
    "cv_final_roc",
    "cv_final_advanced_roc",
    "train_roc",
    "internal_test_final_roc",
    "external_test_final_roc",
    "alldata_model_path",
    "alltraindata_model_path",
    "split_file",
]
COMPACT_COLUMNS = [
    "experiment",
    "status",
    "classifier",
    "task",
    "random_state",
    "feature_selector",
    "top_k",
    "feature_selection_scope",
    "selected_feature_count",
    "train_auc",
    "train_auc_ci_low",
    "train_auc_ci_high",
    "train_accuracy",
    "train_sensitivity",
    "train_specificity",
    "cv_final_selected_method",
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
    "internal_test_final_selected_method",
    "internal_test_final_auc",
    "internal_test_final_auc_ci_low",
    "internal_test_final_auc_ci_high",
    "internal_test_final_accuracy",
    "internal_test_final_sensitivity",
    "internal_test_final_specificity",
    "external_test_final_selected_method",
    "external_test_final_auc",
    "external_test_final_auc_ci_low",
    "external_test_final_auc_ci_high",
    "external_test_final_accuracy",
    "external_test_final_sensitivity",
    "external_test_final_specificity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Excel summaries for whole-image ML experiments.")
    parser.add_argument("--task", choices=["Prognosis", "Pathologic"], default=DEFAULT_TASK)
    parser.add_argument(
        "--models_root",
        type=Path,
        default=None,
        help=f"Root folder containing one subfolder per classifier. Default: /host/d/projects/Habitats/models/{{task}}/{IMAGE_TYPE}.",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=None,
        help=f"Output Excel path. Default: models_root/{DEFAULT_OUTPUT_NAME}",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_json(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_top_k(value: Any, experiment_name: str = "") -> Any:
    if value is None:
        return "None" if experiment_name.lower().endswith("_none") else ""
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return "None"
    return value


def parse_experiment_name(experiment_name: str) -> dict[str, Any]:
    match = EXPERIMENT_RE.match(experiment_name)
    if not match:
        return {}
    parsed: dict[str, Any] = {
        "random_state": int(match.group("random_state")),
        "feature_selector": match.group("feature_selector"),
    }
    top_k = match.group("top_k")
    top_label = match.group("top_label")
    if top_label:
        if top_label.lower() == "none":
            parsed["top_k"] = None
        elif top_label.lower().startswith("top"):
            parsed["top_k"] = int(top_label[3:])
    elif top_k is not None:
        if top_k.lower() in {"none", "null"}:
            parsed["top_k"] = None
        else:
            try:
                parsed["top_k"] = int(top_k)
            except ValueError:
                parsed["top_k"] = top_k
    return parsed


def read_optional_best_params(experiment_dir: Path, data: dict[str, Any]) -> Any:
    if "best_params" in data:
        return data.get("best_params")
    best_params_path = experiment_dir / "best_params.json"
    if best_params_path.exists():
        best_info = load_json(best_params_path)
        return best_info.get("best_params", best_info)
    return None


def blank_full_row() -> dict[str, str]:
    return {column: "" for column in FULL_COLUMNS}


def build_completed_row(classifier_dir: str, experiment_dir: Path, data: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_experiment_name(experiment_dir.name)
    row: dict[str, Any] = blank_full_row()
    row.update(
        {
            "experiment": experiment_dir.name,
            "status": data.get("status", "completed"),
            "classifier": data.get("classifier", classifier_dir),
            "model": data.get("model", ""),
            "task": data.get("task", ""),
            "random_state": data.get("random_state", parsed.get("random_state", "")),
            "feature_selector": data.get("feature_selector", parsed.get("feature_selector", "")),
            "top_k": normalize_top_k(data.get("top_k", parsed.get("top_k", "")), experiment_dir.name),
            "feature_selection_scope": data.get("feature_selection_scope", ""),
            "selected_feature_count": data.get("selected_feature_count", ""),
            "best_gridsearch_auc": data.get("best_gridsearch_auc", ""),
            "gridsearch_range": data.get("gridsearch_range", ""),
            "gridsearch_size": data.get("gridsearch_size", ""),
            "best_params": compact_json(read_optional_best_params(experiment_dir, data)),
            "train_size": data.get("train_size", ""),
            "internal_test_size": data.get("internal_test_size", ""),
            "external_test_size": data.get("external_test_size", ""),
            "cv_final_selected_method": data.get("cv_final_selected_method", ""),
            "internal_test_final_selected_method": data.get("internal_test_final_selected_method", ""),
            "internal_test_best_selected_model_fold": data.get("internal_test_best_selected_model_fold", ""),
            "external_test_final_selected_method": data.get("external_test_final_selected_method", ""),
            "external_test_best_selected_model_fold": data.get("external_test_best_selected_model_fold", ""),
            "skip_reason": "",
            "selected_feature_table": data.get("selected_feature_table", ""),
            "cv_predictions": data.get("cv_predictions", ""),
            "train_predictions": data.get("train_predictions", ""),
            "train_metrics": data.get("train_metrics", ""),
            "cv_final_advanced_fold_selection": data.get("cv_final_advanced_fold_selection", ""),
            "cv_final_advanced_combination_search": data.get("cv_final_advanced_combination_search", ""),
            "internal_test_predictions": data.get("internal_test_predictions", data.get("test_predictions", "")),
            "external_test_predictions": data.get("external_test_predictions", ""),
            "cv_together_roc": data.get("cv_together_roc", ""),
            "cv_allotherdata_roc": data.get("cv_allotherdata_roc", ""),
            "cv_final_roc": data.get("cv_final_roc", data.get("cv_better_roc", "")),
            "cv_final_advanced_roc": data.get("cv_final_advanced_roc", ""),
            "train_roc": data.get("train_roc", ""),
            "internal_test_final_roc": data.get("internal_test_final_roc", data.get("test_final_roc", "")),
            "external_test_final_roc": data.get("external_test_final_roc", ""),
            "alldata_model_path": data.get("alldata_model_path", ""),
            "alltraindata_model_path": data.get("alltraindata_model_path", ""),
            "split_file": data.get("split_file", ""),
        }
    )
    for prefix in METRIC_PREFIXES:
        for metric in METRICS:
            key = f"{prefix}_{metric}"
            row[key] = data.get(key, "")
    return row


def build_skipped_row(classifier_dir: str, experiment_dir: Path, data: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_experiment_name(experiment_dir.name)
    row = blank_full_row()
    row.update(
        {
            "experiment": experiment_dir.name,
            "status": data.get("status", "skipped"),
            "classifier": data.get("classifier", classifier_dir),
            "model": data.get("model", ""),
            "task": data.get("task", ""),
            "random_state": data.get("random_state", parsed.get("random_state", "")),
            "feature_selector": data.get("feature_selector", parsed.get("feature_selector", "")),
            "top_k": normalize_top_k(data.get("top_k", parsed.get("top_k", "")), experiment_dir.name),
            "skip_reason": data.get("reason", ""),
        }
    )
    return row


def build_incomplete_row(classifier_dir: str, experiment_dir: Path) -> dict[str, Any]:
    parsed = parse_experiment_name(experiment_dir.name)
    row = blank_full_row()
    row.update(
        {
            "experiment": experiment_dir.name,
            "status": "incomplete",
            "classifier": classifier_dir,
            "random_state": parsed.get("random_state", ""),
            "feature_selector": parsed.get("feature_selector", ""),
            "top_k": normalize_top_k(parsed.get("top_k", ""), experiment_dir.name),
            "skip_reason": "No summary.json or SKIPPED.json found.",
        }
    )
    return row


def discover_classifier_dirs(models_root: Path) -> list[Path]:
    classifier_dirs = [p for p in models_root.iterdir() if p.is_dir()]
    preferred_rank = {name: i for i, name in enumerate(PREFERRED_CLASSIFIER_ORDER)}
    return sorted(classifier_dirs, key=lambda p: (preferred_rank.get(p.name, len(preferred_rank)), p.name.lower()))


def top_k_sort_value(value: Any) -> int:
    if value in ("", None):
        return -1
    if isinstance(value, str) and value.lower() == "none":
        return 10**9
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**9 - 1


def sort_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_sort_top_k"] = df["top_k"].apply(top_k_sort_value)
    df = df.sort_values(by=["classifier", "random_state", "feature_selector", "_sort_top_k", "experiment"], kind="mergesort").drop(columns=["_sort_top_k"])
    return df


def summarize_classifier(classifier_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment_dir in sorted(p for p in classifier_dir.iterdir() if p.is_dir()):
        summary_path = experiment_dir / "summary.json"
        skipped_path = experiment_dir / "SKIPPED.json"
        if summary_path.exists():
            rows.append(build_completed_row(classifier_dir.name, experiment_dir, load_json(summary_path)))
        elif skipped_path.exists():
            rows.append(build_skipped_row(classifier_dir.name, experiment_dir, load_json(skipped_path)))
        else:
            rows.append(build_incomplete_row(classifier_dir.name, experiment_dir))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for column in FULL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return sort_summary(df[FULL_COLUMNS])


def compact_summary(full_df: pd.DataFrame) -> pd.DataFrame:
    if full_df.empty:
        return full_df
    df = full_df.copy()
    for column in COMPACT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return sort_summary(df[COMPACT_COLUMNS])


def safe_sheet_name(name: str) -> str:
    clean = re.sub(r"[\[\]:*?/\\]", "_", name)
    return clean[:31] or "Sheet"


def autosize_sheet(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    worksheet = writer.sheets[sheet_name]
    for idx, column in enumerate(df.columns, start=1):
        values = [str(column)] + [str(v) for v in df[column].fillna("").head(200)]
        width = min(max(len(v) for v in values) + 2, 80)
        worksheet.column_dimensions[worksheet.cell(row=1, column=idx).column_letter].width = width
    worksheet.freeze_panes = "A2"


def write_sheet(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    sheet = safe_sheet_name(sheet_name)
    df.to_excel(writer, sheet_name=sheet, index=False)
    autosize_sheet(writer, sheet, df)


def main() -> None:
    args = parse_args()
    models_root = args.models_root or (MODEL_ROOT / args.task / IMAGE_TYPE)
    out_path = args.out_path or (models_root / DEFAULT_OUTPUT_NAME)
    if not models_root.exists():
        raise FileNotFoundError(f"Models root does not exist: {models_root}")
    classifier_dirs = discover_classifier_dirs(models_root)
    if not classifier_dirs:
        raise RuntimeError(f"No classifier folders found under: {models_root}")
    full_by_classifier: dict[str, pd.DataFrame] = {}
    for classifier_dir in classifier_dirs:
        df = summarize_classifier(classifier_dir)
        if not df.empty:
            full_by_classifier[classifier_dir.name] = df
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if not full_by_classifier:
            write_sheet(writer, "Summary", pd.DataFrame({"message": ["No experiments found."]}))
        else:
            all_full = sort_summary(pd.concat(full_by_classifier.values(), ignore_index=True))
            all_compact = compact_summary(all_full)
            write_sheet(writer, "All_full", all_full)
            write_sheet(writer, "All_compact", all_compact)
            for classifier_name, full_df in full_by_classifier.items():
                write_sheet(writer, f"{classifier_name}_full", full_df)
                write_sheet(writer, f"{classifier_name}_compact", compact_summary(full_df))
    print(f"Saved summary workbook: {out_path}")


if __name__ == "__main__":
    main()
