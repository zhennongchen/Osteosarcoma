import sys
sys.path.append('/host/d/Github/')

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import Osteosarcoma.Build_lists.Build_list as Build_list
import Osteosarcoma.functions_collection as ff
import Osteosarcoma.Image_3D.Generator as Generator
import Osteosarcoma.Image_3D.vit.model as vit_model


# ============================================================
# Manual dataset definition
# ============================================================
# Command line only chooses which model/epoch to load.
# The concrete prediction dataset is intentionally defined here.

MODEL_RANDOM_STATE = 0
MODEL_VAL_FOLD = 5

FOLD_LIST = [0, 1, 2, 3, 4, 5]

# Examples:
#   PRED_FOLD = [5]              -> internal test fold
#   PRED_FOLD = [0, 1, 2, 3, 4]  -> train/CV folds when MODEL_VAL_FOLD = 5
#   PRED_FOLD = [0]              -> one specific fold
PRED_FOLD = [5]


# ============================================================
# Fixed paths and model/data settings
# ============================================================

PATIENT_LIST_ROOT = '/host/e/D/Data/Habitats/Jishuitan/Patient_lists'
DATA_ROOT = '/host/e/D/Data/Habitats/Jishuitan/resampled_data'
MODEL_ROOT = '/host/d/projects/Habitats/models'

IMAGE_SIZE = (80,80, 96)
VIT_PATCH_SIZE = (16, 16, 4)


# ============================================================
# Small helpers
# ============================================================

def none_or_path(value):
    if value is None:
        return None
    if str(value).lower() in {'none', 'null', ''}:
        return None
    return value


def fold_tag(folds):
    return ''.join(str(fold) for fold in folds)


def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def bootstrap_auc_ci(y_true, y_score, n_bootstrap=2000, ci=0.95, random_state=0):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    auc_values = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[sample_idx])) < 2:
            continue
        auc_values.append(roc_auc_score(y_true[sample_idx], y_score[sample_idx]))
    if len(auc_values) == 0:
        return np.nan, np.nan
    alpha = (1.0 - ci) / 2.0
    low, high = np.percentile(auc_values, [100 * alpha, 100 * (1.0 - alpha)])
    return float(low), float(high)


def find_best_threshold_by_youden(y_true, y_prob):
    """
    Find the threshold maximizing sensitivity + specificity.

    This is equivalent to maximizing Youden index:
        sensitivity + specificity - 1
    Since the user asked for se + sp max, the constant -1 does not affect
    which threshold is selected.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    candidate_thresholds = np.unique(y_prob)
    if candidate_thresholds.size == 0:
        return np.nan

    best_threshold = candidate_thresholds[0]
    best_score = -np.inf
    best_accuracy = np.nan
    best_sensitivity = np.nan
    best_specificity = np.nan
    best_cm = (0, 0, 0, 0)

    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        if np.isnan(sensitivity) or np.isnan(specificity):
            continue

        score = sensitivity + specificity
        accuracy = accuracy_score(y_true, y_pred)

        # Tie-breaker: if se+sp is equal, prefer higher accuracy.
        if score > best_score or (np.isclose(score, best_score) and accuracy > best_accuracy):
            best_score = score
            best_threshold = threshold
            best_accuracy = accuracy
            best_sensitivity = sensitivity
            best_specificity = specificity
            best_cm = (tn, fp, fn, tp)

    tn, fp, fn, tp = best_cm
    return {
        'Best_threshold': float(best_threshold),
        'Best_threshold_score_se_plus_sp': float(best_score),
        'Accuracy': float(best_accuracy),
        'Sensitivity': float(best_sensitivity),
        'Specificity': float(best_specificity),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
    }


def dataset_metrics(y_true, y_prob):
    auc_ci_low, auc_ci_high = bootstrap_auc_ci(y_true, y_prob)
    metrics = {
        'AUC': safe_auc(y_true, y_prob),
        'AUC_ci_low': auc_ci_low,
        'AUC_ci_high': auc_ci_high,
    }
    metrics.update(find_best_threshold_by_youden(y_true, y_prob))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Predict with trained 3D ViT. Dataset folds are edited inside this file.'
    )

    parser.add_argument('--label', type=str, default='Prognosis')
    parser.add_argument('--trial_name', type=str, default='vit_3D')
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--trained_model_path', type=none_or_path, default=None)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto')

    return parser.parse_args()


def build_vit_model():
    """Define ViT3D exactly as in vit/train.py."""
    model = vit_model.ViT3D(
        image_size=IMAGE_SIZE,
        patch_size=VIT_PATCH_SIZE,
        in_channels=1,
        num_classes=2,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        attention_dropout=0.1,
    )
    return model


def load_state_dict_into_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    # Trainer saves dictionaries containing model / opt / ema states.
    # Use the actual model state by default, matching normal inference.
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True) 
    return model


@torch.no_grad()
def predict_dataloader(model, dataloader, device):
    model.eval()

    all_prob = []
    all_label = []

    for batch in tqdm(dataloader, desc='Predicting'):
        x, y = batch
        x = x.to(device=device, dtype=torch.float32)

        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1]

        all_prob.extend(prob.detach().cpu().numpy().tolist())
        all_label.extend(y.detach().cpu().numpy().astype(int).tolist())

    return np.asarray(all_label, dtype=int), np.asarray(all_prob, dtype=float)


def main():
    args = parse_args()

    label = args.label
    label_type = f'{label}_label'
    label_lower = label.lower()

    if MODEL_VAL_FOLD not in FOLD_LIST:
        raise ValueError(f'MODEL_VAL_FOLD must be one of {FOLD_LIST}. Got {MODEL_VAL_FOLD}.')

    invalid_pred_folds = [fold for fold in PRED_FOLD if fold not in FOLD_LIST]
    if len(invalid_pred_folds) > 0:
        raise ValueError(f'PRED_FOLD contains invalid folds: {invalid_pred_folds}. Valid folds: {FOLD_LIST}.')

    train_fold = [fold for fold in FOLD_LIST if fold != MODEL_VAL_FOLD]

    patient_list_file = os.path.join(
        PATIENT_LIST_ROOT,
        f'image_label_info_set12_5fold_{label_lower}_random{MODEL_RANDOM_STATE}.xlsx',
    )

    setting_output_path = os.path.join(
        MODEL_ROOT,
        label,
        'vit',
        args.trial_name,
        f'random{MODEL_RANDOM_STATE}_fold{MODEL_VAL_FOLD}',
    )

    if args.trained_model_path is None:
        trained_model_path = os.path.join(
            setting_output_path,
            'models',
            f'model-{args.epoch}.pt',
        )
    else:
        trained_model_path = args.trained_model_path

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print('============================================================')
    print('3D ViT prediction')
    print('label:', label)
    print('trial_name:', args.trial_name)
    print('epoch:', args.epoch)
    print('patient list:', patient_list_file)
    print('model random_state:', MODEL_RANDOM_STATE)
    print('model val_fold:', MODEL_VAL_FOLD)
    print('model train_fold:', train_fold)
    print('prediction fold:', PRED_FOLD)
    print('trained model:', trained_model_path)
    print('device:', device)
    print('============================================================')

    if not os.path.isfile(patient_list_file):
        raise FileNotFoundError(patient_list_file)
    if not os.path.isfile(trained_model_path):
        raise FileNotFoundError(trained_model_path)

    build = Build_list.Build(patient_list_file)
    fold_list_pred, patient_set_list_pred, patient_index_list_pred, label_list_pred, _, _ = build.__build__(
        batch_list=PRED_FOLD,
        label_column_name=label_type,
    )

    x_file_list_pred = [
        os.path.join(DATA_ROOT, patient_set_list_pred[i], patient_index_list_pred[i], 'img_n4.nii.gz')
        for i in range(len(patient_index_list_pred))
    ]
    y_list_pred = [int(label_list_pred[i]) for i in range(len(label_list_pred))]

    print('Prediction cases:', len(patient_index_list_pred))
    print('Prediction label=1 fraction:', float(np.mean(np.asarray(y_list_pred).astype(int))))

    pred_dataset = Generator.Dataset_3D(
        patient_set_list_pred,
        patient_index_list_pred,
        x_file_list_pred,
        y_list_pred,
        DATA_ROOT,
        target_image_size=IMAGE_SIZE,
        normalize_factor='equation',
        shuffle=False,
        augment=False,
        augment_frequency=0,
    )

    pred_loader = DataLoader(
        pred_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_vit_model()
    model = load_state_dict_into_model(model, trained_model_path, device)
    model.to(device)

    y_true, y_prob = predict_dataloader(model, pred_loader, device)
    metrics = dataset_metrics(y_true, y_prob)

    pred_df = pd.DataFrame({
        'Patient_set': patient_set_list_pred,
        'Patient_index': patient_index_list_pred,
        'fold': fold_list_pred,
        'label': y_true,
        'probability': y_prob,
    })

    metrics_df = pd.DataFrame([{
        'label': label,
        'trial_name': args.trial_name,
        'epoch': args.epoch,
        'model_random_state': MODEL_RANDOM_STATE,
        'model_val_fold': MODEL_VAL_FOLD,
        'prediction_folds': ','.join(str(fold) for fold in PRED_FOLD),
        'n_cases': int(len(y_true)),
        'positive_fraction': float(np.mean(y_true)),
        **metrics,
    }])

    prediction_out_dir = os.path.join(setting_output_path, 'predictions')
    ff.make_folder([prediction_out_dir])

    pred_tag = fold_tag(PRED_FOLD)
    pred_path = os.path.join(
        prediction_out_dir,
        f'prediction_epoch{args.epoch}_fold{pred_tag}.xlsx',
    )
    metrics_path = os.path.join(
        prediction_out_dir,
        f'metrics_epoch{args.epoch}_fold{pred_tag}.xlsx',
    )

    pred_df.to_excel(pred_path, index=False)
    metrics_df.to_excel(metrics_path, index=False)

    print('\nPrediction metrics:')
    for key, value in metrics.items():
        print(f'  {key}: {value}')

    print('\nSaved prediction table:', pred_path)
    print('Saved metrics table:', metrics_path)


if __name__ == '__main__':
    main()
