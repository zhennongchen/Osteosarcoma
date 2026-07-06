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
import Osteosarcoma.Image_2D.Generator as Generator
import Osteosarcoma.Image_2D.resnet50.model as model_module


# ============================================================
# Manual defaults
# ============================================================
# You can edit these, or override them from command line.

DEFAULT_MODEL_RANDOM_STATE = 0
DEFAULT_MODEL_VAL_FOLD = 5
DEFAULT_PRED_FOLDS = [0,1,2,3,4,5]

FOLD_LIST = [0, 1, 2, 3, 4, 5]


# ============================================================
# Fixed paths and model/data settings matched to current train.py
# ============================================================

PATIENT_LIST_ROOT = '/host/e/D/Data/Habitats/Jishuitan/Patient_lists'
DATA_ROOT = '/host/e/D/Data/Habitats/Jishuitan/resampled_data_new'
MODEL_ROOT = '/host/d/projects/Habitats/models'
IMAGE_SIZE = (144, 144)


# ============================================================
# Helpers
# ============================================================

def none_or_path(value):
    if value is None:
        return None
    if str(value).lower() in {'none', 'null', ''}:
        return None
    return value


def parse_fold_list(value):
    """Parse fold strings like '4', '012345', '0,1,2,3,4', or '0 1 2 3 4'."""
    if value is None:
        return DEFAULT_PRED_FOLDS
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    value = str(value).strip()
    if value == '':
        return DEFAULT_PRED_FOLDS
    if ',' in value or ' ' in value:
        return [int(v) for v in value.replace(',', ' ').split()]
    return [int(ch) for ch in value]


def fold_tag(folds):
    return ''.join(str(fold) for fold in folds)


def setting_suffix_from_split(split_mode, val_fold):
    val_folds = parse_fold_list(val_fold)
    if val_folds is None or len(val_folds) == 0:
        raise ValueError('--val_fold must contain at least one fold.')
    if str(split_mode).lower() == 'cv':
        if len(val_folds) != 1:
            raise ValueError(f'cv mode requires one validation fold. Got: {val_folds}')
        return f'fold{fold_tag(val_folds)}', val_folds
    if str(split_mode).lower() == 'all':
        return f'all_fold{fold_tag(val_folds)}', val_folds
    raise ValueError(f"--split_mode must be 'cv' or 'all'. Got: {split_mode}")


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
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    candidate_thresholds = np.unique(y_prob)
    if candidate_thresholds.size == 0:
        return {
            'Best_threshold': np.nan,
            'Best_threshold_score_se_plus_sp': np.nan,
            'Accuracy': np.nan,
            'Sensitivity': np.nan,
            'Specificity': np.nan,
            'TN': 0,
            'FP': 0,
            'FN': 0,
            'TP': 0,
        }

    best_threshold = float(candidate_thresholds[0])
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
        if score > best_score or (np.isclose(score, best_score) and accuracy > best_accuracy):
            best_score = score
            best_threshold = float(threshold)
            best_accuracy = float(accuracy)
            best_sensitivity = float(sensitivity)
            best_specificity = float(specificity)
            best_cm = (tn, fp, fn, tp)

    tn, fp, fn, tp = best_cm
    return {
        'Best_threshold': best_threshold,
        'Best_threshold_score_se_plus_sp': float(best_score),
        'Accuracy': best_accuracy,
        'Sensitivity': best_sensitivity,
        'Specificity': best_specificity,
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
    }


def dataset_metrics(y_true, y_prob):
    auc_ci_low, auc_ci_high = bootstrap_auc_ci(y_true, y_prob)
    metrics = {'AUC': safe_auc(y_true, y_prob), 'AUC_ci_low': auc_ci_low, 'AUC_ci_high': auc_ci_high}
    metrics.update(find_best_threshold_by_youden(y_true, y_prob))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Predict with trained 2D / 2.5D CNN. Uses current Image_2D train/generator settings.')
    parser.add_argument('--label', type=str, default='Prognosis')
    parser.add_argument('--trial_name', type=str, required=True, help='Exact model folder name. No automatic name building is performed.')
    parser.add_argument('--model_family', type=str, default='efficientnet', choices=['resnet', 'efficientnet', 'shallow_cnn'])
    parser.add_argument('--model_depth', type=int, default=18, choices=[18, 34, 50])
    parser.add_argument('--efficientnet_version', type=str, default='b0', choices=['b0', 'b1', 'b2', 'b3', 'v2_s'])
    parser.add_argument('--init_channel', type=int, default=8)
    parser.add_argument('--fine_tune_stage', type=str, default='all', choices=['all', 'fc', '1', '2'])
    parser.add_argument('--only_tumor_pixels', type=str, default='roi', choices=['roi', 'seg'], help='Kept for train.py compatibility; generator returns [full,bbox,tumor].')
    parser.add_argument('--augment_context', type=str, default='full', choices=['simple', 'full'])
    parser.add_argument('--input_mode', type=str, default='2.5d', choices=['2d', '2.5d'])
    parser.add_argument('--random_state', type=int, default=DEFAULT_MODEL_RANDOM_STATE)
    parser.add_argument('--split_mode', type=str, default='all', choices=['cv', 'all'])
    parser.add_argument('--val_fold', type=str, default=str(DEFAULT_MODEL_VAL_FOLD), help='Which trained fold model to load. Examples: 4 or 012345.')
    parser.add_argument('--pred_folds', type=parse_fold_list, default=DEFAULT_PRED_FOLDS, help='Prediction folds, e.g. "5" or "0,1,2,3,4".')
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--trained_model_path', type=none_or_path, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def load_state_dict_into_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        cleaned_state_dict[key] = value
    model_state_keys = model.state_dict().keys()
    if 'fc.weight' in cleaned_state_dict and 'fc.1.weight' in model_state_keys:
        cleaned_state_dict['fc.1.weight'] = cleaned_state_dict.pop('fc.weight')
    if 'fc.bias' in cleaned_state_dict and 'fc.1.bias' in model_state_keys:
        cleaned_state_dict['fc.1.bias'] = cleaned_state_dict.pop('fc.bias')
    model.load_state_dict(cleaned_state_dict, strict=True)
    return model


@torch.no_grad()
def predict_dataloader(model, dataloader, device, input_mode='2.5d'):
    model.eval()
    input_mode = str(input_mode).lower()
    if input_mode not in {'2d', '2.5d'}:
        raise ValueError(f"input_mode must be '2d' or '2.5d'. Got: {input_mode}")

    all_prob = []
    all_label = []

    for batch in tqdm(dataloader, desc='Predicting'):
        slice_front, slice_middle, slice_rear, y = batch

        if input_mode == '2d':
            logits = model(slice_middle.to(device=device, dtype=torch.float32))
        else:
            logits_front = model(slice_front.to(device=device, dtype=torch.float32))
            logits_middle = model(slice_middle.to(device=device, dtype=torch.float32))
            logits_rear = model(slice_rear.to(device=device, dtype=torch.float32))
            logits = (logits_front + logits_middle + logits_rear) / 3.0

        prob = torch.softmax(logits, dim=1)[:, 1]
        all_prob.extend(prob.detach().cpu().numpy().tolist())
        all_label.extend(y.detach().cpu().numpy().astype(int).tolist())

    return np.asarray(all_label, dtype=int), np.asarray(all_prob, dtype=float)


def main():
    args = parse_args()

    label = args.label
    label_type = f'{label}_label'
    label_lower = label.lower()
    trial_name = args.trial_name
    if trial_name.lower() in {'none', 'null', ''}:
        raise ValueError('--trial_name must be explicitly set. No automatic trial name is generated now.')

    setting_suffix, model_val_folds = setting_suffix_from_split(args.split_mode, args.val_fold)
    invalid_model_folds = [fold for fold in model_val_folds if fold not in FOLD_LIST]
    if invalid_model_folds:
        raise ValueError(f'--val_fold contains invalid folds: {invalid_model_folds}. Valid folds: {FOLD_LIST}.')
    invalid_pred_folds = [fold for fold in args.pred_folds if fold not in FOLD_LIST]
    if invalid_pred_folds:
        raise ValueError(f'--pred_folds contains invalid folds: {invalid_pred_folds}. Valid folds: {FOLD_LIST}.')

    patient_list_file = os.path.join(
        PATIENT_LIST_ROOT,
        f'image_label_info_set12_5fold_{label_lower}_random{args.random_state}.xlsx',
    )
    setting_output_path = os.path.join(
        MODEL_ROOT,
        label,
        trial_name,
        f'random{args.random_state}_{setting_suffix}',
    )

    trained_model_path = args.trained_model_path or os.path.join(
        setting_output_path,
        'models',
        f'model-{args.epoch}.pt',
    )

    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else ('cpu' if args.device == 'auto' else args.device))

    print('============================================================')
    print('2D/2.5D CNN prediction')
    print('label:', label)
    print('trial_name:', trial_name)
    print('model_family:', args.model_family)
    print('model_depth:', args.model_depth)
    print('efficientnet_version:', args.efficientnet_version)
    print('init_channel:', args.init_channel)
    print('fine_tune_stage:', args.fine_tune_stage)
    print('only_tumor_pixels:', args.only_tumor_pixels)
    print('augment_context:', args.augment_context)
    print('input_mode:', args.input_mode)
    print('epoch:', args.epoch)
    print('patient list:', patient_list_file)
    print('model random_state:', args.random_state)
    print('split_mode:', args.split_mode)
    print('model val_fold:', model_val_folds)
    print('prediction folds:', args.pred_folds)
    print('trained model:', trained_model_path)
    print('device:', device)
    print('image_size:', IMAGE_SIZE)
    print('============================================================')

    if not os.path.isfile(patient_list_file):
        raise FileNotFoundError(patient_list_file)
    if not os.path.isfile(trained_model_path):
        raise FileNotFoundError(trained_model_path)

    build = Build_list.Build(patient_list_file)
    fold_list_pred, patient_set_list_pred, patient_index_list_pred, label_list_pred, _, _ = build.__build__(
        batch_list=args.pred_folds,
        label_column_name=label_type,
    )

    x_file_list_pred = [
        os.path.join(DATA_ROOT, patient_set_list_pred[i], patient_index_list_pred[i], 'img_slices.nii.gz')
        for i in range(len(patient_index_list_pred))
    ]
    y_list_pred = [int(label_list_pred[i]) for i in range(len(label_list_pred))]

    print('Prediction cases:', len(patient_index_list_pred))
    print('Prediction label=1 fraction:', float(np.mean(np.asarray(y_list_pred).astype(int))))

    pred_dataset = Generator.Dataset_2D(
        patient_set_list_pred,
        patient_index_list_pred,
        x_file_list_pred,
        y_list_pred,
        DATA_ROOT,
        target_image_size=IMAGE_SIZE,
        normalize_factor='equation',
        only_tumor_pixels=args.only_tumor_pixels,
        augment_context=args.augment_context,
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

    dropout_p = 0.3
    if args.model_family == 'resnet':
        model = model_module.build_resnet_model(
            model_depth=args.model_depth,
            num_classes=2,
            use_imagenet=False,
            dropout_p=dropout_p,
        )
    elif args.model_family == 'efficientnet':
        model = model_module.build_efficientnet_model(
            efficientnet_version=args.efficientnet_version,
            num_classes=2,
            use_imagenet=False,
            dropout_p=dropout_p,
        )
    elif args.model_family == 'shallow_cnn':
        model = model_module.build_shallow_cnn_model(
            init_channel=args.init_channel,
            num_classes=2,
            in_channels=3,
            dropout_p=dropout_p,
        )
    else:
        raise ValueError(f'Unsupported model_family: {args.model_family}')
    model = load_state_dict_into_model(model, trained_model_path, device)
    model.to(device)
    model.eval()

    y_true, y_prob = predict_dataloader(model, pred_loader, device, input_mode=args.input_mode)
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
        'trial_name': trial_name,
        'epoch': args.epoch,
        'model_random_state': args.random_state,
        'model_val_fold': fold_tag(model_val_folds),
        'prediction_folds': ','.join(str(fold) for fold in args.pred_folds),
        'n_cases': int(len(y_true)),
        'positive_fraction': float(np.mean(y_true)),
        'model_family': args.model_family,
        'model_depth': args.model_depth,
        'efficientnet_version': args.efficientnet_version,
        'init_channel': args.init_channel,
        'fine_tune_stage': args.fine_tune_stage,
        'only_tumor_pixels': args.only_tumor_pixels,
        'augment_context': args.augment_context,
        'input_mode': args.input_mode,
        'eval_mode': 'model.eval',
        **metrics,
    }])

    prediction_out_dir = os.path.join(setting_output_path, 'predictions')
    ff.make_folder([prediction_out_dir])

    pred_tag = fold_tag(args.pred_folds)
    pred_path = os.path.join(prediction_out_dir, f'prediction_eval_epoch{args.epoch}_fold{pred_tag}.xlsx')
    metrics_path = os.path.join(prediction_out_dir, f'metrics_eval_epoch{args.epoch}_fold{pred_tag}.xlsx')

    pred_df.to_excel(pred_path, index=False)
    metrics_df.to_excel(metrics_path, index=False)

    print('\nPrediction metrics:')
    for key, value in metrics.items():
        print(f'  {key}: {value}')
    print('\nSaved prediction table:', pred_path)
    print('Saved metrics table:', metrics_path)


if __name__ == '__main__':
    main()
