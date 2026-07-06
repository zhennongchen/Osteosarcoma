import os
import sys
from pathlib import Path

sys.path.append('/host/d/Github/')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Osteosarcoma.Build_lists.Build_list as Build_list
import Osteosarcoma.Image_3D.Generator_ResNet as Generator_ResNet
import Osteosarcoma.Image_3D.resnet.model as resnet_model

LABEL = 'Prognosis'
RANDOM_STATE = 0
VAL_FOLD = 4
TRAIN_FOLDS = [0, 1, 2, 3]
MODEL_DEPTH = 18
IN_CHANNELS = 3
EPOCHS = [45, 120]
TRIAL_DIR = Path('/host/d/projects/Habitats/models/Prognosis/resnet18_3D_FTall_AUGfull_96x96x64_nomed_adam/random0_fold4')
OUT_ROOT = TRIAL_DIR / 'gradcam_3d'
DATA_ROOT = '/host/e/D/Data/Habitats/Jishuitan/resampled_data_new'
PATIENT_LIST_FILE = f'/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12_5fold_{LABEL.lower()}_random{RANDOM_STATE}.xlsx'
TARGET_SIZE = (96, 96, 64)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 20
TOP_Z = 6


def build_dataset(folds):
    build = Build_list.Build(PATIENT_LIST_FILE)
    _, ps, pi, labels, _, _ = build.__build__(batch_list=folds, label_column_name=LABEL + '_label')
    x_files = [os.path.join(DATA_ROOT, ps[i], pi[i], 'img.nii.gz') for i in range(len(pi))]
    y = [int(v) for v in labels]
    ds = Generator_ResNet.Dataset_3D(
        ps, pi, x_files, y, DATA_ROOT,
        target_image_size=TARGET_SIZE,
        normalize_factor='medicalnet',
        only_tumor_pixels='seg',
        augment_context='full',
        shuffle=False,
        augment=False,
        augment_frequency=0,
    )
    return ds


def load_model(epoch):
    ckpt_path = TRIAL_DIR / 'models' / f'model-{epoch}.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    model = resnet_model.build_resnet3d_model(model_depth=MODEL_DEPTH, num_classes=2, in_channels=IN_CHANNELS)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    cleaned = {}
    for k, v in state.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, ckpt_path


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.activations = []
        self.gradients = []
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activations.append(output)

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def clear(self):
        self.activations = []
        self.gradients = []

    def close(self):
        for h in self.handles:
            h.remove()

    def compute_cam(self):
        act = self.activations[-1]
        grad = self.gradients[-1]
        weights = grad.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=TARGET_SIZE, mode='trilinear', align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        if np.max(cam) > np.min(cam):
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        else:
            cam = np.zeros_like(cam)
        return cam


def predict_dataset(model, ds, name, out_dir):
    rows = []
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    offset = 0
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            labels = y.numpy().astype(int)
            for j in range(len(labels)):
                i = offset + j
                rows.append({
                    'idx': i,
                    'patient_set': ds.patient_set_list[i],
                    'patient_index': ds.patient_index_list[i],
                    'label': int(labels[j]),
                    'prob_class1': float(probs[j]),
                    'pred_class': int(preds[j]),
                    'correct': int(preds[j]) == int(labels[j]),
                })
            offset += len(labels)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f'{name}_predictions.csv', index=False)
    auc = roc_auc_score(df['label'], df['prob_class1']) if df['label'].nunique() == 2 else np.nan
    print(f'{name} AUC={auc:.4f}, prob range=({df.prob_class1.min():.4g}, {df.prob_class1.max():.4g})')
    print(df.groupby('label')['prob_class1'].describe().to_string())
    return df


def pick_cases(pred_df, subset_name, n=3):
    chosen = []
    if subset_name == 'train':
        frames = [
            pred_df[(pred_df.label == 1) & (pred_df.pred_class == 1)].sort_values('prob_class1', ascending=False),
            pred_df[(pred_df.label == 0) & (pred_df.pred_class == 0)].sort_values('prob_class1', ascending=True),
            pred_df[pred_df.correct == False].sort_values('prob_class1', ascending=False),
            pred_df.sort_values('prob_class1', ascending=False),
        ]
    else:
        wrong = pred_df[pred_df.correct == False]
        frames = [
            wrong[wrong.label == 1].sort_values('prob_class1', ascending=True),
            wrong[wrong.label == 0].sort_values('prob_class1', ascending=False),
            pred_df.sort_values('prob_class1', ascending=False),
            pred_df.sort_values('prob_class1', ascending=True),
        ]
    for frame in frames:
        for idx in frame['idx'].tolist():
            if idx not in chosen:
                chosen.append(idx)
                break
        if len(chosen) >= n:
            break
    for idx in pred_df['idx'].tolist():
        if len(chosen) >= n:
            break
        if idx not in chosen:
            chosen.append(idx)
    return chosen[:n]


def normalize_display(img):
    img = img.astype(float)
    valid = img[img > 0]
    if valid.size > 10:
        lo, hi = np.percentile(valid, [1, 99])
    else:
        lo, hi = float(np.min(img)), float(np.max(img))
    if hi <= lo:
        hi = lo + 1e-8
    return np.clip((img - lo) / (hi - lo), 0, 1)


def overlay_cam(gray, cam, alpha=0.45):
    cmap = plt.get_cmap('jet')(cam)[..., :3]
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    return np.clip((1 - alpha) * rgb + alpha * cmap, 0, 1)


def region_energy(cam, mask):
    total = float(cam.sum())
    if total <= 0:
        return np.nan
    return float(cam[mask].sum() / total)


def top_z_indices(cam, top_n=6):
    z_scores = cam.sum(axis=(0, 1))
    order = np.argsort(z_scores)[::-1]
    top = sorted(order[:top_n].tolist())
    return top, z_scores


def run_one_case(model, gradcam, ds, subset_name, case_order, idx, target_mode, out_dir):
    ps = ds.patient_set_list[idx]
    pi = ds.patient_index_list[idx]
    y = int(ds.y_list[idx])
    img_file = ds.x_file_list[idx]
    label_file = os.path.join(DATA_ROOT, ps, pi, 'label.nii.gz')
    bbox_file = os.path.join(DATA_ROOT, ps, pi, 'bbox_mask.nii.gz')
    raw = ds.load_file(img_file, label_file, bbox_file)  # [3,X,Y,Z], channels are full,bbox-only,tumor-only images
    x, _ = ds[idx]
    x = x.unsqueeze(0).to(DEVICE)

    model.eval()
    gradcam.clear()
    model.zero_grad(set_to_none=True)
    logits = model(x)
    prob1 = torch.softmax(logits, dim=1)[0, 1].item()
    pred = int(torch.argmax(logits, dim=1).item())
    target_class = pred if target_mode == 'predclass' else 1
    logits[0, target_class].backward()
    cam = gradcam.compute_cam()

    z_list, z_scores = top_z_indices(cam, TOP_Z)
    fig, axes = plt.subplots(len(z_list), 4, figsize=(14, 2.7 * len(z_list)), dpi=150)
    if len(z_list) == 1:
        axes = axes[None, :]
    records = []
    for r, z in enumerate(z_list):
        full = raw[0, :, :, z]
        bbox_img = raw[1, :, :, z]
        tumor_img = raw[2, :, :, z]
        bbox_mask = bbox_img > 0
        tumor_mask = tumor_img > 0
        full_mask = full > 0
        cam_slice = cam[:, :, z]
        gray = normalize_display(full)
        panels = [normalize_display(full), normalize_display(bbox_img), normalize_display(tumor_img), overlay_cam(gray, cam_slice)]
        titles = [f'full z={z}', 'bbox-only image', 'tumor-only image', 'Grad-CAM overlay']
        for j, ax in enumerate(axes[r]):
            if j == 3:
                ax.imshow(panels[j])
                if np.any(tumor_mask):
                    ax.contour(tumor_mask, colors='lime', linewidths=0.7)
                if np.any(bbox_mask):
                    ax.contour(bbox_mask, colors='white', linewidths=0.4, alpha=0.8)
            else:
                ax.imshow(panels[j], cmap='gray')
            ax.set_title(titles[j], fontsize=9)
            ax.axis('off')
        records.append({
            'subset': subset_name,
            'case_order': case_order,
            'patient_set': ps,
            'patient_index': pi,
            'label': y,
            'prob_class1': prob1,
            'pred_class': pred,
            'target_mode': target_mode,
            'target_class_for_cam': target_class,
            'z': z,
            'z_cam_sum': float(z_scores[z]),
            'cam_energy_in_tumor_slice': region_energy(cam_slice, tumor_mask),
            'cam_energy_in_bbox_slice': region_energy(cam_slice, bbox_mask),
            'cam_energy_in_full_slice': region_energy(cam_slice, full_mask),
            'tumor_area_fraction_slice': float(np.mean(tumor_mask)),
            'bbox_area_fraction_slice': float(np.mean(bbox_mask)),
        })
    vol_tumor = raw[2] > 0
    vol_bbox = raw[1] > 0
    vol_full = raw[0] > 0
    for rec in records:
        rec['cam_energy_in_tumor_volume'] = region_energy(cam, vol_tumor)
        rec['cam_energy_in_bbox_volume'] = region_energy(cam, vol_bbox)
        rec['cam_energy_in_full_volume'] = region_energy(cam, vol_full)
        rec['tumor_volume_fraction'] = float(np.mean(vol_tumor))
        rec['bbox_volume_fraction'] = float(np.mean(vol_bbox))

    fig.suptitle(f'{subset_name} case {case_order}: {ps}/{pi} | y={y} | p1={prob1:.3f} | pred={pred} | target={target_class} | {target_mode}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = out_dir / f'{subset_name}_case{case_order}_{ps}_{pi}_y{y}_p{prob1:.3f}_pred{pred}_{target_mode}.png'
    fig.savefig(out_png)
    plt.close(fig)
    return records


def run_epoch(epoch):
    out_dir = OUT_ROOT / f'epoch{epoch}_evalmode'
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = build_dataset(TRAIN_FOLDS)
    val_ds = build_dataset([VAL_FOLD])
    model, ckpt_path = load_model(epoch)
    print('\n============================================================')
    print('epoch:', epoch)
    print('checkpoint:', ckpt_path)
    print('out_dir:', out_dir)
    train_pred = predict_dataset(model, train_ds, 'train', out_dir)
    val_pred = predict_dataset(model, val_ds, 'val', out_dir)
    train_indices = pick_cases(train_pred, 'train', n=3)
    val_indices = pick_cases(val_pred, 'val', n=3)
    print('representative train')
    print(train_pred[train_pred.idx.isin(train_indices)].to_string(index=False))
    print('representative val')
    print(val_pred[val_pred.idx.isin(val_indices)].to_string(index=False))

    gradcam = GradCAM3D(model, model.layer4[-1])
    try:
        for target_mode in ['predclass', 'class1']:
            mode_dir = out_dir / target_mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            records = []
            for order, idx in enumerate(train_indices, start=1):
                records.extend(run_one_case(model, gradcam, train_ds, 'train', order, idx, target_mode, mode_dir))
            for order, idx in enumerate(val_indices, start=1):
                records.extend(run_one_case(model, gradcam, val_ds, 'val', order, idx, target_mode, mode_dir))
            df = pd.DataFrame(records)
            df.to_csv(mode_dir / 'gradcam_region_summary.csv', index=False)
            # One row per case for volume-level interpretation.
            case = df.groupby(['subset','case_order','patient_set','patient_index','label','prob_class1','pred_class'])[[
                'cam_energy_in_tumor_volume','cam_energy_in_bbox_volume','cam_energy_in_full_volume','tumor_volume_fraction','bbox_volume_fraction'
            ]].mean().reset_index()
            case['tumor_enrichment_volume'] = case['cam_energy_in_tumor_volume'] / (case['tumor_volume_fraction'] + 1e-9)
            case['bbox_enrichment_volume'] = case['cam_energy_in_bbox_volume'] / (case['bbox_volume_fraction'] + 1e-9)
            case.to_csv(mode_dir / 'gradcam_case_summary.csv', index=False)
            print('\n==== epoch', epoch, target_mode)
            print(case.groupby('subset')[['cam_energy_in_tumor_volume','cam_energy_in_bbox_volume','cam_energy_in_full_volume','tumor_enrichment_volume','bbox_enrichment_volume']].mean().to_string())
    finally:
        gradcam.close()


def main():
    print('device:', DEVICE)
    for epoch in EPOCHS:
        run_epoch(epoch)


if __name__ == '__main__':
    main()
