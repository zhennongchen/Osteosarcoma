import os
import sys
from pathlib import Path

sys.path.append('/host/d/Github/')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Osteosarcoma.Build_lists.Build_list as Build_list
import Osteosarcoma.Image_2D.Generator as Generator
import Osteosarcoma.Image_2D.resnet50.model as model_module

LABEL = 'Prognosis'
RANDOM_STATE = 0
VAL_FOLD = 4
TRAIN_FOLDS = [0, 1, 2, 3]
MODEL_DEPTH = 18
CHECKPOINT_EPOCH = 115
TRIAL_DIR = Path('/host/d/projects/Habitats/models/Prognosis/resnet18_2.5D_FTall_AUGfull/random0_fold4')
CHECKPOINT_PATH = TRIAL_DIR / 'models' / f'model-{CHECKPOINT_EPOCH}.pt'
BASE_OUT_DIR = TRIAL_DIR / 'gradcam' / f'epoch{CHECKPOINT_EPOCH}_layer4_evalmode_corrected_display'
DATA_ROOT = '/host/e/D/Data/Habitats/Jishuitan/resampled_data_new'
PATIENT_LIST_FILE = f'/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12_5fold_{LABEL.lower()}_random{RANDOM_STATE}.xlsx'
TARGET_SIZE = (144, 144)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_dataset(folds):
    build = Build_list.Build(PATIENT_LIST_FILE)
    fold_list, ps, pi, labels, _, _ = build.__build__(batch_list=folds, label_column_name=LABEL + '_label')
    x_files = [os.path.join(DATA_ROOT, ps[i], pi[i], 'img_slices.nii.gz') for i in range(len(pi))]
    y = [int(v) for v in labels]
    ds = Generator.Dataset_2D(
        ps, pi, x_files, y, DATA_ROOT,
        target_image_size=TARGET_SIZE,
        normalize_factor='equation',
        only_tumor_pixels='roi',
        augment_context='full',
        shuffle=False,
        augment=False,
        augment_frequency=0,
    )
    return ds


def load_model():
    model = model_module.build_resnet_model(model_depth=MODEL_DEPTH, num_classes=2, use_imagenet=False)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    cleaned = {}
    for k, v in state.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


class MultiCallGradCAM:
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

    def compute_cams(self):
        acts = self.activations
        grads = list(reversed(self.gradients))
        cams = []
        for act, grad in zip(acts, grads):
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=TARGET_SIZE, mode='bilinear', align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
            if np.max(cam) > np.min(cam):
                cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
            else:
                cam = np.zeros_like(cam)
            cams.append(cam)
        return cams


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


def predict_dataset(model, ds):
    rows = []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            front, middle, rear, y = ds[i]
            inputs = [front.unsqueeze(0).to(DEVICE), middle.unsqueeze(0).to(DEVICE), rear.unsqueeze(0).to(DEVICE)]
            logits = sum(model(x) for x in inputs) / 3.0
            prob1 = torch.softmax(logits, dim=1)[0, 1].item()
            pred = int(torch.argmax(logits, dim=1).item())
            rows.append({
                'idx': i,
                'patient_set': ds.patient_set_list[i],
                'patient_index': ds.patient_index_list[i],
                'label': int(ds.y_list[i]),
                'prob_class1': prob1,
                'pred_class': pred,
                'correct': pred == int(ds.y_list[i]),
            })
    return pd.DataFrame(rows)


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


def run_one_case(model, gradcam, ds, subset_name, case_order, global_idx, target_mode, out_dir):
    ps = ds.patient_set_list[global_idx]
    pi = ds.patient_index_list[global_idx]
    y = int(ds.y_list[global_idx])
    img_file = ds.x_file_list[global_idx]
    label_file = os.path.join(DATA_ROOT, ps, pi, 'label_slices.nii.gz')
    bbox_file = os.path.join(DATA_ROOT, ps, pi, 'bbox_mask_slices.nii.gz')
    raw = ds.load_file(img_file, label_file, bbox_file)  # [3,H,W,3] channels are full,bbox-only,tumor-only images
    front, middle, rear, _ = ds[global_idx]
    inputs = [front.unsqueeze(0).to(DEVICE), middle.unsqueeze(0).to(DEVICE), rear.unsqueeze(0).to(DEVICE)]
    slice_names = ['front', 'middle', 'rear']

    model.eval()
    gradcam.clear()
    model.zero_grad(set_to_none=True)
    logits_list = [model(x) for x in inputs]
    logits = sum(logits_list) / 3.0
    prob1 = torch.softmax(logits, dim=1)[0, 1].item()
    pred = int(torch.argmax(logits, dim=1).item())
    target_class = pred if target_mode == 'predclass' else 1
    logits[0, target_class].backward()
    cams = gradcam.compute_cams()

    fig, axes = plt.subplots(3, 4, figsize=(14, 11), dpi=150)
    records = []
    for s, (slice_name, cam) in enumerate(zip(slice_names, cams)):
        raw_slice = raw[:, :, :, s]
        full_img = raw_slice[0]
        bbox_img = raw_slice[1]
        tumor_img = raw_slice[2]
        bbox_mask = bbox_img > 0
        tumor_mask = tumor_img > 0
        full_mask = full_img > 0
        gray = normalize_display(full_img)
        overlay = overlay_cam(gray, cam)
        panels = [normalize_display(full_img), normalize_display(bbox_img), normalize_display(tumor_img), overlay]
        titles = ['channel 1: full image', 'channel 2: bbox-only image', 'channel 3: tumor-only image', f'Grad-CAM {target_mode} {slice_name}']
        for j, ax in enumerate(axes[s]):
            if j == 3:
                ax.imshow(panels[j])
                ax.contour(tumor_mask, colors='lime', linewidths=0.6)
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
            'target_class_for_cam': target_class,
            'target_mode': target_mode,
            'slice': slice_name,
            'cam_energy_in_tumor': region_energy(cam, tumor_mask),
            'cam_energy_in_bbox': region_energy(cam, bbox_mask),
            'cam_energy_in_full_nonzero': region_energy(cam, full_mask),
            'tumor_area_fraction': float(np.mean(tumor_mask)),
            'bbox_area_fraction': float(np.mean(bbox_mask)),
        })
    fig.suptitle(f'{subset_name} case {case_order}: {ps}/{pi} | y={y} | p1={prob1:.3f} | pred={pred} | target={target_class} | epoch={CHECKPOINT_EPOCH}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = out_dir / f'{subset_name}_case{case_order}_{ps}_{pi}_y{y}_p{prob1:.3f}_pred{pred}_{target_mode}.png'
    fig.savefig(out_png)
    plt.close(fig)
    return records


def main():
    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = build_dataset(TRAIN_FOLDS)
    val_ds = build_dataset([VAL_FOLD])
    model = load_model()
    model.eval()
    train_pred = predict_dataset(model, train_ds)
    val_pred = predict_dataset(model, val_ds)
    train_pred.to_csv(BASE_OUT_DIR / 'train_predictions_epoch115.csv', index=False)
    val_pred.to_csv(BASE_OUT_DIR / 'val_predictions_epoch115.csv', index=False)
    train_indices = pick_cases(train_pred, 'train', n=3)
    val_indices = pick_cases(val_pred, 'val', n=3)
    print('checkpoint:', CHECKPOINT_PATH)
    print('out_dir:', BASE_OUT_DIR)
    print('representative train')
    print(train_pred[train_pred.idx.isin(train_indices)].to_string(index=False))
    print('representative val')
    print(val_pred[val_pred.idx.isin(val_indices)].to_string(index=False))

    gradcam = MultiCallGradCAM(model, model.layer4[-1])
    try:
        for target_mode in ['predclass', 'class1']:
            out_dir = BASE_OUT_DIR / target_mode
            out_dir.mkdir(parents=True, exist_ok=True)
            records = []
            for order, idx in enumerate(train_indices, start=1):
                records.extend(run_one_case(model, gradcam, train_ds, 'train', order, idx, target_mode, out_dir))
            for order, idx in enumerate(val_indices, start=1):
                records.extend(run_one_case(model, gradcam, val_ds, 'val', order, idx, target_mode, out_dir))
            df = pd.DataFrame(records)
            df.to_csv(out_dir / 'gradcam_region_summary.csv', index=False)
            df['tumor_enrichment'] = df['cam_energy_in_tumor'] / (df['tumor_area_fraction'] + 1e-9)
            df['bbox_enrichment'] = df['cam_energy_in_bbox'] / (df['bbox_area_fraction'] + 1e-9)
            print('\n====', target_mode)
            print(df.groupby('subset')[['cam_energy_in_tumor','cam_energy_in_bbox','cam_energy_in_full_nonzero','tumor_enrichment','bbox_enrichment']].mean().to_string())
    finally:
        gradcam.close()


if __name__ == '__main__':
    main()
