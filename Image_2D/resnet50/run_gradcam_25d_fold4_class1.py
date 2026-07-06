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
CHECKPOINT_EPOCH = 30
TRIAL_DIR = Path('/host/d/projects/Habitats/models/Prognosis/resnet18_2.5D_FTall_AUGfull/random0_fold4')
CHECKPOINT_PATH = TRIAL_DIR / 'models' / f'model-{CHECKPOINT_EPOCH}.pt'
OUT_DIR = TRIAL_DIR / 'gradcam' / f'epoch{CHECKPOINT_EPOCH}_layer4_class1'
DATA_ROOT = '/host/e/D/Data/Habitats/Jishuitan/resampled_data_new'
PATIENT_LIST_FILE = f'/host/e/D/Data/Habitats/Jishuitan/Patient_lists/image_label_info_set12_5fold_{LABEL.lower()}_random{RANDOM_STATE}.xlsx'
TARGET_SIZE = (144, 144)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pick_cases(patient_set_list, patient_index_list, label_list, n=3):
    labels = [int(x) for x in label_list]
    chosen = []
    # Try to include both labels if possible.
    for target in [1, 0, 1, 0]:
        for i, y in enumerate(labels):
            if y == target and i not in chosen:
                chosen.append(i)
                break
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        for i in range(len(labels)):
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= n:
                break
    return chosen[:n]


def build_dataset(folds):
    build = Build_list.Build(PATIENT_LIST_FILE)
    fold_list, ps, pi, labels, _, _ = build.__build__(
        batch_list=folds,
        label_column_name=LABEL + '_label',
    )
    x_files = [os.path.join(DATA_ROOT, ps[i], pi[i], 'img_slices.nii.gz') for i in range(len(pi))]
    y = [int(v) for v in labels]
    ds = Generator.Dataset_2D(
        ps,
        pi,
        x_files,
        y,
        DATA_ROOT,
        target_image_size=TARGET_SIZE,
        normalize_factor='equation',
        only_tumor_pixels='roi',
        augment_context='full',
        shuffle=False,
        augment=False,
        augment_frequency=0,
    )
    return ds, fold_list, ps, pi, y, x_files


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
    model.train(True)  # user requested train-mode style diagnostics
    return model


class MultiCallGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
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
        # Backward hooks are returned in reverse call order in many PyTorch versions.
        acts = self.activations
        grads = list(reversed(self.gradients))
        cams = []
        for act, grad in zip(acts, grads):
            # act/grad: [1,C,h,w]
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


def make_display_image(raw_slice):
    # raw_slice shape [3,H,W], use full channel for anatomy display.
    img = raw_slice[0].astype(float)
    lo, hi = np.percentile(img[img > 0], [1, 99]) if np.any(img > 0) else (np.min(img), np.max(img))
    if hi <= lo:
        lo, hi = np.min(img), np.max(img)
    out = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return out


def overlay_cam(gray, cam, alpha=0.45):
    cmap = plt.get_cmap('jet')(cam)[..., :3]
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    return np.clip((1 - alpha) * rgb + alpha * cmap, 0, 1)


def region_energy(cam, mask):
    total = float(cam.sum())
    if total <= 0:
        return np.nan
    return float(cam[mask].sum() / total)


def run_one_case(model, gradcam, ds, subset_name, case_pos, global_idx):
    ps = ds.patient_set_list[global_idx]
    pi = ds.patient_index_list[global_idx]
    y = int(ds.y_list[global_idx])
    img_file = ds.x_file_list[global_idx]
    label_file = os.path.join(DATA_ROOT, ps, pi, 'label_slices.nii.gz')
    bbox_file = os.path.join(DATA_ROOT, ps, pi, 'bbox_mask_slices.nii.gz')

    raw = ds.load_file(img_file, label_file, bbox_file)  # [3,H,W,3], before normalization
    front, middle, rear, label_tensor = ds[global_idx]
    inputs = [front.unsqueeze(0).to(DEVICE), middle.unsqueeze(0).to(DEVICE), rear.unsqueeze(0).to(DEVICE)]
    slice_names = ['front', 'middle', 'rear']

    gradcam.clear()
    model.zero_grad(set_to_none=True)
    logits_list = [model(x) for x in inputs]
    logits = sum(logits_list) / 3.0
    prob1 = torch.softmax(logits, dim=1)[0, 1].item()
    pred_class = int(torch.argmax(logits, dim=1).item())
    score = logits[0, 1]
    score.backward()
    cams = gradcam.compute_cams()

    fig, axes = plt.subplots(3, 4, figsize=(14, 11), dpi=150)
    records = []
    for s, (name, cam) in enumerate(zip(slice_names, cams)):
        raw_slice = raw[:, :, :, s]  # [semantic,H,W]
        full = raw_slice[0]
        bbox_mask = raw_slice[1] > 0
        tumor_mask = raw_slice[2] > 0
        full_mask = full > 0
        gray = make_display_image(raw_slice)
        overlay = overlay_cam(gray, cam)

        panels = [gray, bbox_mask.astype(float), tumor_mask.astype(float), overlay]
        titles = ['full image', 'bbox mask', 'tumor mask', f'Grad-CAM class1 {name}']
        for j in range(4):
            ax = axes[s, j]
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
            'case_order': case_pos,
            'patient_set': ps,
            'patient_index': pi,
            'label': y,
            'prob_class1': prob1,
            'pred_class': pred_class,
            'target_class_for_cam': 1,
            'slice': name,
            'cam_energy_in_tumor': region_energy(cam, tumor_mask),
            'cam_energy_in_bbox': region_energy(cam, bbox_mask),
            'cam_energy_in_full_nonzero': region_energy(cam, full_mask),
            'tumor_area_fraction': float(np.mean(tumor_mask)),
            'bbox_area_fraction': float(np.mean(bbox_mask)),
        })

    fig.suptitle(
        f'{subset_name} case {case_pos}: {ps}/{pi} | y={y} | prob1={prob1:.3f} | pred={pred_class} | epoch={CHECKPOINT_EPOCH}',
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = OUT_DIR / f'{subset_name}_case{case_pos}_{ps}_{pi}_y{y}_p{prob1:.3f}_pred{pred_class}.png'
    fig.savefig(out_png)
    plt.close(fig)
    return records



def predict_dataset(model, ds):
    model.train(True)
    rows=[]
    with torch.no_grad():
        for i in range(len(ds)):
            front, middle, rear, y = ds[i]
            inputs=[front.unsqueeze(0).to(DEVICE), middle.unsqueeze(0).to(DEVICE), rear.unsqueeze(0).to(DEVICE)]
            logits=sum(model(x) for x in inputs)/3.0
            prob1=torch.softmax(logits, dim=1)[0,1].item()
            pred=int(torch.argmax(logits, dim=1).item())
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


def pick_representative_cases(pred_df, subset_name, n=3):
    chosen=[]
    if subset_name == 'train':
        # Confident memorized examples: high-prob true positive, low-prob true negative,
        # and another confident correct example from the opposite/ranking side.
        tp = pred_df[(pred_df.label==1) & (pred_df.pred_class==1)].sort_values('prob_class1', ascending=False)
        tn = pred_df[(pred_df.label==0) & (pred_df.pred_class==0)].sort_values('prob_class1', ascending=True)
        fp_or_fn = pred_df[pred_df.correct == False].copy()
        for frame in [tp, tn, fp_or_fn.sort_values('prob_class1', ascending=False), pred_df.sort_values('prob_class1', ascending=False), pred_df.sort_values('prob_class1', ascending=True)]:
            for idx in frame['idx'].tolist():
                if idx not in chosen:
                    chosen.append(idx)
                    break
                
            if len(chosen) >= n:
                break
    else:
        # Validation: prioritize wrong cases, then extremes.
        wrong = pred_df[pred_df.correct == False].copy()
        fn = wrong[wrong.label==1].sort_values('prob_class1', ascending=True)
        fp = wrong[wrong.label==0].sort_values('prob_class1', ascending=False)
        for frame in [fn, fp, wrong.sort_values('prob_class1', ascending=False), pred_df.sort_values('prob_class1', ascending=False), pred_df.sort_values('prob_class1', ascending=True)]:
            for idx in frame['idx'].tolist():
                if idx not in chosen:
                    chosen.append(idx)
                    break
            if len(chosen) >= n:
                break
    if len(chosen) < n:
        for idx in pred_df['idx'].tolist():
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= n:
                break
    return chosen[:n]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, *_ = build_dataset(TRAIN_FOLDS)
    val_ds, *_ = build_dataset([VAL_FOLD])
    train_indices = pick_cases(train_ds.patient_set_list, train_ds.patient_index_list, train_ds.y_list, n=3)
    val_indices = pick_cases(val_ds.patient_set_list, val_ds.patient_index_list, val_ds.y_list, n=3)
    print('checkpoint:', CHECKPOINT_PATH)
    print('out_dir:', OUT_DIR)
    print('train indices:', train_indices)
    print('val indices:', val_indices)

    model = load_model()

    train_pred = predict_dataset(model, train_ds)
    val_pred = predict_dataset(model, val_ds)
    train_pred.to_csv(OUT_DIR / 'train_predictions_epoch30.csv', index=False)
    val_pred.to_csv(OUT_DIR / 'val_predictions_epoch30.csv', index=False)
    train_indices = pick_representative_cases(train_pred, 'train', n=3)
    val_indices = pick_representative_cases(val_pred, 'val', n=3)
    print('representative train indices:', train_indices)
    print(train_pred[train_pred.idx.isin(train_indices)].to_string(index=False))
    print('representative val indices:', val_indices)
    print(val_pred[val_pred.idx.isin(val_indices)].to_string(index=False))

    gradcam = MultiCallGradCAM(model, model.layer4[-1])
    all_records = []
    try:
        for order, idx in enumerate(train_indices, start=1):
            all_records.extend(run_one_case(model, gradcam, train_ds, 'train', order, idx))
        for order, idx in enumerate(val_indices, start=1):
            all_records.extend(run_one_case(model, gradcam, val_ds, 'val', order, idx))
    finally:
        gradcam.close()

    df = pd.DataFrame(all_records)
    csv_path = OUT_DIR / 'gradcam_region_summary.csv'
    df.to_csv(csv_path, index=False)
    print('saved csv:', csv_path)
    print(df.groupby('subset')[['cam_energy_in_tumor','cam_energy_in_bbox','cam_energy_in_full_nonzero']].mean().to_string())


if __name__ == '__main__':
    main()
