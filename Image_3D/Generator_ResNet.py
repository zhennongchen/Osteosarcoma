# dataset classes for MedicalNet-style 3D ResNet

import os
import random

import numpy as np
import nibabel as nb
from scipy import ndimage
import torch
from torch.utils.data import Dataset

import Osteosarcoma.Data_processing as Data_processing


# ============================================================
# Shared crop / normalization / augmentation helpers
# ============================================================

def center_crop_or_pad_by_bbox(arr, bbox_mask, target_shape, pad_value=0):
    arr = np.asarray(arr)
    bbox_mask = np.asarray(bbox_mask) > 0
    target_shape = tuple(int(v) for v in target_shape)

    if arr.shape != bbox_mask.shape:
        raise ValueError(f'arr and bbox_mask shapes do not match: {arr.shape} vs {bbox_mask.shape}')
    if arr.ndim != len(target_shape):
        raise ValueError(f'arr ndim {arr.ndim} does not match target_shape {target_shape}')

    coords = np.where(bbox_mask)
    if len(coords[0]) == 0:
        raise RuntimeError('Empty bbox mask.')

    center = [int(round((coords[d].min() + coords[d].max()) / 2.0)) for d in range(arr.ndim)]
    out = np.full(target_shape, pad_value, dtype=arr.dtype)

    src_slices = []
    dst_slices = []
    for d, target_len in enumerate(target_shape):
        start = center[d] - target_len // 2
        end = start + target_len

        src_start = max(start, 0)
        src_end = min(end, arr.shape[d])
        dst_start = max(0, -start)
        dst_end = dst_start + (src_end - src_start)

        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


def apply_percentile_cutoff(stacked_img, percentile=95):
    stacked_img = stacked_img.astype(np.float32, copy=True)
    bbox_channel = stacked_img[1]
    valid = bbox_channel[bbox_channel > 0]
    if valid.size == 0:
        return stacked_img
    cutoff = float(np.percentile(valid, percentile))
    stacked_img[stacked_img > cutoff] = cutoff
    return stacked_img


def medicalnet_zscore(volume):
    """Z-score non-zero voxels and keep zero background as zero.

    MedicalNet's original preprocessing may replace zero background with
    random noise, but that is harmful for this project because our 3 channels
    encode [full context, bbox-only, tumor-only]. The zero regions are semantic
    masks, so inference must keep them deterministic and meaningful.
    """
    volume = volume.astype(np.float32)
    foreground = volume != 0
    out = np.zeros_like(volume, dtype=np.float32)

    if np.any(foreground):
        pixels = volume[foreground]
        mean = float(pixels.mean())
        std = float(pixels.std())
        if std < 1e-6:
            std = 1.0
        out[foreground] = (volume[foreground] - mean) / std

    return out.astype(np.float32)



def restore_zero_background_after_intensity_aug(i, original_nonzero_mask):
    """Keep semantic zero regions zero after intensity augmentation.

    In the current 3-channel design, zeros mean outside full/bbox/tumor
    support. Noise, contrast, and sharpness can otherwise turn padded or
    masked-out voxels into non-zero foreground.
    """
    out = i.astype(np.float32, copy=True)
    out[~original_nonzero_mask] = 0
    return out


def random_noise_channel_first(i, std_fraction=0.02):
    dynamic_range = float(np.max(i) - np.min(i))
    noise_std = dynamic_range * std_fraction
    if noise_std <= 0:
        return i, noise_std
    noise = np.random.normal(0, noise_std, size=i.shape).astype(np.float32)
    return (i + noise).astype(np.float32), noise_std


def random_brightness_np(i, v=None):
    if v is None:
        v = np.random.uniform(0.9, 1.1)
    return (i * v).astype(np.float32), v


def random_contrast_np(i, v=None):
    if v is None:
        v = np.random.uniform(0.9, 1.1)
    valid = i[i != 0]
    mean = float(valid.mean()) if valid.size > 0 else float(i.mean())
    return (mean + (i - mean) * v).astype(np.float32), v


def random_sharpness_np(i, amount=None, sigma=1.0):
    if amount is None:
        amount = np.random.uniform(0.1, 0.4)
    out = np.empty_like(i, dtype=np.float32)
    for c in range(i.shape[0]):
        blur = ndimage.gaussian_filter(i[c], sigma=sigma)
        out[c] = i[c] + amount * (i[c] - blur)
    return out.astype(np.float32), amount


def random_flip_channel_first_3d(i):
    flip_x = random.choice([False, True])
    flip_y = random.choice([False, True])
    if flip_x:
        i = np.flip(i, axis=1)
    if flip_y:
        i = np.flip(i, axis=2)
    return np.ascontiguousarray(i), flip_x, flip_y


def random_rotate_channel_first_3d(i, degree=None, degree_range=(-10, 10), fill_val=None, order=1):
    if degree is None:
        degree = random.uniform(degree_range[0], degree_range[1])
    if fill_val is None:
        fill_val = float(np.min(i))
    if degree == 0:
        return i, degree
    out = ndimage.rotate(i, degree, axes=(1, 2), reshape=False, order=order, mode='constant', cval=fill_val)
    return out.astype(np.float32), degree


def random_translate_channel_first_3d(i, x_translate=None, y_translate=None, translate_range=(-10, 10), fill_val=None, order=1):
    if x_translate is None or y_translate is None:
        x_translate = int(random.uniform(translate_range[0], translate_range[1]))
        y_translate = int(random.uniform(translate_range[0], translate_range[1]))
    if fill_val is None:
        fill_val = float(np.min(i))
    out = ndimage.shift(i, shift=(0, x_translate, y_translate, 0), order=order, mode='constant', cval=fill_val)
    return out.astype(np.float32), x_translate, y_translate


class Dataset_3D(Dataset):
    def __init__(
        self,
        patient_set_list,
        patient_index_list,
        x_file_list,
        y_list,
        data_root,
        target_image_size,
        normalize_factor='medicalnet',
        only_tumor_pixels='seg',
        percentile_cutoff=95,
        augment_context='simple',
        shuffle=False,
        augment=False,
        augment_frequency=0,
    ):
        super().__init__()

        self.patient_set_list = patient_set_list
        self.patient_index_list = patient_index_list
        self.x_file_list = x_file_list
        self.y_list = y_list
        self.data_root = data_root

        self.image_size = target_image_size
        self.normalize_factor = normalize_factor
        self.only_tumor_pixels = only_tumor_pixels  # kept for backward-compatible train.py calls
        self.percentile_cutoff = percentile_cutoff
        self.augment_context = augment_context

        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.num_files = len(self.y_list)

        self.index_array = self.generate_index_array()
        self.current_input_file = None
        self.current_input_data = None
        self.current_y = None

    def generate_index_array(self):
        np.random.seed()
        if self.shuffle:
            f_list = np.random.permutation(self.num_files)
        else:
            f_list = np.arange(self.num_files)
        return [f for f in f_list]

    def __len__(self):
        return self.num_files

    def load_file(self, filename, label_filename=None, bbox_filename=None):
        img = nb.load(filename).get_fdata().astype(np.float32)
        if label_filename is None or bbox_filename is None:
            raise ValueError('3D generator requires label.nii.gz and bbox_mask.nii.gz filenames.')

        label = nb.load(label_filename).get_fdata()
        bbox = nb.load(bbox_filename).get_fdata() > 0

        if label.shape != img.shape or bbox.shape != img.shape:
            raise ValueError(f'Shape mismatch: img={img.shape}, label={label.shape}, bbox={bbox.shape}')

        target_shape = [self.image_size[0], self.image_size[1], self.image_size[2]]
        full = center_crop_or_pad_by_bbox(img, bbox, target_shape, pad_value=0)
        bbox_crop = center_crop_or_pad_by_bbox(bbox.astype(np.uint8), bbox, target_shape, pad_value=0) > 0
        label_crop = center_crop_or_pad_by_bbox((label > 0).astype(np.uint8), bbox, target_shape, pad_value=0) > 0

        bbox_only = full.copy()
        bbox_only[~bbox_crop] = 0
        tumor_only = full.copy()
        tumor_only[~label_crop] = 0

        # Channel meaning: [full context, bbox-only, tumor-only]
        img_stack = np.stack([full, bbox_only, tumor_only], axis=0).astype(np.float32)
        img_stack = apply_percentile_cutoff(img_stack, percentile=self.percentile_cutoff)
        return img_stack.astype(np.float32)

    def augment_image(self, img_numpy):
        if self.augment_context not in {'simple', 'full'}:
            raise ValueError(f"augment_context must be 'simple' or 'full'. Got: {self.augment_context}")

        if self.augment_context == 'full':
            original_nonzero_mask = img_numpy != 0

            if random.uniform(0, 1) < self.augment_frequency:
                img_numpy, _ = random_noise_channel_first(img_numpy, std_fraction=0.02)
                img_numpy = restore_zero_background_after_intensity_aug(img_numpy, original_nonzero_mask)
            if random.uniform(0, 1) < self.augment_frequency:
                img_numpy, _ = random_brightness_np(img_numpy)
                img_numpy = restore_zero_background_after_intensity_aug(img_numpy, original_nonzero_mask)
            if random.uniform(0, 1) < self.augment_frequency:
                img_numpy, _ = random_contrast_np(img_numpy)
                img_numpy = restore_zero_background_after_intensity_aug(img_numpy, original_nonzero_mask)
            if random.uniform(0, 1) < self.augment_frequency:
                img_numpy, _ = random_sharpness_np(img_numpy)
                img_numpy = restore_zero_background_after_intensity_aug(img_numpy, original_nonzero_mask)

        if random.uniform(0, 1) < self.augment_frequency:
            img_numpy, _, _ = random_flip_channel_first_3d(img_numpy)
        if random.uniform(0, 1) < self.augment_frequency:
            img_numpy, _ = random_rotate_channel_first_3d(img_numpy, order=1)
        if random.uniform(0, 1) < self.augment_frequency:
            img_numpy, _, _ = random_translate_channel_first_3d(img_numpy, order=1)

        return img_numpy.astype(np.float32)

    def __getitem__(self, index):
        f = self.index_array[index]
        patient_set = self.patient_set_list[f]
        patient_index = self.patient_index_list[f]

        input_filename = self.x_file_list[f]
        label_filename = os.path.join(self.data_root, patient_set, patient_index, 'label.nii.gz')
        bbox_filename = os.path.join(self.data_root, patient_set, patient_index, 'bbox_mask.nii.gz')

        if input_filename != self.current_input_file:
            img = self.load_file(input_filename, label_filename, bbox_filename)
            y = self.y_list[f]

            self.current_input_file = input_filename
            self.current_input_data = img
            self.current_y = y

        img_numpy = np.copy(self.current_input_data) 

        if self.augment:
            img_numpy = self.augment_image(img_numpy)

        # Normalize after augmentation, as requested.
        if self.normalize_factor == 'medicalnet':
            img_numpy = medicalnet_zscore(img_numpy)
        elif self.normalize_factor == 'equation':
            img_numpy = Data_processing.normalize_image(
                img_numpy,
                normalize_factor='equation',
                image_max=np.max(img_numpy),
                image_min=np.min(img_numpy),
                invert=False,
            )
        else:
            img_numpy = Data_processing.normalize_image(img_numpy, normalize_factor=self.normalize_factor)

        # Dataset item shape: [C, X, Y, Z] = [3, target_x, target_y, target_z]
        input_data = torch.from_numpy(img_numpy).float()
        output_data = torch.tensor(self.current_y).long()
        return input_data, output_data

    def on_epoch_end(self):
        self.index_array = self.generate_index_array()
        self.current_input_file = None
        self.current_input_data = None
        self.current_y = None
