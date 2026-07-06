# dataset classes

import os
import numpy as np
import nibabel as nb
import random
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import Osteosarcoma.Data_processing as Data_processing


# ============================================================
# Shared crop / augmentation helpers
# ============================================================

def center_crop_or_pad_by_bbox(arr, bbox_mask, target_shape, pad_value=0):
    """Crop/pad arr to target_shape around the bbox center.

    arr and bbox_mask must have the same spatial shape. Voxels outside the
    original image are padded with pad_value. This keeps original pixels inside
    the crop and avoids resizing/interpolation.
    """
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
    """Clip all channels using percentile from bbox-only channel (>0 voxels)."""
    stacked_img = stacked_img.astype(np.float32, copy=True)
    bbox_channel = stacked_img[1]
    valid = bbox_channel[bbox_channel > 0]
    if valid.size == 0:
        return stacked_img
    cutoff = float(np.percentile(valid, percentile))
    stacked_img[stacked_img > cutoff] = cutoff
    return stacked_img



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
    """Light unsharp-mask style sharpness for channel-first 2D/2.5D data.

    Expected shape is [C,H,W] or [C,H,W,S]. For 2.5D, the same operation
    settings are used for all slices, preserving spatial correspondence.
    """
    if amount is None:
        amount = np.random.uniform(0.1, 0.4)
    out = np.empty_like(i, dtype=np.float32)
    if i.ndim == 3:
        for c in range(i.shape[0]):
            blur = ndimage.gaussian_filter(i[c], sigma=sigma)
            out[c] = i[c] + amount * (i[c] - blur)
    elif i.ndim == 4:
        for c in range(i.shape[0]):
            for s in range(i.shape[3]):
                blur = ndimage.gaussian_filter(i[c, :, :, s], sigma=sigma)
                out[c, :, :, s] = i[c, :, :, s] + amount * (i[c, :, :, s] - blur)
    else:
        raise ValueError(f'Expected [C,H,W] or [C,H,W,S]. Got shape: {i.shape}')
    return out.astype(np.float32), amount


def random_flip_channel_first_2d(i):
    flip_x = random.choice([False, True])
    flip_y = random.choice([False, True])
    if flip_x:
        i = np.flip(i, axis=1)
    if flip_y:
        i = np.flip(i, axis=2)
    return np.ascontiguousarray(i), flip_x, flip_y


def random_rotate_channel_first_2d(i, degree=None, degree_range=(-10, 10), fill_val=None, order=1):
    if degree is None:
        degree = random.uniform(degree_range[0], degree_range[1])
    if fill_val is None:
        fill_val = float(np.min(i))
    if degree == 0:
        return i, degree
    # Rotate in the x-y plane. For [C,H,W,S], all slices/channels share the
    # same rotation angle and interpolation rule.
    out = ndimage.rotate(i, degree, axes=(1, 2), reshape=False, order=order, mode='constant', cval=fill_val)
    return out.astype(np.float32), degree


def random_translate_channel_first_2d(i, x_translate=None, y_translate=None, translate_range=(-10, 10), fill_val=None, order=1):
    if x_translate is None or y_translate is None:
        x_translate = int(random.uniform(translate_range[0], translate_range[1]))
        y_translate = int(random.uniform(translate_range[0], translate_range[1]))
    if fill_val is None:
        fill_val = float(np.min(i))
    if i.ndim == 3:
        shift = (0, x_translate, y_translate)
    elif i.ndim == 4:
        shift = (0, x_translate, y_translate, 0)
    else:
        raise ValueError(f'Expected [C,H,W] or [C,H,W,S]. Got shape: {i.shape}')
    out = ndimage.shift(i, shift=shift, order=order, mode='constant', cval=fill_val)
    return out.astype(np.float32), x_translate, y_translate


class Dataset_2D(Dataset):
    def __init__(
        self,
        patient_set_list,
        patient_index_list,
        x_file_list,
        y_list,
        data_root,
        target_image_size,
        normalize_factor,
        only_tumor_pixels='roi',
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
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f'Expected img_slices with shape [X,Y,3]. Got: {img.shape}')
        if label_filename is None or bbox_filename is None:
            raise ValueError('2D generator requires label_slices and bbox_mask_slices filenames.')

        label = nb.load(label_filename).get_fdata()
        bbox = nb.load(bbox_filename).get_fdata() > 0
        if label.shape != img.shape or bbox.shape != img.shape:
            raise ValueError(f'Shape mismatch: img={img.shape}, label={label.shape}, bbox={bbox.shape}')

        target_x, target_y = self.image_size
        target_shape = (target_x, target_y)
        slice_stacks = []

        # Build [full, bbox-only, tumor-only] for each of the three consecutive
        # slices. The output is [semantic_channel, H, W, slice].
        for z in range(3):
            img2 = img[:, :, z]
            label2 = label[:, :, z] > 0
            bbox2 = bbox[:, :, z]

            if not np.any(bbox2):
                # Fallback to the middle-slice bbox if a neighboring slice has
                # no bbox pixels after preprocessing.
                bbox2 = bbox[:, :, 1]
            if not np.any(bbox2):
                raise RuntimeError(f'Empty bbox mask for all 2D slices: {bbox_filename}')

            full = center_crop_or_pad_by_bbox(img2, bbox2, target_shape, pad_value=0)
            bbox_crop = center_crop_or_pad_by_bbox(bbox2.astype(np.uint8), bbox2, target_shape, pad_value=0) > 0
            label_crop = center_crop_or_pad_by_bbox(label2.astype(np.uint8), bbox2, target_shape, pad_value=0) > 0

            bbox_only = full.copy()
            bbox_only[~bbox_crop] = 0
            tumor_only = full.copy()
            tumor_only[~label_crop] = 0

            slice_stacks.append(np.stack([full, bbox_only, tumor_only], axis=0).astype(np.float32))

        img_stack = np.stack(slice_stacks, axis=-1).astype(np.float32)
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
            img_numpy, _, _ = random_flip_channel_first_2d(img_numpy)
        if random.uniform(0, 1) < self.augment_frequency:
            img_numpy, _ = random_rotate_channel_first_2d(img_numpy, order=1)
        if random.uniform(0, 1) < self.augment_frequency:
            img_numpy, _, _ = random_translate_channel_first_2d(img_numpy, order=1)

        return img_numpy.astype(np.float32)

    def __getitem__(self, index):
        f = self.index_array[index]
        patient_set = self.patient_set_list[f]
        patient_index = self.patient_index_list[f]

        input_filename = self.x_file_list[f]
        label_filename = os.path.join(self.data_root, patient_set, patient_index, 'label_slices.nii.gz')
        bbox_filename = os.path.join(self.data_root, patient_set, patient_index, 'bbox_mask_slices.nii.gz')

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
        img_numpy = Data_processing.normalize_image(
            img_numpy,
            normalize_factor=self.normalize_factor,
            image_max=np.max(img_numpy),
            image_min=np.min(img_numpy),
            invert=False,
            final_max=1,
            final_min=0,
        ).astype(np.float32)

        # The channels are not RGB, but ImageNet-pretrained ResNet expects
        # this scale. Apply the same channel-wise normalization to every slice.
        imagenet_mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1, 1)
        imagenet_std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1, 1)
        img_numpy = (img_numpy - imagenet_mean) / imagenet_std

        slice_front = torch.from_numpy(np.ascontiguousarray(img_numpy[:, :, :, 0])).float()
        slice_middle = torch.from_numpy(np.ascontiguousarray(img_numpy[:, :, :, 1])).float()
        slice_rear = torch.from_numpy(np.ascontiguousarray(img_numpy[:, :, :, 2])).float()
        output_data = torch.tensor(self.current_y).long()
        return slice_front, slice_middle, slice_rear, output_data

    def on_epoch_end(self):
        self.index_array = self.generate_index_array()
        self.current_input_file = None
        self.current_input_data = None
        self.current_y = None
