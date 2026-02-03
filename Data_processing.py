import numpy as np
import nibabel as nb
import os
from skimage.measure import block_reduce
from scipy import ndimage
from dipy.align.reslice import reslice
import pandas as pd
import Osteosarcoma.functions_collection as ff

def crop_or_pad(array, target, value):
    # Pad each axis to at least the target.
    margin = target - np.array(array.shape)
    padding = [(0, max(x, 0)) for x in margin]
    array = np.pad(array, padding, mode="constant", constant_values=value)
    for i, x in enumerate(margin):
        array = np.roll(array, shift=+(x // 2), axis=i)

    if type(target) == int:
        target = [target] * array.ndim

    ind = tuple([slice(0, t) for t in target])
    return array[ind]

def correct_shift_caused_in_pad_crop_loop(img):
    # if an image goes from [a,b,c] --> pad --> [A,B,c] --> crop --> [a,b,c], when a,b is even, it goes back to original image, but when a,b is odd, it need to shift by 1 pixel in x and y
    if img.shape[0] % 2 == 1:

        img = np.roll(img, shift = 1, axis = 0)
        img = np.roll(img, shift = 1, axis = 1)
    else:
        img = np.copy(img)
    return img


def adapt(x, cutoff = False,add_noise = False, sigma = 5, normalize = True, expand_dim = True):
    x = np.load(x, allow_pickle = True)
    
    if cutoff == True:
        x = cutoff_intensity(x, -1000)
    
    if add_noise == True:
        ValueError('WRONG NOISE ADDITION CODE')
        x =  x + np.random.normal(0, sigma, x.shape) 

    if normalize == True:
        x = normalize_image(x)
    
    if expand_dim == True:
        x = np.expand_dims(x, axis = -1)
    # print('after adapt, shape of x is: ', x.shape)
    return x


def normalize_image(x, normalize_factor = 1000, image_max = 100, image_min = -100, final_max = 1, final_min = -1 , invert = False):
    # a common normalization method in CT
    # if you use (x-mu)/std, you need to preset the mu and std
    if invert == False:
        if isinstance(normalize_factor, int): # direct division
            return x.astype(np.float32) / normalize_factor
        else: # normalize_factor == 'equation'
            return (final_max - final_min) / (image_max - image_min) * (x.astype(np.float32) - image_min) + (final_min)
    else:
        if isinstance(normalize_factor, int): # direct division
            return x * normalize_factor
        else: # normalize_factor == 'equation'
            return (x - final_min) * (image_max - image_min) / (final_max - final_min) + image_min


def cutoff_intensity(x,cutoff_low = None, cutoff_high = None):
    xx = np.copy(x)

    if cutoff_low is not None and np.min(x) < cutoff_low:
        xx[x <= cutoff_low] = cutoff_low
    
    if cutoff_high is not None and np.max(x) > cutoff_high:
        xx[x >= cutoff_high] = cutoff_high
    return xx

# function: translate image
def translate_image(image, shift):
    assert len(shift) in [2, 3], "Shift must be a list of 2 elements for 2D or 3 elements for 3D"
    assert len(image.shape) in [2, 3], "Image must be either 2D or 3D"
    assert len(image.shape) == len(shift), "Shift dimensions must match image dimensions"

    fill_val = np.min(image)  # Fill value is the minimum value in the image
    translated_image = np.full_like(image, fill_val)  # Create an image filled with fill_val

    if image.ndim == 2:  # 2D image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                new_i = i - shift[0]
                new_j = j - shift[1]
                if 0 <= new_i < image.shape[0] and 0 <= new_j < image.shape[1]:
                    translated_image[new_i, new_j] = image[i, j]
    elif image.ndim == 3:  # 3D image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    new_i = i - shift[0]
                    new_j = j - shift[1]
                    new_k = k - shift[2]
                    if 0 <= new_i < image.shape[0] and 0 <= new_j < image.shape[1] and 0 <= new_k < image.shape[2]:
                        translated_image[new_i, new_j, new_k] = image[i, j, k]
    else:
        raise ValueError("Image dimensions not supported")

    return translated_image


# function: rotate image
def rotate_image(image, degrees, order, fill_val = None):

    if fill_val is None:
        fill_val = np.min(image)
        
    if image.ndim == 2:  # 2D image
        assert isinstance(degrees, (int, float)), "Degrees should be a single number for 2D rotation"
        rotated_img = ndimage.rotate(image, degrees, reshape=False, mode='constant', cval=fill_val, order = order)

    elif image.ndim == 3:  # 3D image
        assert len(degrees) == 3 and all(isinstance(deg, (int, float)) for deg in degrees), "Degrees should be a list of three numbers for 3D rotation"
        # Rotate around x-axis
        rotated_img = ndimage.rotate(image, degrees[0], axes=(1, 2), reshape=False, mode='constant', cval=fill_val, order  = order)
        # Rotate around y-axis
        rotated_img = ndimage.rotate(rotated_img, degrees[1], axes=(0, 2), reshape=False, mode='constant', cval=fill_val, order = order)
        # Rotate around z-axis
        rotated_img = ndimage.rotate(rotated_img, degrees[2], axes=(0, 1), reshape=False, mode='constant', cval=fill_val, order = order)
    else:
        raise ValueError("Image must be either 2D or 3D")

    return rotated_img


def resample_nifti(nifti, 
                   order,
                   mode, #'nearest' or 'constant' or 'reflect' or 'wrap'    
                   cval,
                   in_plane_resolution_mm=1.25,
                   slice_thickness_mm=None,
                   number_of_slices=None):
    
    # sometimes dicom to nifti programs don't define affine correctly.
    resolution = np.array(nifti.header.get_zooms()[:3] + (1,))
    if (np.abs(nifti.affine)==np.identity(4)).all():
        nifti.set_sform(nifti.affine*resolution)


    data   = nifti.get_fdata().copy()
    shape  = nifti.shape[:3]
    affine = nifti.affine.copy()
    zooms  = nifti.header.get_zooms()[:3] 

    if number_of_slices is not None:
        new_zooms = (in_plane_resolution_mm,
                     in_plane_resolution_mm,
                     (zooms[2] * shape[2]) / number_of_slices)
    elif slice_thickness_mm is not None:
        new_zooms = (in_plane_resolution_mm,
                     in_plane_resolution_mm,
                     slice_thickness_mm)            
    else:
        new_zooms = (in_plane_resolution_mm,
                     in_plane_resolution_mm,
                     zooms[2])

    new_zooms = np.array(new_zooms)
    for i, (n_i, res_i, res_new_i) in enumerate(zip(shape, zooms, new_zooms)):
        n_new_i = (n_i * res_i) / res_new_i
        # to avoid rounding ambiguities
        if (n_new_i  % 1) == 0.5: 
            new_zooms[i] -= 0.001

    data_resampled, affine_resampled = reslice(data, affine, zooms, new_zooms, order=order, mode=mode , cval = cval)
    nifti_resampled = nb.Nifti1Image(data_resampled, affine_resampled)

    x=nifti_resampled.header.get_zooms()[:3]
    y=new_zooms
    if not np.allclose(x,y, rtol=1e-02):
        print('not all close: ', x,y)

    return nifti_resampled       
    
# function: get 3D bbox from label
def bbox3d(label_arr, buffer_x=5, buffer_y=5, buffer_z=5):
    tumor = (label_arr == 1)

    if not np.any(tumor):
        raise ValueError(f"No tumor voxels found in label: {label_path}")

    # ---------- 2) find tight bbox ----------
    xx, yy, zz = np.where(tumor)

    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()
    z_min, z_max = zz.min(), zz.max()

    # ---------- 3) apply buffer and clip ----------
    X, Y, Z = label_arr.shape

    x0 = max(x_min - buffer_x, 0)
    x1 = min(x_max + buffer_x, X - 1)

    y0 = max(y_min - buffer_y, 0)
    y1 = min(y_max + buffer_y, Y - 1)

    z0 = max(z_min - buffer_z, 0)
    z1 = min(z_max + buffer_z, Z - 1)

    print("Tight bbox (x,y,z):",
        (x_min, x_max), (y_min, y_max), (z_min, z_max))
    print("Buffered bbox:",
        f"x[{x0}:{x1}], y[{y0}:{y1}], z[{z0}:{z1}]")

    # ---------- 4) build bbox mask ----------
    bbox_arr = np.zeros_like(label_arr, dtype=np.uint8)
    bbox_arr[x0:x1+1, y0:y1+1, z0:z1+1] = 1

    return bbox_arr, x0, x1, y0, y1, z0, z1



def patchify(label_arr, label_nii, x0, x1, y0, y1, z0, z1, patch_size_mm, min_tumor_fraction, out_dir, patch_table_path):
    tumor = (label_arr == 1)

    # spacing from header (in mm)
    # nibabel: header['pixdim'] = [?, dx, dy, dz, ...]
    pixdim = label_nii.header.get_zooms()[:3]  # (dx,dy,dz)
    dx, dy, dz = float(pixdim[0]), float(pixdim[1]), float(pixdim[2])

    print("Spacing (mm):", dx, dy, dz)

    # ---------------- compute patch size in voxels ----------------
    nx = int(np.ceil(patch_size_mm / dx))
    ny = int(np.ceil(patch_size_mm / dy))
    nz = int(np.ceil(patch_size_mm / dz))

    # 强制至少 1 voxel
    nx = max(nx, 1)
    ny = max(ny, 1)
    nz = max(nz, 1)

    print("Patch size (vox):", nx, ny, nz)

    X, Y, Z = label_arr.shape

    # ---------------- determine grid ranges (allow extend beyond bbox if not divisible) ----------------
    bbox_len_x = (x1 - x0 + 1)
    bbox_len_y = (y1 - y0 + 1)
    bbox_len_z = (z1 - z0 + 1)

    npx = int(np.ceil(bbox_len_x / nx))
    npy = int(np.ceil(bbox_len_y / ny))
    npz = int(np.ceil(bbox_len_z / nz))

    # grid_end may extend outside bbox; later clip to image boundary
    grid_x_end = x0 + npx * nx - 1
    grid_y_end = y0 + npy * ny - 1
    grid_z_end = z0 + npz * nz - 1

    # clip to image bounds
    grid_x_end = min(grid_x_end, X - 1)
    grid_y_end = min(grid_y_end, Y - 1)
    grid_z_end = min(grid_z_end, Z - 1)

    print("Grid start:", (x0, y0, z0))
    print("Grid end  :", (grid_x_end, grid_y_end, grid_z_end))
    print("Grid n patches:", (npx, npy, npz))

    total_patches = npx * npy * npz

    # ---------------- iterate patches ----------------
    rows = []
    patch_id = 0

    count_included = 0
    for iz in range(npz):
        for iy in range(npy):
            for ix in range(npx):
                # patch bounds in voxel index (inclusive)
                px0 = x0 + ix * nx
                py0 = y0 + iy * ny
                pz0 = z0 + iz * nz

                px1 = px0 + nx - 1
                py1 = py0 + ny - 1
                pz1 = pz0 + nz - 1

                # clip to image bounds (important when extended)
                px0_c = max(px0, 0); py0_c = max(py0, 0); pz0_c = max(pz0, 0)
                px1_c = min(px1, X-1); py1_c = min(py1, Y-1); pz1_c = min(pz1, Z-1)

                # if patch completely outside image (rare), skip
                if (px0_c > px1_c) or (py0_c > py1_c) or (pz0_c > pz1_c):
                    continue

                # tumor fraction inside THIS patch (using tumor voxels only)
                patch_tumor = tumor[px0_c:px1_c+1, py0_c:py1_c+1, pz0_c:pz1_c+1]
                patch_voxels = patch_tumor.size
                tumor_voxels = int(patch_tumor.sum())
                tumor_fraction = tumor_voxels / float(patch_voxels)

                if tumor_fraction < min_tumor_fraction:
                    continue
                count_included += 1

                # ---------------- build and save patch mask: (patch region) AND (tumor) ----------------
                patch_mask = np.zeros_like(tumor, dtype=np.uint8)
                patch_mask[px0_c:px1_c+1, py0_c:py1_c+1, pz0_c:pz1_c+1] = patch_tumor.astype(np.uint8)

                patch_mask_nii = nb.Nifti1Image(patch_mask, affine=label_nii.affine, header=label_nii.header)
                patch_mask_path = os.path.join(out_dir, f"patch_{patch_id:04d}.nii.gz")
                nb.save(patch_mask_nii, patch_mask_path)

                # ---------------- record 8 vertices (voxel coords) ----------------
                # 顶点定义（按 voxel index，inclusive）：
                # z0 面： (x0,y0,z0) (x1,y0,z0) (x1,y1,z0) (x0,y1,z0)
                # z1 面： (x0,y0,z1) (x1,y0,z1) (x1,y1,z1) (x0,y1,z1)
                v = [
                    (px0_c, py0_c, pz0_c),
                    (px1_c, py0_c, pz0_c),
                    (px1_c, py1_c, pz0_c),
                    (px0_c, py1_c, pz0_c),
                    (px0_c, py0_c, pz1_c),
                    (px1_c, py0_c, pz1_c),
                    (px1_c, py1_c, pz1_c),
                    (px0_c, py1_c, pz1_c),
                ]

                row = {
                    "patch_id": patch_id,
                    "tumor_fraction": tumor_fraction,
                    "mask_path": patch_mask_path,  # 可选：方便后续 radiomics batch
                }
                for i, (vx, vy, vz) in enumerate(v, start=1):
                    row[f"x_{i}"] = int(vx)
                    row[f"y_{i}"] = int(vy)
                    row[f"z_{i}"] = int(vz)

                rows.append(row)
                patch_id += 1

    # print("Kept patches:", patch_id)

    # ---------------- save table ----------------
    df_patches = pd.DataFrame(rows)
    df_patches.to_excel(patch_table_path, index=False)
    # print("Saved patch table:", patch_table_path)

    return count_included, total_patches
