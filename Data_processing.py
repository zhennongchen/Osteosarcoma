import numpy as np
import nibabel as nb
import os
from skimage.measure import block_reduce
from scipy import ndimage
from dipy.align.reslice import reslice
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
    
    
    
    