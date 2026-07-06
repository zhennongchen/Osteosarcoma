# dataset classes

import os
import numpy as np
import nibabel as nb
import random
import pandas as pd
from scipy import ndimage
from skimage.measure import block_reduce

import torch
from torch.utils.data import Dataset
import Osteosarcoma.Data_processing as Data_processing
import Osteosarcoma.functions_collection as ff


# random function
def random_rotate(i, z_rotate_degree = None, z_rotate_range = [-10,10], fill_val = None, order = 1):
    # only do rotate according to z (in-plane rotation)
    if z_rotate_degree is None:
        z_rotate_degree = random.uniform(z_rotate_range[0], z_rotate_range[1])

    if fill_val is None:
        fill_val = np.min(i)
    
    if z_rotate_degree == 0:
        return i, z_rotate_degree
    else:
        if len(i.shape) == 2:
            return Data_processing.rotate_image(np.copy(i), z_rotate_degree, order = order, fill_val = fill_val, ), z_rotate_degree
        else:
            return Data_processing.rotate_image(np.copy(i), [0,0,z_rotate_degree], order = order, fill_val = fill_val, ), z_rotate_degree

def random_translate(i, x_translate = None,  y_translate = None, translate_range = [-10,10]):
    # only do translate according to x and y
    if x_translate is None or y_translate is None:
        x_translate = int(random.uniform(translate_range[0], translate_range[1]))
        y_translate = int(random.uniform(translate_range[0], translate_range[1]))
    
    if len(i.shape) == 2:
        return Data_processing.translate_image(np.copy(i), [x_translate,y_translate]), x_translate,y_translate
    else:
        return Data_processing.translate_image(np.copy(i), [x_translate,y_translate,0]), x_translate,y_translate


class Dataset_3D(Dataset):
    def __init__(
        self,
        patient_set_list,
        patient_index_list,
        x_file_list,
        y_list,
        data_root,

        target_image_size,
        normalize_factor,
        shuffle = False,
        augment = False,
        augment_frequency = 0,
    ):
        super().__init__()
     
        self.patient_set_list = patient_set_list
        self.patient_index_list = patient_index_list
        self.x_file_list = x_file_list
        self.y_list = y_list
        self.data_root = data_root

        self.image_size = target_image_size
        self.normalize_factor = normalize_factor

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
        index_array = []
        
        if self.shuffle == True:
            f_list = np.random.permutation(self.num_files)
        else:
            f_list = np.arange(self.num_files)

        for f in f_list:
            index_array.append(f)
        
        return index_array

    def __len__(self):
        return self.num_files
    

    def load_file(self, filename, bbox_filename = None):
        ii = nb.load(filename).get_fdata()
        print('original image shape is: ', ii.shape)
        if bbox_filename != None:
            bbox_mask = nb.load(bbox_filename).get_fdata()
            bbox_mask = (bbox_mask > 0).astype(np.int)
            coords = np.where(bbox_mask > 0)
            x_min, x_max = np.min(coords[0]), np.max(coords[0])
            y_min, y_max = np.min(coords[1]), np.max(coords[1])
            z_min, z_max = np.min(coords[2]), np.max(coords[2])
            ii = ii[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        print('bbox cropped image shape is: ', ii.shape)

        # preprocess the image
        ii = Data_processing.normalize_image(ii, normalize_factor = self.normalize_factor, image_max = np.max(ii), image_min = np.min(ii), invert = False)
        ii = Data_processing.crop_or_pad(ii, [self.image_size[0], self.image_size[1], ii.shape[2]], value= np.min(ii))
        return ii
        
    def __getitem__(self, index):
        f = self.index_array[index]
        patient_set = self.patient_set_list[f]
        patient_index = self.patient_index_list[f]
        print('now load patient set: ', patient_set, ' and patient index: ', patient_index)

        input_filename = self.x_file_list[f]
        bbox_filename = os.path.join(self.data_root, patient_set, patient_index, 'bbox_mask.nii.gz')

        if input_filename != self.current_input_file:
            # load input
            img = self.load_file(input_filename,bbox_filename)
            
            y = self.y_list[f]

            self.current_input_file = input_filename
            self.current_input_data = img
            self.current_y = y

        # augmentation
        if self.augment == True:
            if random.uniform(0,1) < self.augment_frequency:
                img, z_rotate_degree = random_rotate(img , order = 1)
                img, translate, y_translate = random_translate(img)

        input_data = torch.from_numpy(img).unsqueeze(0).float()
        # y is a integer, we can directly convert it to a tensor
        output_data = torch.tensor(y).long()

        print('input data shape is: ', input_data.shape, ' and output data is: ', output_data)
        return input_data, output_data
        
    
    def on_epoch_end(self):
        self.index_array = self.generate_index_array()
        self.current_input_file = None
        self.current_input_data = None
        self.current_y = None

    
