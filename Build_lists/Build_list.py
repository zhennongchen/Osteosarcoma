from email.mime import image
import numpy as np
import os
import pandas as pd


class Build():  
    def __init__(self,file_list):
        self.file_list = file_list
        self.data = pd.read_excel(file_list, dtype = {'Patient_index': str})
        # exclude the ones with column "Include" not equal to "Yes"
        if 'Include' in self.data.columns:
            self.data = self.data[self.data['Include'] == 'Yes'].reset_index(drop=True)

    def __build__(self,batch_list = None, index_list = None):
        # assert batch_list and index list are not both None or both not None
        assert (batch_list is None) != (index_list is None), "Either batch_list or index_list must be provided, but not both."

        if index_list is None:
            for b in range(len(batch_list)):
                cases = self.data.loc[self.data['batch'] == batch_list[b]]
                if b == 0:
                    c = cases.copy()
                else:
                    c = pd.concat([c,cases])
        else: # index is just the index of row
            c = self.data.loc[index_list]

        batch_list = np.asarray(c['batch']) if 'batch' in c.columns else None
        patient_index_list = np.asarray(c['Patient_index']) if 'Patient_index' in c.columns else None
        label_list = np.asarray(c['Pathologic_Response_Necrosis_gt90pct']) if 'Pathologic_Response_Necrosis_gt90pct' in c.columns else None
        image_path_list = np.asarray(c['Image_filepath']) if 'Image_filepath' in c.columns else None
        mask_path_list = np.asarray(c['Mask_filepath']) if 'Mask_filepath' in c.columns else None

        return batch_list, patient_index_list, label_list, image_path_list, mask_path_list
      