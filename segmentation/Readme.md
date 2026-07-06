# Segment ascending aorta using nnUNetRecEncUNet
**Author: Zhennong Chen, Xi'an Jiaotong-Liverpool University, 2026**

## step 0: install nnUNet
do it in the terminal as the root user of docker  <br />
git clone https://github.com/MIC-DKFZ/nnUNet.git  <br />
cd nnUNet  <br />
pip install -e .  <br />
可能会改变numpy，torch等依赖的版本，需要注意

## step 1: in the terminal as the root user of docker, type the following paths
export nnUNet_raw="/host/e/D/Data/Habitats/External/nnUNet_raw"  <br />
export nnUNet_preprocessed="/host/e/D/Data/Habitats/External/nnUNet_preprocessed"  <br />
export nnUNet_results="/host/d/projects/Habitats/segmentation/models"  <br />
export nnUNet_compile=0 <br />

## step 2: preprocess the data using image_preprocessing.ipynb

## step 3: prepare the nnUNet dataset using prepare_nnunet_data.ipynb

## step 4. plan and preprocess data for nnUNet experiments
- in the terminal, type:  <br />
   nnUNetv2_plan_and_preprocess -d 603 -c 3d_fullres -pl nnUNetPlannerResEncM -np 1
- in the generated text file (nnUNetResEncUNetMPlans.txt), change the batch_size to 1 if GPU memory is limited.  <br />


## step 5. train 
- in the terminal, type:
   nnUNetv2_train 504 3d_fullres fold -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_onlyMirror01_DA5  <br />
- fold can be 0,1,2,3,4, empirically, each fold generates similar results  <br />
- if fine-tuned from the model trained on public dataset, add the argument: -pretrained_weights /host/d/projects/aorta_seg/models/Dataset503_AortaProcessed/checkpoint_final_fold0.pth(replace with your own path)   <br />
- if need to continue training (for example you accidentally interrupt the trianing), Add --c   <br />


## step 6. predict 
- in the model folder (nUNet_results), manually change the checkpoint_best.pth to checkpoint_final.pth if there is no checkpoint_final.pth  <br />
- make folders where you are going to save the prediction results, e.g.,  <br />
  /host/d/projects/aorta_seg/models/Dataset504_AortaTAA/results/EncUNetM_3d_fullres/predicts_raw/fold_0
- in the terminal, type:  <br />
nnUNetv2_predict_from_modelfolder -i /host/e/D/Data/Habitats/External/nnUNet_raw/Dataset603_TumorExternal/imagesTs -o /host/d/projects/Habitats/segmentation/models/Dataset603_TumorExternal/results/EncUNetM_3d_fullres/predicts_raw/fold_0 -m /host/d/projects/Habitats/segmentation/models/Dataset602_Tumor/nnUNetTrainer_onlyMirror01_DA5__nnUNetResEncUNetMPlans__3d_fullres -f 0

## step 7. post-processing and quantitative analysis using post_processing.ipynb




