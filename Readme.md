# Segment Osteosarcoma using nnUNetRecEncUNet
**Author: Zhennong Chen, Xi'an Jiaotong-Liverpool University, 2026**

## environment:
pip uninstall -y \
  torch torchvision torchaudio triton cuda-toolkit cuda-bindings cuda-pathfinder \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver \
  nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink \
  nvidia-nvshmem-cu13 nvidia-nvtx

pip cache purge

# install PyTorch 2.8.0 + CUDA 12.9 
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129



## step 0: install nnUNet
do it in the terminal as the root user of docker  <br />
git clone https://github.com/MIC-DKFZ/nnUNet.git  <br />
cd nnUNet  <br />
pip install -e .  <br />

## step 1: in the terminal as the root user of docker, type the following paths
export nnUNet_raw="/host/d/Data/Habitats/Jishuitan/nnUNet_raw"  <br />
export nnUNet_preprocessed="/host/d/Data/Habitats/Jishuitan/nnUNet_preprocessed"  <br />
export nnUNet_results="/host/d/projects/Habitats/segmentation/models"  <br />
export nnUNet_compile=0 <br />

## step 2: preprocess the data using image_preprocessing.ipynb

## step 3: prepare the nnUNet dataset using prepare_nnunet_data.ipynb
- For the first 81 cases, i call it Dataset601_TumorFirstSet  <br />

## step 4. plan and preprocess data for nnUNet experiments
- in the terminal, type:  <br />
   nnUNetv2_plan_and_preprocess -d 601 -c 3d_fullres -pl nnUNetPlannerResEncM -np 1
- in the generated text file (nnUNetResEncUNetMPlans.txt), change the batch_size to 1 if GPU memory is limited.  <br />


## step 5. train 
- in the terminal, type:
   nnUNetv2_train 601 3d_fullres fold -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_onlyMirror01_DA5  <br />
- fold can be 0,1,2,3,4, empirically, each fold generates similar results  <br />
- if fine-tuned from the model trained on public dataset, add the argument: -pretrained_weights /host/d/projects/aorta_seg/models/Dataset503_AortaProcessed/checkpoint_final_fold0.pth(replace with your own path)   <br />
- if need to continue training (for example you accidentally interrupt the trianing), Add --c   <br />


## step 6. predict 
- in the model folder (nUNet_results), manually change the checkpoint_best.pth to checkpoint_final.pth if there is no checkpoint_final.pth  <br />
- make folders where you are going to save the prediction results, e.g.,  <br />
  /host/d/projects/Habitats/segmentation/models/Dataset601_TumorFirstSet/results/predicts_raw/fold_0
- in the terminal, type:  <br />
nnUNetv2_predict_from_modelfolder -i /host/d/Data/Habitats/Jishuitan/nnUNet_raw/Dataset601_TumorFirstSet/imagesTs -o /host/d/projects/Habitats/segmentation/models/Dataset601_TumorFirstSet/results/predicts_raw/fold_0 -m /host/d/projects/Habitats/segmentation/models/Dataset601_TumorFirstSet/nnUNetTrainer_onlyMirror01_DA5__nnUNetResEncUNetMPlans__3d_fullres -f 0

## step 7. post-processing and quantitative analysis using post_processing.ipynb




