# Deep Learning Paper Notes For Habitat Project

Primary local papers:

```text
/host/d/projects/Habitats/papers/Integration of multi-scale radiomics and deep learning for Ki-67 prediction in clear cell renal carcinoma.pdf
/host/d/projects/Habitats/papers/Integration of multi-scale radiomics and deep learning for Ki-67 prediction in clear cell renal carcinoma_supplements.pdf
/host/d/projects/Habitats/papers/MRI-based habitat radiomics combined with vision transformer for identifying vulnerable intracranial atherosclerotic plaques and predicting stroke events a multicenter, retrospective study.pdf
/host/d/projects/Habitats/papers/MRI-based habitat radiomics combined with vision transformer for identifying vulnerable intracranial atherosclerotic plaques and predicting stroke events a multicenter, retrospective study.docx
```

## Ki-67 ccRCC Paper: ResNet-Based DL Feature Extraction

Relevant text: main paper section `DL model and feature extraction`; supplement `Table S5`.

The study used transfer learning with two ResNet50-based DL feature extractors:

```text
2D-ResNet50
3D-ResNet50
```

### Inputs

2D model:

```text
largest cross-sectional CT slice of the tumor
roi_size = [224, 224]
```

The 2D input is tumor-focused, not the full image. The `224 x 224` size matches the standard ImageNet/ResNet input convention.

3D model:

```text
minimum bounding box enclosing the entire tumor
roi_size = [96, 96, 96]
```

The 3D input is a tumor bounding box, not a full CT volume. The paper does not explain why `96 x 96 x 96` was chosen. Treat it as an engineering tradeoff: large enough to represent tumor volume, smaller than 128³/160³ to control memory and overfitting.

### Pretrained Weights

Main text states:

```text
2D-ResNet50: weights pre-trained on ImageNet
3D-ResNet50: weights pre-trained on the Med3D dataset
```

Important nuance: the Med3D phrase appears in the method paragraph, but the paper does not provide a clear checkpoint link or detailed citation in the extracted text. When explaining this, distinguish article text from implementation knowledge.

Engineering interpretation:

- 2D ImageNet weights are available directly in torchvision, e.g. `ResNet50_Weights.IMAGENET1K_V2`.
- 3D Med3D-style weights are usually obtained from Med3D/MedicalNet-style public 3D ResNet checkpoints, or implemented by loading a compatible 3D ResNet checkpoint if available.

### Training Hyperparameters From Table S5

```text
2DResNet50:
  normalize_method = Mean Subtraction
  model_name = resnet50
  gpus = GPU 0
  batch_size = 32
  epochs = 50
  init_lr = 0.001
  optimizer = SGD
  activation = L2 Regularization + ReLU Activation
  roi_size = [224, 224]
  feature extraction layer = avgpool

3DResNet50:
  normalize_method = Min-Max
  model_name = resnet50
  gpus = GPU 0
  batch_size = 4
  epochs = 100
  init_lr = 0.0003
  optimizer = Adam
  activation = L2 Regularization + ReLU Activation
  roi_size = [96, 96, 96]
  feature extraction layer = avgpool
```

### Data Augmentation

The main text says both DL models used:

```text
random horizontal flips
random vertical flips
random cropping
```

### DL Features And Downstream ML

After training, they extracted output vectors from the final average pooling layer as DL feature representations.

Main-result details:

```text
initial DL features: 4096 for both DL_2D and DL_3D
DL_2D retained features: 12; best classifier: EXT
DL_3D retained features: 10; best classifier: LGBM
```

Interpretation: one ResNet50 avgpool vector is typically 2048-dimensional. The reported 4096 features likely reflect two CT phases concatenated, but treat this as an inference unless the implementation is shown.

The paper found 2D DL outperformed 3D DL. Their discussion attributes this to the larger data/parameter burden of 3D networks and the benefit of ImageNet pretraining for 2D ResNet.

## Plaque Habitat + ViT Paper: Vision Transformer

Relevant text: main paper workflow and `Construction of radiomics model`; supplement `Supplementary Appendix 5. Construction of the ViT model` and `Figure S2`.

### Input

The ViT model uses plaque-focused input, not the full image:

```text
minimum bounding cube for each plaque / plaque-affected vessel
ROI size = 64 x 64 x 48
```

The input is HR-VWI images mapped to a three-dimensional spatial matrix.

### Patch Design

Supplement text states:

```text
ROI size = 64 x 64 x 48
patch size = 16
frame patch size = 2
patch grid = 4 x 4 x 24
```

Beginner interpretation:

1. The 3D input is a box with shape `64 x 64 x 48`.
2. In-plane x/y patch size is `16 x 16`.
3. Since `64 / 16 = 4`, each slice plane has `4 x 4 = 16` in-plane patches.
4. Along z/time/frame direction, `frame patch size = 2` groups every 2 slices.
5. Since `48 / 2 = 24`, there are 24 z/frame groups.
6. Total patch tokens are therefore approximately:

```text
4 x 4 x 24 = 384 tokens
```

Each token corresponds roughly to a `16 x 16 x 2` patch. The patch is flattened and linearly projected to an embedding vector of dimension `D`.

### Network Structure

Main and supplement describe:

```text
Patch Embedding
Position Embedding
6 Transformer blocks
MLP Head
probability output
```

The Transformer uses multi-head self-attention with Q/K/V:

```text
Q = Xe WQ
K = Xe WK
V = Xe WV
Attention = Softmax(Q K^T / sqrt(dk)) V
```

Interpretation: every patch token can attend to every other patch token, allowing the network to learn long-range spatial relationships across plaque components. This differs from CNNs, which emphasize local convolutional neighborhoods.

### Training Hyperparameters And Augmentation

Supplement text says:

```text
GPU = NVIDIA GTX 4060
optimizer = Adam
learning rate = 0.001
epochs = 200
batch size = 64
```

Data augmentation:

```text
random flipping
translation
rotation
```

The paper does not clearly report using pretrained ViT weights. Treat ViT as likely trained from scratch unless a specific checkpoint is later found.

## Guidance For Osteosarcoma Adaptation

### Safer First DL Route: 3D ResNet Feature Extractor

A practical first implementation for Osteosarcoma is closer to the Ki-67 paper:

```text
img.nii.gz + label.nii.gz
crop tumor bounding box
resize/pad to fixed 3D size, e.g. 96 x 96 x 96 or another size selected from bbox statistics
train 3D ResNet for Prognosis_label or Pathologic_label
extract avgpool feature vector
fuse with whole-image radiomics and habitat radiomics
```

Prefer collecting bbox shape statistics from the current cohort before choosing the fixed size. For osteosarcoma, tumors may be much larger than plaque, so plaque-style `64 x 64 x 48` should not be copied blindly.

### ViT Route

A ViT-style route is possible but more sensitive to input size, patch size, sample size, and memory:

```text
crop tumor bounding box
resize/pad to fixed shape compatible with patch size
split into 3D patches
token embedding + positional embedding
Transformer encoder
MLP head probability
fusion with radiomics/habitat probabilities
```

Candidate fixed shapes should be divisible by patch dimensions, for example:

```text
96 x 96 x 32 with patch 16 x 16 x 4
128 x 128 x 32 with patch 16 x 16 x 4
128 x 128 x 64 with patch 16 x 16 x 4 or 16 x 16 x 8
```

Choose based on tumor bbox distribution and GPU memory. Do not assume the plaque paper's `64 x 64 x 48` is appropriate for osteosarcoma.
