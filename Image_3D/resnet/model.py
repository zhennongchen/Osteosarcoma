
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score

import Osteosarcoma.functions_collection as ff


# ============================================================
# MedicalNet-style 3D ResNet architecture
# ============================================================
# This file extracts the 3D ResNet backbone idea from MedicalNet, but adapts
# the network for classification instead of segmentation. The original
# MedicalNet model ends with conv_seg for voxel-wise segmentation; here we keep
# conv1/bn/layer1-4 compatible with MedicalNet pretrained weights and replace
# the head with adaptive average pooling + fc.


def exists(x):
    return x is not None


def divisible_by(numer, denom):
    return (numer % denom) == 0


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def downsample_basic_block(x, planes, stride, no_cuda=True):
    # Kept for compatibility with MedicalNet shortcut type A.
    # Use device/dtype-aware zeros instead of the old Variable/.data pattern.
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        out.size(4),
        dtype=out.dtype,
        device=out.device,
    )
    out = torch.cat([out, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class MedicalNetResNet3D(nn.Module):
    """
    MedicalNet-compatible 3D ResNet for binary classification.

    Input:
        [B, 1, D, H, W] or project convention [B, 1, X, Y, Z].
        The convolution itself is agnostic to anatomical axis naming.

    Backbone:
        Same layer naming as MedicalNet (`conv1`, `bn1`, `layer1`-`layer4`),
        so pretrained weights load directly after stripping `module.`.

    Head:
        AdaptiveAvgPool3d(1) + Linear for classification.
    """

    def __init__(
        self,
        block,
        layers,
        num_classes=2,
        shortcut_type='B',
        no_cuda=True,
        in_channels=1,
    ):
        super().__init__()
        self.inplanes = 64
        self.no_cuda = no_cuda
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # Match MedicalNet: layer3/layer4 use dilation rather than further stride.
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x


def resnet10(num_classes=2, shortcut_type='B', in_channels=1, **kwargs):
    return MedicalNetResNet3D(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, shortcut_type=shortcut_type, in_channels=in_channels)


def resnet18(num_classes=2, shortcut_type='A', in_channels=1, **kwargs):
    return MedicalNetResNet3D(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, shortcut_type=shortcut_type, in_channels=in_channels)


def resnet34(num_classes=2, shortcut_type='A', in_channels=1, **kwargs):
    return MedicalNetResNet3D(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, shortcut_type=shortcut_type, in_channels=in_channels)


def resnet50(num_classes=2, shortcut_type='B', in_channels=1, **kwargs):
    return MedicalNetResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, shortcut_type=shortcut_type, in_channels=in_channels)


def resnet101(num_classes=2, shortcut_type='B', in_channels=1, **kwargs):
    return MedicalNetResNet3D(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, shortcut_type=shortcut_type, in_channels=in_channels)


def resnet152(num_classes=2, shortcut_type='B', in_channels=1, **kwargs):
    return MedicalNetResNet3D(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, shortcut_type=shortcut_type, in_channels=in_channels)


def resnet200(num_classes=2, shortcut_type='B', in_channels=1, **kwargs):
    return MedicalNetResNet3D(Bottleneck, [3, 24, 36, 3], num_classes=num_classes, shortcut_type=shortcut_type, in_channels=in_channels)


def build_resnet3d_model(model_depth=50, num_classes=2, in_channels=1):
    if model_depth == 10:
        return resnet10(num_classes=num_classes, in_channels=in_channels, shortcut_type='B')
    if model_depth == 18:
        return resnet18(num_classes=num_classes, in_channels=in_channels, shortcut_type='A')
    if model_depth == 34:
        return resnet34(num_classes=num_classes, in_channels=in_channels, shortcut_type='A')
    if model_depth == 50:
        return resnet50(num_classes=num_classes, in_channels=in_channels, shortcut_type='B')
    if model_depth == 101:
        return resnet101(num_classes=num_classes, in_channels=in_channels, shortcut_type='B')
    if model_depth == 152:
        return resnet152(num_classes=num_classes, in_channels=in_channels, shortcut_type='B')
    if model_depth == 200:
        return resnet200(num_classes=num_classes, in_channels=in_channels, shortcut_type='B')
    raise ValueError(f'Unsupported model_depth: {model_depth}')


def load_medicalnet_pretrained(model, pretrain_path, verbose=True):
    """
    Load MedicalNet pretrained backbone weights into the classifier.

    MedicalNet checkpoints are segmentation models with keys like
    `module.conv1.weight`. Our classifier has matching backbone names but a
    different classification head. We therefore load only keys that exist and
    have the same shape, skipping `fc` and any segmentation head keys.
    """
    if pretrain_path is None:
        return model
    if not os.path.isfile(pretrain_path):
        raise FileNotFoundError(pretrain_path)

    checkpoint = torch.load(pretrain_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model_state = model.state_dict()

    matched = {}
    skipped = []
    adapted = []
    for key, value in state_dict.items():
        clean_key = key[len('module.'):] if key.startswith('module.') else key
        if clean_key in model_state and tuple(model_state[clean_key].shape) == tuple(value.shape):
            matched[clean_key] = value
        elif clean_key == 'conv1.weight' and clean_key in model_state:
            target_shape = tuple(model_state[clean_key].shape)
            source_shape = tuple(value.shape)
            # MedicalNet conv1 is usually [64,1,7,7,7]. For 3-channel input,
            # copy the single-channel filters to all input channels and divide
            # by channel count to keep activation scale similar.
            if len(source_shape) == 5 and len(target_shape) == 5 and source_shape[1] == 1 and target_shape[1] > 1 and source_shape[0] == target_shape[0] and source_shape[2:] == target_shape[2:]:
                adapted_weight = value.repeat(1, target_shape[1], 1, 1, 1) / float(target_shape[1])
                matched[clean_key] = adapted_weight
                adapted.append((clean_key, source_shape, target_shape))
            else:
                skipped.append(clean_key)
        else:
            skipped.append(clean_key)

    model_state.update(matched)
    model.load_state_dict(model_state)

    if verbose:
        print(f'Loaded MedicalNet pretrained weights from: {pretrain_path}')
        print(f'  matched keys: {len(matched)}')
        print(f'  skipped keys: {len(skipped)}')
        if len(adapted) > 0:
            print('  adapted keys:', adapted)
        if len(skipped) > 0:
            print('  first skipped keys:', skipped[:10])
    return model



def configure_fine_tuning(model, fine_tune_stage='all'):
    """
    Freeze/unfreeze a MedicalNet-style 3D ResNet for small-sample fine-tuning.

    fine_tune_stage options:
        all: train the whole network
        fc:  train only the final fully connected layer
        1:   train layer4's last residual block plus fc
        2:   train layer4's last two residual blocks plus fc

    This mirrors the Image_2D ResNet fine-tuning semantics.
    """
    fine_tune_stage = str(fine_tune_stage).lower()
    valid_stages = {'all', 'fc', '1', '2'}
    if fine_tune_stage not in valid_stages:
        raise ValueError(f"fine_tune_stage must be one of {sorted(valid_stages)}. Got: {fine_tune_stage}")

    for param in model.parameters():
        param.requires_grad = False

    if fine_tune_stage == 'all':
        modules_to_train = [model]
    elif fine_tune_stage == 'fc':
        modules_to_train = [model.fc]
    else:
        num_last_blocks = int(fine_tune_stage)
        if num_last_blocks > len(model.layer4):
            raise ValueError(
                f"Requested last {num_last_blocks} blocks from layer4, "
                f"but layer4 only has {len(model.layer4)} blocks."
            )
        modules_to_train = list(model.layer4[-num_last_blocks:]) + [model.fc]

    for module in modules_to_train:
        for param in module.parameters():
            param.requires_grad = True

    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    print(f"Fine-tune stage: {fine_tune_stage}")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.2%})")
    return model


def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_total_parameters(model):
    return sum(param.numel() for param in model.parameters())


def trainable_parameter_groups(model):
    params = [param for param in model.parameters() if param.requires_grad]
    if len(params) == 0:
        raise RuntimeError('No trainable parameters found. Check fine_tune_stage.')
    return params


# ============================================================
# Trainer
# ============================================================


class Trainer(object):
    def __init__(
        self,
        model,
        generator_train,
        generator_val,
        train_batch_size,
        *,
        accum_iter=1,
        train_num_steps=100,
        results_folder=None,
        train_lr=1e-3,
        train_lr_decay_every=50,
        optimizer='sgd',
        train_momentum=0.9,
        train_weight_decay=1e-4,
        save_models_every=1,
        validation_every=1,
        ema_update_every=10,
        ema_decay=0.95,
        adam_betas=(0.9, 0.99),
        amp=False,
        mixed_precision_type='fp16',
        split_batches=True,
        max_grad_norm=1.,
        num_workers=0,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no',
        )

        self.model = model
        self.loss_function = nn.CrossEntropyLoss()

        self.batch_size = train_batch_size
        self.accum_iter = accum_iter
        self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm
        self.train_lr_decay_every = train_lr_decay_every
        self.save_model_every = save_models_every
        self.validation_every = validation_every

        self.ds = generator_train
        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=num_workers,
        )
        self.dl = self.accelerator.prepare(dl)

        self.ds_val = generator_val
        dl_val = DataLoader(
            self.ds_val,
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=num_workers,
        )
        self.dl_val = self.accelerator.prepare(dl_val)

        optimizer = str(optimizer).lower()
        if optimizer == 'adam':
            self.opt = Adam(trainable_parameter_groups(model), lr=train_lr, betas=adam_betas)
            scheduler_gamma = 0.95
        elif optimizer == 'sgd':
            self.opt = SGD(
                trainable_parameter_groups(model),
                lr=train_lr,
                momentum=train_momentum,
                weight_decay=train_weight_decay,
                nesterov=True,
            )
            scheduler_gamma = 0.9
        else:
            raise ValueError(f"optimizer must be 'adam' or 'sgd'. Got: {optimizer}")

        self.optimizer_name = optimizer
        self.scheduler = StepLR(self.opt, step_size=1, gamma=scheduler_gamma)

        self.results_folder = results_folder
        self.model_folder = os.path.join(self.results_folder, 'models')
        self.log_folder = os.path.join(self.results_folder, 'log')
        ff.make_folder([self.results_folder, self.model_folder, self.log_folder])

        if self.accelerator.is_main_process:
            self.ema = EMA(model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.step = 0
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def set_frozen_batchnorm_eval(self):
        # Frozen BatchNorm layers should not update running statistics.
        # This is especially important for small 3D medical datasets.
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                params = list(module.parameters(recurse=False))
                if len(params) == 0 or all(not param.requires_grad for param in params):
                    module.eval()

    def _snapshot_batchnorm_state(self):
        """Snapshot BN running stats before train-mode evaluation.

        Diagnostic AUC/loss evaluation can intentionally run with
        `model.train(True)` to use batch statistics. BatchNorm updates running
        mean/var in train mode even under no_grad(), so restore these buffers
        afterward to avoid contaminating the actual training state.
        """
        state = []
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                state.append((
                    module,
                    None if module.running_mean is None else module.running_mean.detach().clone(),
                    None if module.running_var is None else module.running_var.detach().clone(),
                    None if module.num_batches_tracked is None else module.num_batches_tracked.detach().clone(),
                ))
        return state

    def _restore_batchnorm_state(self, state):
        for module, running_mean, running_var, num_batches_tracked in state:
            if running_mean is not None and module.running_mean is not None:
                module.running_mean.copy_(running_mean)
            if running_var is not None and module.running_var is not None:
                module.running_var.copy_(running_var)
            if num_batches_tracked is not None and module.num_batches_tracked is not None:
                module.num_batches_tracked.copy_(num_batches_tracked)

    def save(self, step_num):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict() if self.accelerator.is_main_process else None,
            'decay_steps': self.scheduler.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        torch.save(data, os.path.join(self.model_folder, 'model-' + str(step_num) + '.pt'))

    def load_model(self, trained_model_filename):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(trained_model_filename, map_location=device)
        model = self.accelerator.unwrap_model(self.model)

        if isinstance(data, dict) and 'model' in data:
            model_state = data['model']
            self.step = int(data.get('step', self.step))
            if 'opt' in data:
                self.opt.load_state_dict(data['opt'])
            if self.accelerator.is_main_process and data.get('ema') is not None:
                self.ema.load_state_dict(data['ema'])
            if 'decay_steps' in data:
                self.scheduler.load_state_dict(data['decay_steps'])
            if exists(self.accelerator.scaler) and exists(data.get('scaler')):
                self.accelerator.scaler.load_state_dict(data['scaler'])
        else:
            model_state = data

        cleaned_state = {}
        for key, value in model_state.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            cleaned_state[key] = value
        model.load_state_dict(cleaned_state, strict=True)

    def _safe_auc(self, labels, probs):
        labels = np.asarray(labels).astype(int)
        probs = np.asarray(probs).astype(float)
        if len(np.unique(labels)) < 2:
            return np.nan
        return float(roc_auc_score(labels, probs))

    def _classification_probability(self, logits):
        return torch.softmax(logits, dim=1)[:, 1]

    def evaluate_loader(self, dataloader, name='dataset'):
        """Evaluate loss/AUC with augmentation disabled and model.eval()."""
        accelerator = self.accelerator
        device = accelerator.device

        dataset = getattr(dataloader, 'dataset', None)
        old_augment = None
        if dataset is not None and hasattr(dataset, 'augment'):
            old_augment = dataset.augment
            dataset.augment = False

        self.model.eval()

        losses = []
        all_probs = []
        all_labels = []

        try:
            with torch.no_grad():
                for batch in dataloader:
                    batch_input, batch_gt = batch
                    data_input = batch_input.to(device)
                    data_gt = batch_gt.to(device).long()
                    with accelerator.autocast():
                        logits = self.model(data_input)
                        loss = self.loss_function(logits, data_gt)
                        probs = self._classification_probability(logits)
                    gathered_probs = accelerator.gather_for_metrics(probs.detach())
                    gathered_labels = accelerator.gather_for_metrics(data_gt.detach())
                    gathered_loss = accelerator.gather_for_metrics(loss.detach().reshape(1))
                    all_probs.extend(gathered_probs.cpu().numpy().tolist())
                    all_labels.extend(gathered_labels.cpu().numpy().tolist())
                    losses.extend(gathered_loss.cpu().numpy().tolist())
        finally:
            if old_augment is not None:
                dataset.augment = old_augment
            self.model.train(True)
            self.set_frozen_batchnorm_eval()

        mean_loss = float(np.mean(losses)) if len(losses) > 0 else np.nan
        auc = self._safe_auc(all_labels, all_probs)
        print(f'{name} loss: {mean_loss:.4f}')
        print(f'{name} AUC : {auc:.4f}' if not np.isnan(auc) else f'{name} AUC : nan')
        return mean_loss, auc, np.asarray(all_labels), np.asarray(all_probs)

    def train(self, pre_trained_model=None, start_step=None):
        accelerator = self.accelerator

        if pre_trained_model is not None:
            self.load_model(pre_trained_model)
            print('model loaded from ', pre_trained_model)
        else:
            print('no project checkpoint provided; using MedicalNet-pretrained initialization')

        if start_step is not None:
            self.step = start_step

        training_log = []
        train_eval_loss = np.nan
        val_loss = np.nan
        train_auc = np.nan
        val_auc = np.nan

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                print('training epoch: ', self.step + 1)
                print('optimizer: ', self.optimizer_name)
                print('learning rate: ', self.scheduler.get_last_lr()[0])

                average_loss = []
                self.model.train(True)
                self.set_frozen_batchnorm_eval()
                self.opt.zero_grad()

                for batch_idx, batch in enumerate(self.dl, start=1):
                    batch_input, batch_gt = batch
                    data_input = batch_input.to(self.device)
                    data_gt = batch_gt.to(self.device).long()

                    with self.accelerator.autocast():
                        logits = self.model(data_input)
                        loss = self.loss_function(logits, data_gt)

                    loss_to_backward = loss / self.accum_iter
                    self.accelerator.backward(loss_to_backward)

                    is_update_step = batch_idx % self.accum_iter == 0 or batch_idx == len(self.dl)
                    if is_update_step:
                        accelerator.wait_for_everyone()
                        accelerator.clip_grad_norm_(trainable_parameter_groups(self.model), self.max_grad_norm)
                        self.opt.step()
                        self.opt.zero_grad()
                        if self.accelerator.is_main_process:
                            self.ema.update()
                    average_loss.append(float(loss.item()))

                average_loss = float(np.mean(average_loss)) if len(average_loss) > 0 else np.nan
                pbar.set_description(f'average loss: {average_loss:.4f}')
                accelerator.wait_for_everyone()
                self.step += 1

                if self.step != 0 and divisible_by(self.step, self.save_model_every):
                    self.save(self.step)
                if self.step != 0 and divisible_by(self.step, self.train_lr_decay_every):
                    self.scheduler.step()
                if self.step != 0 and divisible_by(self.step, self.validation_every):
                    print('validation at step: ', self.step)
                    train_eval_loss, train_auc, _, _ = self.evaluate_loader(self.dl, name='train')
                    val_loss, val_auc, _, _ = self.evaluate_loader(self.dl_val, name='validation')

                training_log.append([
                    self.step,
                    self.scheduler.get_last_lr()[0],
                    average_loss,
                    train_eval_loss,
                    val_loss,
                    train_auc,
                    val_auc,
                ])
                df = pd.DataFrame(
                    training_log,
                    columns=[
                        'iteration',
                        'learning_rate',
                        'training_loss',
                        'train_eval_loss',
                        'validation_loss',
                        'train_auc',
                        'validation_auc',
                    ],
                )
                ff.make_folder([self.log_folder])
                df.to_excel(os.path.join(self.log_folder, 'training_log.xlsx'), index=False)

                self.ds.on_epoch_end()
                self.ds_val.on_epoch_end()
                pbar.update(1)

        accelerator.print('training complete')
