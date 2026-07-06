import sys
sys.path.append('/host/d/Github/')

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from ema_pytorch import EMA
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

import Osteosarcoma.Build_lists.Build_list as Build_list
import Osteosarcoma.functions_collection as ff


def exists(x):
    return x is not None


def divisible_by(numer, denom):
    return (numer % denom) == 0

def default_imagenet_weights(weights_enum):
    # Torchvision weight enum names differ by model/version.
    # DEFAULT is the safest choice when available; older versions may only have V1.
    if hasattr(weights_enum, 'DEFAULT'):
        return weights_enum.DEFAULT
    if hasattr(weights_enum, 'IMAGENET1K_V2'):
        return weights_enum.IMAGENET1K_V2
    if hasattr(weights_enum, 'IMAGENET1K_V1'):
        return weights_enum.IMAGENET1K_V1
    raise AttributeError(f'No usable ImageNet weights found for {weights_enum}')


def build_resnet_model(model_depth=50, num_classes=2, use_imagenet=True, dropout_p=0.3):
    """
    Build a 2D ResNet classifier.

    If use_imagenet=True, initialize the backbone from ImageNet weights and
    replace the final fully connected layer for binary classification.
    If use_imagenet=False, the whole model is randomly initialized. This is
    used when loading a full project checkpoint, because the checkpoint already
    contains all trained weights.
    """
    if use_imagenet:
        if model_depth == 18:
            model = resnet18(weights=default_imagenet_weights(ResNet18_Weights))
        elif model_depth == 34:
            model = resnet34(weights=default_imagenet_weights(ResNet34_Weights))
        elif model_depth == 50:
            model = resnet50(weights=default_imagenet_weights(ResNet50_Weights))
        else:
            raise ValueError(f"Unsupported model depth: {model_depth}")
    else:
        if model_depth == 18:
            model = resnet18(weights=None)
        elif model_depth == 34:
            model = resnet34(weights=None)
        elif model_depth == 50:
            model = resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported model depth: {model_depth}")

    model.fc = nn.Sequential(
        nn.Dropout(p=float(dropout_p)),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model



class ShallowCNN2D(nn.Module):
    """Small CNN baseline for limited-size medical imaging datasets.

    The model is intentionally much smaller than ResNet/EfficientNet. It uses
    repeated Conv-BN-ReLU-MaxPool blocks, adaptive average pooling, then a
    dropout-regularized classifier. The final pooled feature vector has
    init_channel * 8 dimensions.
    """

    def __init__(self, in_channels=3, init_channel=8, num_classes=2, dropout_p=0.3):
        super().__init__()
        init_channel = int(init_channel)
        if init_channel <= 0:
            raise ValueError(f'init_channel must be positive. Got: {init_channel}')

        channels = [init_channel, init_channel * 2, init_channel * 4, init_channel * 8]
        self.features = nn.Sequential(
            self._make_block(in_channels, channels[0]),
            self._make_block(channels[0], channels[1]),
            self._make_block(channels[1], channels[2]),
            self._make_block(channels[2], channels[3]),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=float(dropout_p)),
            nn.Linear(channels[-1], num_classes),
        )

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_shallow_cnn_model(init_channel=8, num_classes=2, in_channels=3, dropout_p=0.3):
    return ShallowCNN2D(
        in_channels=in_channels,
        init_channel=init_channel,
        num_classes=num_classes,
        dropout_p=dropout_p,
    )


def build_efficientnet_model(efficientnet_version='b0', num_classes=2, use_imagenet=True, dropout_p=0.3):
    """Build a 2D EfficientNet classifier from torchvision.

    EfficientNet-B0 is the default small pretrained model. The generator does
    not need to change because the current 2D/2.5D tensors are already
    three-channel and ImageNet-normalized. EfficientNet uses adaptive pooling,
    so our 144 x 144 input size is acceptable even though ImageNet pretraining
    used larger natural images.
    """
    efficientnet_version = str(efficientnet_version).lower()
    model_table = {
        'b0': (efficientnet_b0, EfficientNet_B0_Weights),
        'b1': (efficientnet_b1, EfficientNet_B1_Weights),
        'b2': (efficientnet_b2, EfficientNet_B2_Weights),
        'b3': (efficientnet_b3, EfficientNet_B3_Weights),
        'v2_s': (efficientnet_v2_s, EfficientNet_V2_S_Weights),
    }
    if efficientnet_version not in model_table:
        raise ValueError(
            f"Unsupported efficientnet_version: {efficientnet_version}. "
            f"Choose from {sorted(model_table)}."
        )

    constructor, weights_enum = model_table[efficientnet_version]
    weights = default_imagenet_weights(weights_enum) if use_imagenet else None
    model = constructor(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=float(dropout_p)),
        nn.Linear(in_features, num_classes),
    )
    return model


def _configure_resnet_fine_tuning(model, fine_tune_stage):
    """
    Freeze/unfreeze a torchvision ResNet for small-sample fine-tuning.

    fine_tune_stage options:
        all: train the whole network
        fc:  train only the final fully connected layer
        1:   train layer4's last residual block plus fc
        2:   train layer4's last two residual blocks plus fc

    This intentionally uses residual-block counts instead of stage numbers.
    In ResNet, a full layer4 stage contains many parameters; for example,
    training all of layer4 in ResNet18 opens roughly 75% of the model. The
    '1' and '2' modes are therefore safer small-sample fine-tuning choices.
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


def _configure_efficientnet_fine_tuning(model, fine_tune_stage):
    """Freeze/unfreeze EfficientNet blocks for small-sample fine-tuning.

    fine_tune_stage options mirror the ResNet code:
        all: train the whole network
        fc:  train only the classifier
        1:   train the last feature block plus classifier
        2:   train the last two feature blocks plus classifier
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
        modules_to_train = [model.classifier]
    else:
        num_last_blocks = int(fine_tune_stage)
        if num_last_blocks > len(model.features):
            raise ValueError(
                f"Requested last {num_last_blocks} EfficientNet feature blocks, "
                f"but features only has {len(model.features)} blocks."
            )
        modules_to_train = list(model.features[-num_last_blocks:]) + [model.classifier]

    for module in modules_to_train:
        for param in module.parameters():
            param.requires_grad = True

    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    print(f"Fine-tune stage: {fine_tune_stage}")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.2%})")
    return model



def _configure_shallow_cnn_fine_tuning(model, fine_tune_stage):
    """Freeze/unfreeze the shallow CNN.

    all: train the whole network
    fc:  train only the classifier
    1:   train the last feature block plus classifier
    2:   train the last two feature blocks plus classifier
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
        modules_to_train = [model.classifier]
    else:
        num_last_blocks = int(fine_tune_stage)
        if num_last_blocks > len(model.features):
            raise ValueError(
                f"Requested last {num_last_blocks} shallow CNN blocks, "
                f"but features only has {len(model.features)} blocks."
            )
        modules_to_train = list(model.features[-num_last_blocks:]) + [model.classifier]

    for module in modules_to_train:
        for param in module.parameters():
            param.requires_grad = True

    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    print(f"Fine-tune stage: {fine_tune_stage}")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.2%})")
    return model


def configure_fine_tuning(model, fine_tune_stage='all'):
    if hasattr(model, 'layer4') and hasattr(model, 'fc'):
        return _configure_resnet_fine_tuning(model, fine_tune_stage=fine_tune_stage)
    if isinstance(model, ShallowCNN2D):
        return _configure_shallow_cnn_fine_tuning(model, fine_tune_stage=fine_tune_stage)
    if hasattr(model, 'features') and hasattr(model, 'classifier'):
        return _configure_efficientnet_fine_tuning(model, fine_tune_stage=fine_tune_stage)
    raise ValueError('Unsupported model type for fine-tuning configuration.')


def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_total_parameters(model):
    return sum(param.numel() for param in model.parameters())


def trainable_parameter_groups(model):
    params = [param for param in model.parameters() if param.requires_grad]
    if len(params) == 0:
        raise RuntimeError('No trainable parameters found. Check fine_tune_stage.')
    return params


class Trainer(object):
    """
    Trainer for 2D ResNet binary classification.

    This mirrors the current Image_3D/vit Trainer style:
        - Accelerator for device and optional mixed precision
        - EMA model tracking
        - gradient accumulation
        - periodic model saving
        - periodic full train/validation evaluation
        - Excel log with train loss, validation loss, train AUC, validation AUC
    """

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
        input_mode='2d',
        train_momentum=0.9,
        train_weight_decay=5e-4,
        label_smoothing=0.05,
        use_class_weight=True,
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

        class_weight = None
        if use_class_weight:
            y_arr = np.asarray(generator_train.y_list).astype(int)
            class_count = np.bincount(y_arr, minlength=2).astype(float)
            # Balanced class weights: N / (num_classes * count_c).
            # This keeps the binary CrossEntropyLoss compatible with [B,2] logits.
            class_weight_np = len(y_arr) / (2.0 * np.maximum(class_count, 1.0))
            class_weight = torch.tensor(class_weight_np, dtype=torch.float32, device=self.device)
            print('class counts:', class_count.astype(int).tolist())
            print('class weights:', class_weight_np.tolist())
        self.label_smoothing = float(label_smoothing)
        self.use_class_weight = bool(use_class_weight)
        self.loss_function = nn.CrossEntropyLoss(
            weight=class_weight,
            label_smoothing=self.label_smoothing,
        )

        self.batch_size = train_batch_size
        self.accum_iter = accum_iter
        self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm
        self.train_lr_decay_every = train_lr_decay_every
        self.save_model_every = save_models_every
        self.validation_every = validation_every
        self.input_mode = str(input_mode).lower()
        if self.input_mode not in {'2d', '2.5d'}:
            raise ValueError(f"input_mode must be '2d' or '2.5d'. Got: {input_mode}")

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
            self.opt = Adam(
                trainable_parameter_groups(model),
                lr=train_lr,
                betas=adam_betas,
                weight_decay=train_weight_decay,
            )
            scheduler_gamma = 0.95
        elif optimizer == 'adamw':
            self.opt = AdamW(
                trainable_parameter_groups(model),
                lr=train_lr,
                betas=adam_betas,
                weight_decay=train_weight_decay,
            )
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
            raise ValueError(f"optimizer must be 'adam', 'adamw', or 'sgd'. Got: {optimizer}")

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
        # Frozen BatchNorm layers should not keep updating running statistics.
        # This matters in small medical datasets because BN running stats can
        # become another quiet source of train-set overfitting.
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

        model_state_keys = model.state_dict().keys()
        # Backward compatibility: older ResNet checkpoints saved a plain
        # Linear head as fc.weight/fc.bias. New models use Dropout+Linear,
        # so the learnable Linear is fc.1.weight/fc.1.bias.
        if 'fc.weight' in cleaned_state and 'fc.1.weight' in model_state_keys:
            cleaned_state['fc.1.weight'] = cleaned_state.pop('fc.weight')
        if 'fc.bias' in cleaned_state and 'fc.1.bias' in model_state_keys:
            cleaned_state['fc.1.bias'] = cleaned_state.pop('fc.bias')

        model.load_state_dict(cleaned_state, strict=True)

    def _safe_auc(self, labels, probs):
        labels = np.asarray(labels).astype(int)
        probs = np.asarray(probs).astype(float)
        if len(np.unique(labels)) < 2:
            return np.nan
        return float(roc_auc_score(labels, probs))

    def _classification_probability(self, logits):
        return torch.softmax(logits, dim=1)[:, 1]

    def _forward_batch(self, batch, device):
        """Return logits and labels for either 2D or 2.5D mode.

        Dataset_2D returns: slice_front, slice_middle, slice_rear, label.
        2D mode uses only the middle slice. 2.5D mode runs the same ResNet on
        all three slices and averages logits before loss/probability.
        """
        slice_front, slice_middle, slice_rear, batch_gt = batch
        data_gt = batch_gt.to(device).long()

        if self.input_mode == '2d':
            data_input = slice_middle.to(device)
            logits = self.model(data_input)
        elif self.input_mode == '2.5d':
            logits_front = self.model(slice_front.to(device))
            logits_middle = self.model(slice_middle.to(device))
            logits_rear = self.model(slice_rear.to(device))
            logits = (logits_front + logits_middle + logits_rear) / 3.0
        else:
            raise ValueError(f'Unexpected input_mode: {self.input_mode}')

        return logits, data_gt

    def evaluate_loader(self, dataloader, name='dataset'):
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
                    with accelerator.autocast():
                        logits, data_gt = self._forward_batch(batch, device)
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
        device = accelerator.device

        if pre_trained_model is not None:
            self.load_model(pre_trained_model)
            print('model loaded from ', pre_trained_model)
        else:
            print('no project checkpoint provided; training from ImageNet-initialized model')

        if start_step is not None:
            self.step = start_step

        training_log = []
        val_loss = np.nan
        train_eval_loss = np.nan
        train_auc = np.nan
        val_auc = np.nan

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                print('training epoch: ', self.step + 1)
                # print('optimizer: ', self.optimizer_name)
                # print('input mode: ', self.input_mode)
                # print('label smoothing: ', self.label_smoothing)
                # print('use class weight: ', self.use_class_weight)
                print('learning rate: ', self.scheduler.get_last_lr()[0])

                average_loss = []
                self.model.train(True)
                self.set_frozen_batchnorm_eval()
                self.opt.zero_grad()

                for batch_idx, batch in enumerate(self.dl, start=1):
                    with self.accelerator.autocast():
                        logits, data_gt = self._forward_batch(batch, device)
                        loss = self.loss_function(logits, data_gt)

                    loss_to_backward = loss / self.accum_iter
                    self.accelerator.backward(loss_to_backward)

                    is_update_step = (
                        batch_idx % self.accum_iter == 0
                        or batch_idx == len(self.dl)
                    )
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