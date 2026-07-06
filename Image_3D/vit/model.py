import math
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score

import Osteosarcoma.functions_collection as ff



def exists(x):
    return x is not None


def divisible_by(numer, denom):
    return (numer % denom) == 0


def _as_3tuple(value):
    """Allow image_size / patch_size to be passed as an int or a 3-value tuple."""
    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError("Expected a 3-value tuple.")
        return value
    return (value, value, value)


class PatchEmbedding3D(nn.Module):
    """
    Convert a 3D MRI volume into a sequence of patch tokens.

    Input shape:
        x: [batch, channels, x, y, z]

    Current project setting:
        image_size = (128, 128, 160)
        patch_size = (16, 16, 8)

    This produces:
        patch grid = (8, 8, 20)
        number of tokens = 8 * 8 * 20 = 1280

    Implementation note:
        We use Conv3d with kernel_size=stride=patch_size. This is equivalent
        to cutting non-overlapping 3D patches, flattening each patch, and applying
        the same linear projection to every patch. It is much cleaner and faster
        than manually slicing/flattening patches in Python.
    """

    def __init__(
        self,
        *,
        image_size=(128, 128, 160),
        patch_size=(16, 16, 8),
        in_channels=1,
        embed_dim=256,
    ):
        super().__init__()

        self.image_size = _as_3tuple(image_size)
        self.patch_size = _as_3tuple(patch_size)

        for img_dim, patch_dim in zip(self.image_size, self.patch_size):
            if img_dim % patch_dim != 0:
                raise ValueError(
                    f"image_size {self.image_size} must be divisible by "
                    f"patch_size {self.patch_size}."
                )

        self.grid_size = tuple(
            img_dim // patch_dim
            for img_dim, patch_dim in zip(self.image_size, self.patch_size)
        )
        self.num_patches = math.prod(self.grid_size)

        # Each output voxel of this Conv3d corresponds to one 3D patch token.
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(
                "PatchEmbedding3D expects input shape [batch, channel, x, y, z]. "
                f"Got shape {tuple(x.shape)}."
            )

        expected_shape = self.image_size
        actual_shape = tuple(x.shape[-3:])
        if actual_shape != expected_shape:
            raise ValueError(
                f"Expected spatial shape {expected_shape}, got {actual_shape}. "
                "Please crop/pad the generator output before feeding ViT."
            )

        # [B, C, X, Y, Z] -> [B, embed_dim, grid_x, grid_y, grid_z]
        x = self.proj(x)

        # [B, embed_dim, grid_x, grid_y, grid_z] -> [B, num_patches, embed_dim]
        # Now every row in dimension 1 is one token.
        x = x.flatten(2).transpose(1, 2)
        return x


class MLPHead(nn.Module):
    """
    Small classification head.

    The Transformer produces one embedding vector for each patch. Since we are
    not using a CLS token in this first version, the ViT model will mean-pool
    all patch tokens into one volume-level vector, then this head maps that
    vector to class logits.
    """

    def __init__(self, embed_dim, num_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """
    One ViT Transformer encoder block.

    This mirrors the block shown in the paper's Supplementary Figure S2:

        LayerNorm
        -> Multi-Head Self-Attention
        -> Dropout
        -> residual connection
        -> LayerNorm
        -> MLP / feed-forward network
        -> Dropout
        -> residual connection

    The input and output shapes are the same:
        [batch, num_patches, embed_dim]

    Because the shape is preserved, these blocks can be stacked sequentially.
    In the plaque ViT paper, the Transformer encoder is repeated n=6 times:
    block 1 feeds block 2, block 2 feeds block 3, and so on.
    """

    def __init__(
        self,
        *,
        embed_dim=256,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        attention_dropout=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention lets every patch token compare itself with every
        # other patch token. This is the key difference from local convolution.
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=False,
        )
        x = x + self.drop1(attn_output)

        # The MLP updates each token after attention has mixed information
        # across tokens. The residual connection keeps optimization stable.
        x = x + self.mlp(self.norm2(x))
        return x


class ViT3D(nn.Module):
    """
    First-pass 3D Vision Transformer for osteosarcoma MRI classification.

    This follows the broad idea from the plaque habitat + ViT paper:
        3D ROI/bbox volume -> 3D patches -> linear projection -> position
        embedding -> Transformer encoder -> MLP classifier.

    Project-specific choices for v1:
        - input image size: 128 x 128 x 160
        - patch size: 16 x 16 x 8
        - no CLS token
        - no invalid/zero-token masking
        - simple mean pooling over all patch tokens

    Output:
        logits with shape [batch, num_classes].

    For binary classification with CrossEntropyLoss:
        num_classes=2 and labels should be LongTensor with values 0/1.
    """

    def __init__(
        self,
        *,
        image_size=(128, 128, 160),
        patch_size=(16, 16, 8),
        in_channels=1,
        num_classes=2,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        attention_dropout=0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding3D(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.image_size = self.patch_embed.image_size
        self.patch_size = self.patch_embed.patch_size
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.depth = depth

        # Positional embedding tells the model where each patch came from.
        # Without it, token 1 and token 100 would be treated as an unordered set.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # The plaque paper uses n=6 Transformer encoder blocks. We keep the
        # same default depth, and store the blocks explicitly so the sequential
        # structure is easy to read.
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        self.head = MLPHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        # Convert the 3D image into patch tokens.
        x = self.patch_embed(x)

        # Add learned positional information to each token.
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Six sequential Transformer blocks when depth=6:
        # block 1 output -> block 2 input -> ... -> block 6 output.
        for block in self.transformer_blocks:
            x = block(x)

        # No CLS token in this version: use average token embedding as the
        # whole-volume representation.
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.head(features)
        return logits

class Trainer(object):
    """
    Trainer for 3D ViT binary classification.

    This follows the training style of Example_UNet.model.Trainer:
        - Accelerator for device / mixed precision handling
        - EMA model tracking
        - gradient accumulation through accum_iter
        - periodic model saving
        - periodic validation
        - Excel training log

    Classification-specific changes compared with the U-Net example:
        - The model output is [batch, 2] class logits, not a segmentation image.
        - The loss is CrossEntropyLoss, not image reconstruction loss.
        - During validation epochs, we run prediction on all train and val cases,
          convert logits to class-1 probability, and compute AUC.
    """

    def __init__(
        self,
        model,
        generator_train,
        generator_val,
        train_batch_size,

        *,
        accum_iter=10,
        train_num_steps=100,
        results_folder=None,
        train_lr=1e-4,
        train_lr_decay_every=100,
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

        # CLASSIFICATION CHANGE:
        # Generator returns labels as LongTensor with values 0/1.
        # ViT returns logits with shape [batch, 2]. CrossEntropyLoss matches
        # this exactly and internally applies log-softmax.
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
            pin_memory=True,
            num_workers=num_workers,
        )
        self.dl = self.accelerator.prepare(dl)

        self.ds_val = generator_val
        dl_val = DataLoader(
            self.ds_val,
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        self.dl_val = self.accelerator.prepare(dl_val)

        self.opt = Adam(model.parameters(), lr=train_lr, betas=adam_betas)
        self.scheduler = StepLR(self.opt, step_size=1, gamma=0.95)

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

    def save(self, stepNum):
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

        torch.save(data, os.path.join(self.model_folder, 'model-' + str(stepNum) + '.pt'))

    def load_model(self, trained_model_filename):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(trained_model_filename, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process and data.get('ema') is not None:
            self.ema.load_state_dict(data['ema'])

        self.scheduler.load_state_dict(data['decay_steps'])

        if exists(self.accelerator.scaler) and exists(data.get('scaler')):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def _safe_auc(self, labels, probs):
        labels = np.asarray(labels).astype(int)
        probs = np.asarray(probs).astype(float)
        if len(np.unique(labels)) < 2:
            return np.nan
        return float(roc_auc_score(labels, probs))

    def _classification_probability(self, logits):
        # CLASSIFICATION CHANGE:
        # logits are raw scores [batch, 2]. Softmax turns them into class
        # probabilities. We use column 1 as the probability of positive class.
        return torch.softmax(logits, dim=1)[:, 1]

    def evaluate_loader(self, dataloader, name='dataset'):
        """
        Run full-dataset prediction for one dataloader.

        Returns mean loss, AUC, all labels, and all class-1 probabilities.
        This is intentionally called only during validation epochs because it
        loops over every train/val case and can be expensive for 3D ViT.
        """
        accelerator = self.accelerator
        device = accelerator.device

        # During training, the train generator may have augment=True. For AUC,
        # we want deterministic case-level predictions, like the ML pipeline.
        # So we temporarily turn augmentation off during evaluation, then restore
        # the generator to its previous state.
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

                    # gather_for_metrics keeps this compatible with Accelerator.
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
                print('learning rate: ', self.scheduler.get_last_lr()[0])

                average_loss = []
                self.model.train(True)
                self.opt.zero_grad()

                for batch_idx, batch in enumerate(self.dl, start=1):
                    batch_input, batch_gt = batch
                    data_input = batch_input.to(device)
                    data_gt = batch_gt.to(device).long()

                    with self.accelerator.autocast():
                        logits = self.model(data_input)
                        loss = self.loss_function(logits, data_gt) 

                    # CLASSIFICATION CHANGE:
                    # This is standard gradient accumulation. Backward happens
                    # on every mini-batch, optimizer step happens every accum_iter
                    # mini-batches or at the last mini-batch of the epoch.
                    loss_to_backward = loss / self.accum_iter
                    self.accelerator.backward(loss_to_backward)

                    is_update_step = (
                        batch_idx % self.accum_iter == 0
                        or batch_idx == len(self.dl)
                    )
                    if is_update_step:
                        accelerator.wait_for_everyone()
                        accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
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

                # Expensive full prediction pass: do this only at validation epochs.
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

