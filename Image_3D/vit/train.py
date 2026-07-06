
import sys
sys.path.append('/host/d/Github/')

import argparse
import os

import Osteosarcoma.Build_lists.Build_list as Build_list
import Osteosarcoma.functions_collection as ff
import Osteosarcoma.Image_3D.Generator as Generator
import Osteosarcoma.Image_3D.vit.model as vit_model


def none_or_path(value):
    """Allow command line values like --trained_model_path None."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {'none', 'null', ''}:
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D ViT for Osteosarcoma MRI classification.')

    # Experiment identity.
    parser.add_argument('--label', type=str, default='Prognosis')
    parser.add_argument('--trial_name', type=str, default='vit_3D')
    parser.add_argument('--trained_model_path', type=none_or_path, default=None)
    parser.add_argument('--start_step', type=int, default=0)

    # Patient split settings.
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--val_fold', type=int, default=None)

    # Training settings.
    parser.add_argument('--train_batch_size', type=int, default=10)
    parser.add_argument('--train_num_steps', type=int, default=500)
    parser.add_argument('--save_models_every', type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.val_fold is None:
        raise ValueError('Please set --val_fold. Example: --val_fold 4')

    # Define trial name and output folder.
    label = args.label
    label_type = label + '_label'
    trial_name = args.trial_name
    trial_output_path = os.path.join('/host/d/projects/Habitats/models/', label, 'vit', trial_name)
    setting_name = f'random{args.random_state}_fold{args.val_fold}'
    model_output_path = os.path.join(trial_output_path, setting_name)
    ff.make_folder([
        os.path.dirname(trial_output_path),
        trial_output_path,
        model_output_path,
        os.path.join(model_output_path, 'models'),
        os.path.join(model_output_path, 'log'),
    ])

    trained_model_path = args.trained_model_path
    start_step = args.start_step

    # Fixed first-pass ViT image and patch settings.
    image_size = (80,80,96)
    vit_patch_size = (16, 16, 4)

    train_batch_size = args.train_batch_size
    train_num_steps = args.train_num_steps
    save_models_every = args.save_models_every

    print('============================================================')
    print('3D ViT training settings')
    print('label:', label)
    print('label_type:', label_type)
    print('trial_name:', trial_name)
    print('random_state:', args.random_state)
    print('val_fold:', args.val_fold)
    print('setting_name:', setting_name)
    print('trained_model_path:', trained_model_path)
    print('start_step:', start_step)
    print('train_batch_size:', train_batch_size)
    print('train_num_steps:', train_num_steps)
    print('save_models_every:', save_models_every)
    print('trial_output_path:', trial_output_path)
    print('model_output_path:', model_output_path)

    # Step 1: build patient list.
    patient_list_file = (
        '/host/e/D/Data/Habitats/Jishuitan/Patient_lists/'
        f'image_label_info_set12_5fold_{label.lower()}_random{args.random_state}.xlsx'
    )
    print('patient_list_file:', patient_list_file)

    if not os.path.isfile(patient_list_file):
        raise FileNotFoundError(f'Patient split file not found: {patient_list_file}')

    fold_list = [0, 1, 2, 3, 4, 5]
    if args.val_fold not in fold_list:
        raise ValueError(f'--val_fold must be one of {fold_list}. Got {args.val_fold}.')

    val_fold = args.val_fold
    train_fold = [f for f in fold_list if f != val_fold]

    build = Build_list.Build(patient_list_file)
    fold_list_train, patient_set_list_train, patient_index_list_train, label_list_train, _, _ = build.__build__(
        batch_list=train_fold,
        label_column_name=label_type,
    )
    fold_list_val, patient_set_list_val, patient_index_list_val, label_list_val, _, _ = build.__build__(
        batch_list=[val_fold],
        label_column_name=label_type,
    )

    # Step 2: prepare x_file_list and y_list for training and validation.
    data_root = '/host/e/D/Data/Habitats/Jishuitan/resampled_data'
    x_file_list_train = [
        os.path.join(data_root, patient_set_list_train[i], patient_index_list_train[i], 'img_n4.nii.gz')
        for i in range(len(patient_index_list_train))
    ]
    y_list_train = [int(label_list_train[i]) for i in range(len(label_list_train))]

    x_file_list_val = [
        os.path.join(data_root, patient_set_list_val[i], patient_index_list_val[i], 'img_n4.nii.gz')
        for i in range(len(patient_index_list_val))
    ]
    y_list_val = [int(label_list_val[i]) for i in range(len(label_list_val))]


    print('Training cases:', len(x_file_list_train), 'Validation cases:', len(x_file_list_val))

    # Step 3: create data generators.
    train_generator = Generator.Dataset_3D(
        patient_set_list_train,
        patient_index_list_train,
        x_file_list_train,
        y_list_train,
        data_root,
        target_image_size=image_size,
        normalize_factor='equation',
        shuffle=True,
        augment=True,
        augment_frequency=0.8,
    )

    val_generator = Generator.Dataset_3D(
        patient_set_list_val,
        patient_index_list_val,
        x_file_list_val,
        y_list_val,
        data_root,
        target_image_size=image_size,
        normalize_factor='equation',
        shuffle=False,
        augment=False,
        augment_frequency=0,
    )

    # Step 4: define 3D ViT model.
    # The generator should return image tensors with shape:
    # [batch, channel, 128, 128, 160].
    # The ViT then splits each image into non-overlapping 3D patches:
    # (128/16) x (128/16) x (160/8) = 8 x 8 x 20 = 1280 tokens.
    model = vit_model.ViT3D(
        image_size=image_size,
        patch_size=vit_patch_size,
        in_channels=1,
        num_classes=2,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        attention_dropout=0.1,
    )

    # Step 5: define trainer.
    trainer = vit_model.Trainer(
        model,
        train_generator,
        val_generator,
        train_batch_size=train_batch_size,
        accum_iter=1,
        train_num_steps=train_num_steps,
        results_folder=model_output_path,
        train_lr=1e-3,
        train_lr_decay_every=50,
        save_models_every=save_models_every,
        validation_every=save_models_every,
        ema_update_every=10,
        ema_decay=0.95,
        amp=False,
        mixed_precision_type='fp16',
        split_batches=True,
        max_grad_norm=1.,
        num_workers=0,
    )

    trainer.train(pre_trained_model=trained_model_path, start_step=start_step)


if __name__ == '__main__':
    main()
