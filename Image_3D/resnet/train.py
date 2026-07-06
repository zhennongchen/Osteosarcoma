
import sys
sys.path.append('/host/d/Github/')

import argparse
import os

import Osteosarcoma.Build_lists.Build_list as Build_list
import Osteosarcoma.functions_collection as ff
import Osteosarcoma.Image_3D.Generator_ResNet as Generator_ResNet
import Osteosarcoma.Image_3D.resnet.model as resnet_model





def none_or_path(value):
    """Allow command line values like --trained_model_path None."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {'none', 'null', ''}:
        return None
    return value


def parse_fold_list(value):
    """Parse fold strings like '4', '012345', '0,1,2,3,4', or '0 1 2 3 4'."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    value = str(value).strip()
    if value == '':
        return []
    if ',' in value or ' ' in value:
        return [int(v) for v in value.replace(',', ' ').split()]
    return [int(ch) for ch in value]


def fold_tag(folds):
    return ''.join(str(fold) for fold in folds)


def resolve_train_val_folds(split_mode, fold_list_value, val_fold_value):
    fold_list = parse_fold_list(fold_list_value)
    val_fold = parse_fold_list(val_fold_value)

    if fold_list is None or len(fold_list) == 0:
        raise ValueError('--fold_list must contain at least one fold.')
    if val_fold is None or len(val_fold) == 0:
        raise ValueError('Please set --val_fold. Example: --val_fold 4 or --val_fold 012345')

    invalid_folds = [fold for fold in fold_list + val_fold if fold not in [0, 1, 2, 3, 4, 5]]
    if invalid_folds:
        raise ValueError(f'Fold values must be among 0,1,2,3,4,5. Invalid: {invalid_folds}')

    split_mode = str(split_mode).lower()
    if split_mode == 'cv':
        if len(val_fold) != 1:
            raise ValueError(f'cv mode requires exactly one validation fold. Got: {val_fold}')
        if val_fold[0] not in fold_list:
            raise ValueError(f'In cv mode, val_fold must be inside fold_list. fold_list={fold_list}, val_fold={val_fold}')
        train_fold = [fold for fold in fold_list if fold not in val_fold]
        setting_suffix = f'fold{fold_tag(val_fold)}'
    elif split_mode == 'all':
        train_fold = list(fold_list)
        setting_suffix = f'all_fold{fold_tag(val_fold)}'
    else:
        raise ValueError(f"--split_mode must be 'cv' or 'all'. Got: {split_mode}")

    if len(train_fold) == 0:
        raise ValueError(f'Training fold list is empty. split_mode={split_mode}, fold_list={fold_list}, val_fold={val_fold}')

    return fold_list, train_fold, val_fold, setting_suffix


def parse_args():
    parser = argparse.ArgumentParser(description='Train MedicalNet-based 3D ResNet for Osteosarcoma MRI classification.')

    # Experiment identity.
    parser.add_argument('--label', type=str, default='Prognosis')
    parser.add_argument('--trial_name', type=str, required=True, help='Exact output folder name. No automatic name building is performed.')
    parser.add_argument('--model_depth', type=int, default=18, choices=[10, 18, 34, 50, 101, 152, 200], help='Depth of the 3D ResNet model.')
    parser.add_argument('--fine_tune_stage', type=str, default='1', choices=['all', 'fc', '1', '2'])
    parser.add_argument('--only_tumor_pixels', type=str, default='seg', choices=['roi', 'seg'], help='Kept for backward compatibility; generator now always returns [full,bbox,tumor].')
    parser.add_argument('--augment_context', type=str, default='simple', choices=['simple', 'full'])
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--use_medicalnet_pretrained', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--trained_model_path', type=none_or_path, default=None)
    parser.add_argument('--start_step', type=int, default=0)

    # Patient split settings.
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--split_mode', type=str, default='cv', choices=['cv', 'all'])
    parser.add_argument('--fold_list', type=str, default='01234', help='Training fold universe. Examples: 01234, 012345, or 0,1,2,3,4.')
    parser.add_argument('--val_fold', type=str, default=None, help='Validation fold(s). Examples: 4, 012345, or 0,1,2,3,4,5.')

    # Training settings.
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--train_num_steps', type=int, default=500)
    parser.add_argument('--save_models_every', type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.val_fold is None:
        raise ValueError('Please set --val_fold. Example: --val_fold 4')

    label = args.label
    label_type = label + '_label'
    fine_tune_stage = args.fine_tune_stage
    only_tumor_pixels = args.only_tumor_pixels
    trial_name = args.trial_name
    if trial_name.lower() in {'none', 'null', ''}:
        raise ValueError('--trial_name must be explicitly set. No automatic trial name is generated now.')

    fold_list, train_fold, val_fold, setting_suffix = resolve_train_val_folds(
        args.split_mode,
        args.fold_list,
        args.val_fold,
    )

    trial_output_path = os.path.join('/host/d/projects/Habitats/models/', label, trial_name)
    setting_name = f'random{args.random_state}_{setting_suffix}'
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

    # Current first-pass 3D ResNet target size.
    image_size = (96,96, 64)

    train_batch_size = args.train_batch_size
    train_num_steps = args.train_num_steps
    save_models_every = args.save_models_every

    print('============================================================')
    print(f'MedicalNet 3D ResNet{args.model_depth} training settings')
    print('label:', label)
    print('label_type:', label_type)
    print('trial_name:', trial_name)
    print('model_depth:', args.model_depth)
    print('fine_tune_stage:', fine_tune_stage)
    print('only_tumor_pixels:', only_tumor_pixels)
    print('augment_context:', args.augment_context)
    print('in_channels:', args.in_channels)
    print('use_medicalnet_pretrained:', args.use_medicalnet_pretrained)
    print('random_state:', args.random_state)
    print('split_mode:', args.split_mode)
    print('fold_list:', fold_list)
    print('train_fold:', train_fold)
    print('val_fold:', val_fold)
    print('setting_name:', setting_name)
    print('trained_model_path:', trained_model_path)
    print('start_step:', start_step)
    print('optimizer:', args.optimizer)
    print('train_batch_size:', train_batch_size)
    print('train_num_steps:', train_num_steps)
    print('save_models_every:', save_models_every)
    print('image_size:', image_size)
    print('trial_output_path:', trial_output_path)
    print('model_output_path:', model_output_path)

    patient_list_file = (
        '/host/e/D/Data/Habitats/Jishuitan/Patient_lists/'
        f'image_label_info_set12_5fold_{label.lower()}_random{args.random_state}.xlsx'
    )
    print('patient_list_file:', patient_list_file)

    if not os.path.isfile(patient_list_file):
        raise FileNotFoundError(f'Patient split file not found: {patient_list_file}')

    build = Build_list.Build(patient_list_file)
    fold_list_train, patient_set_list_train, patient_index_list_train, label_list_train, _, _ = build.__build__(
        batch_list=train_fold,
        label_column_name=label_type,
    )
    fold_list_val, patient_set_list_val, patient_index_list_val, label_list_val, _, _ = build.__build__(
        batch_list=val_fold,
        label_column_name=label_type,
    )

    data_root = '/host/e/D/Data/Habitats/Jishuitan/resampled_data_new'
    x_file_list_train = [
        os.path.join(data_root, patient_set_list_train[i], patient_index_list_train[i], 'img.nii.gz')
        for i in range(len(patient_index_list_train))
    ]
    y_list_train = [int(label_list_train[i]) for i in range(len(label_list_train))]

    x_file_list_val = [
        os.path.join(data_root, patient_set_list_val[i], patient_index_list_val[i], 'img.nii.gz')
        for i in range(len(patient_index_list_val))
    ]
    y_list_val = [int(label_list_val[i]) for i in range(len(label_list_val))]

    print('Training cases:', len(x_file_list_train), 'Validation cases:', len(x_file_list_val))

    train_generator = Generator_ResNet.Dataset_3D(
        patient_set_list_train,
        patient_index_list_train,
        x_file_list_train,
        y_list_train,
        data_root,
        target_image_size=image_size,
        normalize_factor='medicalnet',
        only_tumor_pixels=only_tumor_pixels,
        augment_context=args.augment_context,
        shuffle=True,
        augment=True,
        augment_frequency=0.8,
    )

    val_generator = Generator_ResNet.Dataset_3D(
        patient_set_list_val,
        patient_index_list_val,
        x_file_list_val,
        y_list_val,
        data_root,
        target_image_size=image_size,
        normalize_factor='medicalnet',
        only_tumor_pixels=only_tumor_pixels,
        augment_context=args.augment_context,
        shuffle=False,
        augment=False,
        augment_frequency=0,
    )

    model = resnet_model.build_resnet3d_model(model_depth=args.model_depth, num_classes=2, in_channels=args.in_channels)
  
    MEDICALNET_PRETRAIN_PATH = f'/host/e/D/Data/Habitats/MedicalNet_weights/pretrain/resnet_{args.model_depth}_23dataset.pth'
    print(f'Using MedicalNet pretrained weights from: {MEDICALNET_PRETRAIN_PATH}')

    # If no project checkpoint is given, start from MedicalNet pretrained weights.
    # If a project checkpoint is given, load it later in Trainer.train and continue training.
    if trained_model_path is None and args.use_medicalnet_pretrained == 'yes':
        model = resnet_model.load_medicalnet_pretrained(
            model,
            MEDICALNET_PRETRAIN_PATH,
            verbose=True,
        )
    elif trained_model_path is None:
        print('No project checkpoint and use_medicalnet_pretrained=no; training from scratch.')

    model = resnet_model.configure_fine_tuning(model, fine_tune_stage=fine_tune_stage)
    
    if args.optimizer == 'sgd':
        train_lr = 1e-3
    elif args.optimizer == 'adam':
        train_lr = 1e-4
    trainer = resnet_model.Trainer(
        model,
        train_generator,
        val_generator,
        train_batch_size=train_batch_size,
        accum_iter=1,
        train_num_steps=train_num_steps,
        results_folder=model_output_path,
        train_lr=train_lr,
        train_lr_decay_every=50,
        optimizer=args.optimizer,
        train_momentum=0.9,
        train_weight_decay=1e-4,
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
