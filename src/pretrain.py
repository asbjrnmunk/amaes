#!/usr/bin/env python

import os

import torch
import lightning as L
import argparse

from models.self_supervised import SelfSupervisedModel
from augmentations.augmentation_composer import get_pretrain_augmentations, get_val_augmentations
from src.data.datamodule import PretrainDataModule

from data.pretrain_split import get_pretrain_split_config

from yucca.pipeline.configuration.configure_task import TaskConfig
from yucca.pipeline.configuration.configure_paths import PathConfig, detect_version
from yucca.pipeline.configuration.configure_plans import get_plan_config
from yucca.pipeline.configuration.configure_checkpoint import get_checkpoint_config
from yucca.pipeline.configuration.configure_seed import seed_everything_and_get_seed_config
from yucca.pipeline.configuration.configure_input_dims import InputDimensionsConfig

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p as ensure_dir_exists
import warnings

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_path", type=str, help="Path to base folder")

    parser.add_argument("--task", type=str, default="Task245_BRAINS-45K")
    parser.add_argument("--model_name", type=str, default="unet_lw_dec")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--token_size", type=int, default=4, help="i.e. MAE patch size, the masking unit.")
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument(
        "--patch_size", type=int, default=128, help="The patch size of the 3D patches extracted from the whole volume."
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default=None)
    parser.add_argument("--new_version", action="store_true")
    parser.add_argument("--optimizer", type=str, default="AdamW")

    parser.add_argument("--augmentation_preset", type=str, choices=["all", "basic", "none"], default="none")
    parser.add_argument("--loss_masked_tokens_only", default=False, action="store_true")

    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--overfit_batches", type=int, default=0)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=None)

    parser.add_argument("--experiment", type=str, default="base_experiment", help="name of experiment")

    args = parser.parse_args()
    planner = "UnsupervisedPlannerUnitSpacing"

    assert args.patch_size % 8 == 0, args.patch_size
    assert args.token_size < args.patch_size

    print(f"Using num_workers: {args.num_workers}, num_devices: {args.num_devices}")
    print("ARGS:", args)

    # Here we configure the outpath we will use to store model files and metadata
    # along with the path to plans file which will also be loaded.
    task_config = TaskConfig(
        continue_from_most_recent=not args.new_version,
        manager_name="AMAES",
        model_name=args.model_name,
        planner_name=planner,
        split_idx=0,
        task=args.task,
        split_method="simple_train_val_split",
        split_param=0.01,  # We use 1% of data for validation split and the rest as training data
        experiment=args.experiment,
        model_dimensions="3D",
        patch_based_training=True,
    )

    # AMAES requires base_dir to include the following three folders
    data_dir = os.path.join(args.base_path, "preprocessed")
    task_dir = os.path.join(data_dir, args.task)
    train_data_dir = os.path.join(task_dir, planner)
    plans_path = os.path.join(train_data_dir, planner + "_plans.json")

    # path where logs, checkpoints etc is stored
    save_dir = os.path.join(args.base_path, "models", args.task, args.model_name)
    versions_dir = os.path.join(save_dir, "versions")
    version = detect_version(versions_dir, task_config.continue_from_most_recent)
    version_dir = os.path.join(versions_dir, f"version_{version}")
    ensure_dir_exists(version_dir)

    path_config = PathConfig(
        plans_path=plans_path,
        save_dir=save_dir,
        task_dir=task_dir,
        train_data_dir=train_data_dir,
        version_dir=version_dir,
        version=version,
    )

    ckpt_config = get_checkpoint_config(
        path_config=path_config,
        continue_from_most_recent=task_config.continue_from_most_recent,
        current_experiment=task_config.experiment,
    )

    seed_config = seed_everything_and_get_seed_config(ckpt_seed=ckpt_config.ckpt_seed)

    plan_config = get_plan_config(
        ckpt_plans=ckpt_config.ckpt_plans,
        plans_path=path_config.plans_path,
        stage="fit",
    )

    assert plan_config.task_type == "self-supervised"

    assert isinstance(task_config.split_param, float)

    splits_config = get_pretrain_split_config(
        method=task_config.split_method,
        idx=task_config.split_idx,
        split_ratio=task_config.split_param,
        path_config=path_config,
    )

    input_dims_config = InputDimensionsConfig(
        batch_size=args.batch_size, patch_size=(args.patch_size,) * 3, num_modalities=1  # type: ignore
    )
    assert len(input_dims_config.patch_size) == 3

    train_transforms = get_pretrain_augmentations(input_dims_config.patch_size, args.augmentation_preset)
    val_transforms = get_val_augmentations()

    data = PretrainDataModule(
        patch_size=input_dims_config.patch_size,
        batch_size=input_dims_config.batch_size,
        num_workers=args.num_workers,
        splits_config=splits_config,
        split_idx=task_config.split_idx,
        train_data_dir=path_config.train_data_dir,
        composed_train_transforms=train_transforms,
        composed_val_transforms=val_transforms,
    )

    effective_batch_size = args.accumulate_grad_batches * args.num_devices * input_dims_config.batch_size
    train_dataset_size = len(data.splits_config.train(task_config.split_idx))
    val_dataset_size = len(data.splits_config.train(task_config.split_idx))
    steps_per_epoch = int(train_dataset_size / effective_batch_size) if args.overfit_batches == 0 else args.overfit_batches
    max_iterations = int(args.epochs * steps_per_epoch)

    print(
        f"Starting training with {max_iterations} max iterations over {args.epochs} epochs "
        f"with {train_dataset_size} datapoints and and effective batch size of {effective_batch_size}"
    )

    model = SelfSupervisedModel(
        model_name=args.model_name,
        config=task_config.lm_hparams()
        | path_config.lm_hparams()
        | ckpt_config.lm_hparams(without=["ckpt_plans"])
        | seed_config.lm_hparams()
        | splits_config.lm_hparams()
        | plan_config.lm_hparams(without=["plans"])
        | input_dims_config.lm_hparams()
        | {"precision": args.precision},
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        steps_per_epoch=steps_per_epoch,
        num_classes=1,
        input_channels=input_dims_config.num_modalities,
        patch_size=input_dims_config.patch_size,
        token_size=args.token_size,
        mask_ratio=args.mask_ratio,
        should_compile=args.compile,
        compile_mode=args.compile_mode,
        rec_loss_masked_only=args.loss_masked_tokens_only,
    )

    trainer = L.Trainer(
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        strategy="ddp" if args.num_devices > 1 else "auto",
        num_nodes=1,
        devices=args.num_devices,
        default_root_dir=path_config.save_dir,
        max_epochs=args.epochs,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
        limit_val_batches=args.limit_val_batches,
        limit_train_batches=args.limit_train_batches,
        overfit_batches=args.overfit_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        num_sanity_val_steps=0 if args.overfit_batches > 0 else 2,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    trainer.fit(model=model, datamodule=data, ckpt_path="last")
    trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
