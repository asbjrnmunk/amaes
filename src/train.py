#!/usr/bin/env python

import os
import logging

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse

from models.supervised import SupervisedModel
from src.augmentations.finetune_augmentation_presets import get_finetune_augmentation_params

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p as ensure_dir_exists

from yucca.modules.data.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
from yucca.modules.callbacks.prediction_writer import WritePredictionFromLogits
from yucca.modules.callbacks.loggers import YuccaLogger

from yucca.pipeline.configuration.split_data import get_split_config
from yucca.pipeline.configuration.configure_task import TaskConfig
from yucca.pipeline.configuration.configure_paths import PathConfig, detect_version
from yucca.pipeline.configuration.configure_plans import get_plan_config
from yucca.pipeline.configuration.configure_checkpoint import get_checkpoint_config
from yucca.pipeline.configuration.configure_seed import seed_everything_and_get_seed_config
from yucca.pipeline.configuration.configure_input_dims import InputDimensionsConfig

from evaluator import Evaluator


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_path", type=str, help="Path to base folder")
    parser.add_argument("--pretrained_weights_path", type=str, help="Ckpt to finetune", default=None)

    parser.add_argument("--model_name", type=str, default="unet_xl")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--planner", type=str, default="YuccaPlanner_1_1_1")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default=None)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--deep_supervision", action="store_true")

    parser.add_argument("--new_version", action="store_true")
    parser.add_argument("--augmentation_preset", type=str, choices=["all", "basic", "none"], default="basic")

    # Training Params
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_batches_per_epoch", type=int, default=100)

    # Experiment Params
    parser.add_argument("--task", type=str, default="Task006_WMH_Flair")
    parser.add_argument("--split_method", type=str, default="simple_train_val_split")
    parser.add_argument("--split_param", type=str, help="", default=0.2)
    parser.add_argument("--split_idx", type=int, default=0, help="Index of the split to use for kfold")
    parser.add_argument("--without_inference", action="store_true")

    parser.add_argument("--experiment", type=str, default="experiment", help="name of experiment")

    args = parser.parse_args()

    planner = args.planner
    assert args.patch_size % 8 == 0, args.patch_size

    if args.split_method == "kfold" or args.split_method == "n_samples":
        split_param = int(args.split_param)
    elif args.split_method == "simple_train_val_split":
        split_param = float(args.split_param)
    else:
        split_param = args.split_param

    run_type = "from_scratch" if args.pretrained_weights_path is None else "finetune"
    experiment = f"{run_type}_{args.experiment}"

    print(f"Using num_workers: {args.num_workers}, num_devices: {args.num_devices}")
    print("ARGS:", args)

    task_config = TaskConfig(
        continue_from_most_recent=not args.new_version,
        manager_name="AMAES",
        model_dimensions="3D",
        model_name=args.model_name,
        patch_based_training=True,
        planner_name=planner,
        split_idx=args.split_idx,
        task=args.task,
        experiment=experiment,
        split_method=args.split_method,
        split_param=split_param,
    )

    # AMAES requires base_dir to include the following three folders
    data_dir = os.path.join(args.base_path, "preprocessed")
    raw_dir = os.path.join(args.base_path, "raw")
    results_dir = os.path.join(args.base_path, "results")
    models_dir = os.path.join(args.base_path, "models")

    task_dir = os.path.join(data_dir, args.task)
    train_data_dir = os.path.join(task_dir, planner)
    plans_path = os.path.join(train_data_dir, planner + "_plans.json")

    # path where logs, checkpoints etc is stored
    save_dir = os.path.join(args.base_path, "models", args.task, args.model_name)
    version = detect_version(save_dir, task_config.continue_from_most_recent)
    version_dir = os.path.join(save_dir, f"version_{version}")
    ensure_dir_exists(version_dir)

    path_config = PathConfig(
        plans_path=plans_path,
        save_dir=save_dir,
        task_dir=task_dir,
        train_data_dir=train_data_dir,
        version_dir=version_dir,
        version=version,
    )

    # ekstra inference paths
    if not args.without_inference:
        pred_data_dir = os.path.join(raw_dir, args.task, "imagesTs")
        gt_data_dir = os.path.join(raw_dir, args.task, "labelsTs")
        prediction_output_dir = os.path.join(
            results_dir, path_config.version_dir.removeprefix(models_dir + "/"), "checkpoints", "last"
        )

        os.makedirs(prediction_output_dir, exist_ok=True)

    # Used to restart training, not initialize weights
    ckpt_config = get_checkpoint_config(
        path_config=path_config,
        continue_from_most_recent=task_config.continue_from_most_recent,
        current_experiment=task_config.experiment,
    )

    seed_config = seed_everything_and_get_seed_config()

    # we load the plan for training, but this will also be used for inference!
    plan_config = get_plan_config(
        ckpt_plans=ckpt_config.ckpt_plans,
        plans_path=path_config.plans_path,
        stage="fit",
    )

    splits_config = get_split_config(method=task_config.split_method, param=task_config.split_param, path_config=path_config)
    modalities = max(1, plan_config.plans.get("num_modalities") or len(plan_config.plans["dataset_properties"]["modalities"]))
    input_dims_config = InputDimensionsConfig(
        batch_size=args.batch_size, patch_size=(args.patch_size,) * 3, num_modalities=modalities  # type: ignore
    )

    latest_ckpt = ModelCheckpoint(
        every_n_epochs=10,
        save_top_k=1,
        filename="last",
        enable_version_counter=False,
    )
    callbacks = [latest_ckpt]

    yucca_logger = YuccaLogger(
        save_dir=save_dir,
        version=version,
        steps_per_epoch=args.train_batches_per_epoch,
    )
    loggers = [yucca_logger]

    # additional augmentation parameters
    aug_params = get_finetune_augmentation_params(args.augmentation_preset)

    augmenter = YuccaAugmentationComposer(
        patch_size=input_dims_config.patch_size,
        task_type_preset="segmentation",
        parameter_dict=aug_params,
        deep_supervision=args.deep_supervision,
    )

    data = YuccaDataModule(
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
        patch_size=input_dims_config.patch_size,
        batch_size=input_dims_config.batch_size,
        train_data_dir=path_config.train_data_dir,
        pred_data_dir=None if args.without_inference else pred_data_dir,
        pred_save_dir=None if args.without_inference else prediction_output_dir,
        image_extension=plan_config.image_extension,
        task_type=plan_config.task_type,
        splits_config=splits_config,
        split_idx=task_config.split_idx,
        num_workers=args.num_workers,
        val_sampler=None,
    )  # type: ignore

    effective_batch_size = args.num_devices * input_dims_config.batch_size
    train_dataset_size = len(data.splits_config.train(task_config.split_idx))
    val_dataset_size = len(data.splits_config.val(task_config.split_idx))
    max_iterations = int(args.epochs * args.train_batches_per_epoch)

    print("Train dataset: ", data.splits_config.train(task_config.split_idx))
    print("Val dataset: ", data.splits_config.val(task_config.split_idx))

    print("run_type: ", run_type)

    print(
        f"Starting training with {max_iterations} max iterations over {args.epochs} epochs "
        f"with train dataset of size {train_dataset_size} datapoints and val dataset of size {val_dataset_size} "
        f"and effective batch size of {effective_batch_size}"
    )

    model = SupervisedModel(
        config=task_config.lm_hparams()
        | path_config.lm_hparams()
        | ckpt_config.lm_hparams()
        | seed_config.lm_hparams()
        | splits_config.lm_hparams()
        | plan_config.lm_hparams()
        | input_dims_config.lm_hparams()
        | {"precision": args.precision, "experiment": experiment, "run_type": run_type},
        learning_rate=args.learning_rate,
        do_compile=args.compile,
        compile_mode="default",
    )

    trainer = L.Trainer(
        callbacks=callbacks,
        logger=loggers,
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        strategy="auto",
        num_nodes=1,
        devices=args.num_devices,
        default_root_dir=path_config.save_dir,
        max_epochs=args.epochs,
        limit_train_batches=args.train_batches_per_epoch,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
    )

    if run_type == "finetune":
        print("Transfering weights for finetuning")
        print(ckpt_config.ckpt_path)
        assert ckpt_config.ckpt_path is None, "You are loading weights when continuing training. Dont do that."

        state_dict = torch.load(args.pretrained_weights_path, map_location=torch.device("cpu"))

        # We check if the pretrained checkpoint is compiled, and if we wish to finetune uncompiled,
        # then we have to remove the string "_orig_mod" from each key in the state_dict.
        if "_orig_mod" in next(iter(state_dict)) and not args.compile:
            uncompiled_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace("_orig_mod.", "")
                uncompiled_state_dict[new_key] = state_dict[key]
            state_dict = uncompiled_state_dict

        num_succesful_weights_transfered = model.load_state_dict(state_dict=state_dict, strict=False)
        assert num_succesful_weights_transfered > 0
    else:
        print("Training from scratch, so no weights will be transfered")

    trainer.fit(model=model, datamodule=data, ckpt_path="last")

    if not args.without_inference:
        pred_writer = WritePredictionFromLogits(output_dir=prediction_output_dir, save_softmax=False, write_interval="batch")
        trainer.callbacks.append(pred_writer)

        trainer.predict(model=model, dataloaders=data, return_predictions=False)

        evaluator = Evaluator(
            model.num_classes,
            folder_with_predictions=prediction_output_dir,
            folder_with_ground_truth=gt_data_dir,
            raw_data_path=raw_dir,
            num_workers=args.num_workers,
            overwrite=True,
        )
        evaluator.run()
