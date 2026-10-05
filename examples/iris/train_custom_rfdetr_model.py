"""Train an RF-DETR model on a COCO dataset."""

import argparse
from pathlib import Path

import torch
from loguru import logger

from telekinesis.iris import dataset, logger as iris_logger, models, trainer


def train_custom_rfdetr_model_example(args: argparse.Namespace) -> None:
    """Configure and train an RF-DETR model on a COCO dataset."""
    seed = 42
    # ===================== Configure Model ================================
    # Detection: nano, small, medium, base, large
    # Segmentation: seg-nano, seg-small, seg-medium, seg-large,
    #               seg-xlarge, seg-2xlarge
    model_config = models.RFDETRConfig(
        variant=models.RFDETRVariant.SEG_MEDIUM,
        num_classes=args.num_classes,
        pretrained=True,
        resolution=None,
        validation_images_to_log=4,
    )
    model = models.RFDETR(config=model_config)

    # ===================== Prepare Dataset ================================
    dataset_dir = args.dataset_dir
    validation_split = 0.3

    training_dataset, validation_dataset = dataset.prepare_coco_datasets(
        dataset_dir=dataset_dir,
        validation_split=validation_split,
        seed=seed,
        train_transforms=model.train_transforms,
        val_transforms=model.val_transforms,
        include_masks=model.requires_masks,
    )

    # ===================== Train Model ====================================
    # All results will be saved in this directory
    output_dir = args.output_dir

    epochs = 400
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-4
    num_workers = 0
    mixed_precision = True
    evaluation_interval = 1
    checkpoint_interval = 1
    resume_from = args.resume_from
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        f"Training RF-DETR {model_config.variant.value} on {len(training_dataset)} images "
        f"at {model.model_config.resolution}x{model.model_config.resolution} using {device}."
    )

    metric_logger = iris_logger.TensorBoardMetricLogger(output_dir / "tensorboard")
    model_trainer = trainer.Trainer(
        model=model,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        num_workers=num_workers,
        mixed_precision=mixed_precision,
        evaluation_interval=evaluation_interval,
        checkpoint_interval=checkpoint_interval,
        seed=seed,
        metric_logger=metric_logger,
    )

    try:
        if resume_from is None:
            model_trainer.train()
        else:
            model_trainer.resume(resume_from)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        logger.error(f"An error occurred during training: {error}")

    tensorboard_dir = (output_dir / "tensorboard").resolve()
    logger.info(f'TensorBoard: tensorboard --logdir "{tensorboard_dir}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the COCO dataset directory.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="Number of object classes in the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to save training outputs.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Optional checkpoint from which to resume training.",
    )
    args = parser.parse_args()
    train_custom_rfdetr_model_example(args)
