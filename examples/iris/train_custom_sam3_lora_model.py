"""Fine-tune the vendored SAM3 image model with LoRA on a COCO dataset."""

import argparse
from pathlib import Path

import torch
from loguru import logger

from telekinesis.iris import dataset, logger as iris_logger, models, trainer


def train_custom_sam3_lora_model_example(args: argparse.Namespace) -> None:
    """Build SAM3 LoRA and train it with the shared Iris trainer."""
    seed = 42

    # ===================== Configure Model ================================
    dataset_dir = args.dataset_dir
    metadata_dir = (
        dataset_dir / "train" if (dataset_dir / "train").is_dir() else dataset_dir
    )
    categories = dataset.COCODataset(metadata_dir).categories
    model_config = models.SAM3LoRAConfig(
        class_names=categories,
        sam3_checkpoint_path=None,
        lora_weights_path=None,
        load_from_hf=True,
        rank=16,
        alpha=32.0,
        dropout=0.1,
        apply_to_vision_encoder=True,
        apply_to_text_encoder=False,
        apply_to_geometry_encoder=True,
        apply_to_detr_encoder=True,
        apply_to_detr_decoder=True,
        apply_to_mask_decoder=True,
        resolution=1008,
        num_negative_prompts=3,
    )
    model = models.SAM3LoRA(config=model_config)

    # ===================== Prepare Dataset ================================
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
    output_dir = args.output_dir
    epochs = 100
    batch_size = 1
    learning_rate = 5e-5
    weight_decay = 0.01
    num_workers = 2
    gradient_clip_norm = 1.0
    mixed_precision = True
    evaluation_interval = 1
    checkpoint_interval = 1
    resume_from = args.resume_from
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Training SAM3 LoRA on {len(training_dataset)} images using {device}")
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
        gradient_clip_norm=gradient_clip_norm,
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
    train_custom_sam3_lora_model_example(args)
