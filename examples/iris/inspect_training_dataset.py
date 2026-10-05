"""Load and inspect a COCO dataset."""

import argparse
from pathlib import Path

from loguru import logger
from torch.utils.data import DataLoader

from telekinesis.iris.dataset import COCODataset


def main(args: argparse.Namespace) -> None:
    """Load a COCO dataset and inspect individual and batched samples."""
    # ===================== Load Dataset ===================================
    dataset = COCODataset(
        dataset_dir=args.dataset_dir,
    )

    # ===================== Inspect Sample =================================
    logger.info(f"Dataset size: {len(dataset)}")

    image, target = dataset[0]
    logger.info("Single sample:")
    logger.info(f"  image shape: {image}")
    logger.info(f"  boxes: {target['boxes']}")
    logger.info(f"  labels: {target['labels']}")
    logger.info(f"  image_id: {target['image_id']}")

    def collate_fn(batch):
        """Group dataset samples into image and target tuples."""
        return tuple(zip(*batch))

    # ===================== Create Data Loader =============================
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ===================== Inspect Batch ==================================
    images, targets = next(iter(dataloader))

    logger.info("")
    logger.info("Batch:")
    logger.info(f"  images: {len(images)}")
    logger.info(f"  targets: {len(targets)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the COCO dataset directory.",
    )
    main(parser.parse_args())