"""Demonstrates the Telekinesis SegmentationImage datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes


def segmentation_image_example():
    """Demonstrate creation, access, visualization, update, NumPy interop, and serialization."""

    # ======================= Create ============================================
    data = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    image = datatypes.SegmentationImage(data)
    logger.info(f"Original SegmentationImage: {image}")

    # ======================= Inspect ===========================================
    logger.info(
        f"label_codes={image.label_codes}, "
        f"number_of_labels={image.number_of_labels}, "
        f"shape={image.data.shape}, "
        f"dtype={image.data.dtype}, "
        f"height={image.height}, "
        f"width={image.width}, "
        f"compression={image.compression}"
    )
    logger.info(f"SegmentationImage data:\n{image.data}")
    logger.info(f"NumPy array:\n{image.to_numpy()}")

    # ======================= Visualize =========================================
    rr.init("segmentation_image_example", spawn=True)
    datatypes.visualize(
        image,
        entity_path="/SegmentationImage/my_segmentation_image",
    )

    # ======================= Update ============================================
    updated_data = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    image.data = updated_data
    logger.info(f"Updated SegmentationImage: {image}")
    datatypes.visualize(
        image,
        entity_path="/SegmentationImage/my_updated_segmentation_image",
    )

    # ======================= NumPy Interop =====================================
    mean = np.mean(image)
    flipped = np.flipud(image)

    logger.info(f"Mean pixel value: {mean}")
    logger.info(f"Flipped shape={flipped.shape}, dtype={flipped.dtype}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(image)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized SegmentationImage: {deserialized}")
    logger.info(f"Round-trip successful: {image == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    segmentation_image_example()
