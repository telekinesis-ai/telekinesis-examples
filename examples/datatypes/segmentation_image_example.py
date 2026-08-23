"""Demonstrates the Telekinesis SegmentationImage datatype."""

import time
from pathlib import Path

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def segmentation_image_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    data = np.random.randint(0, 5, (480, 640), dtype=np.uint8)
    segmentation_image = datatypes.SegmentationImage(data)
    logger.info(f"Created SegmentationImage: {segmentation_image}")

    segmentation_image_from_coerce = datatypes.SegmentationImage.coerce(data)
    logger.info(f"SegmentationImage created via coerce: {segmentation_image_from_coerce}")

    raw_buffer = data.tobytes()
    segmentation_image_from_raw_buffer = datatypes.SegmentationImage.from_raw_buffer(
        raw_buffer, shape=data.shape, dtype=data.dtype
    )
    logger.info(f"SegmentationImage created from raw buffer: {segmentation_image_from_raw_buffer}")

    save_path = Path("results/segmentation_image_example.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    segmentation_image.save_to_path(save_path)
    segmentation_image_from_path = datatypes.SegmentationImage.from_path(save_path)
    logger.info(f"SegmentationImage created from path: {segmentation_image_from_path}")

    encoded_buffer = save_path.read_bytes()
    segmentation_image_from_encoded_buffer = datatypes.SegmentationImage.from_encoded_buffer(
        encoded_buffer
    )
    logger.info(
        f"SegmentationImage created from encoded buffer: {segmentation_image_from_encoded_buffer}"
    )

    # ======================= Inspect ===========================================
    logger.info(f"data={segmentation_image.data}")
    logger.info(f"shape={segmentation_image.shape}")
    logger.info(f"height={segmentation_image.height}")
    logger.info(f"width={segmentation_image.width}")
    logger.info(f"dtype={segmentation_image.dtype}")
    logger.info(f"label_codes={segmentation_image.label_codes}")
    logger.info(f"number_of_labels={segmentation_image.number_of_labels}")
    logger.info(f"compression={segmentation_image.compression}")

    # ======================= Operations =========================================
    updated_data = np.random.randint(0, 5, (480, 640), dtype=np.uint8)
    segmentation_image.data = updated_data
    logger.info(f"Updated SegmentationImage: {segmentation_image}")

    binary_segmentation_image = segmentation_image.to_binary()
    logger.info(f"Binary SegmentationImage: {binary_segmentation_image}")

    segmentation_image_copy = segmentation_image.copy()
    logger.info(f"Copied SegmentationImage: {segmentation_image_copy}")

    segmentation_image_numpy = segmentation_image.to_numpy(copy=True)
    logger.info(f"NumPy array:\n{segmentation_image_numpy}")

    numpy_array = np.asarray(segmentation_image)
    logger.info(f"Mean label value: {np.mean(numpy_array)}")

    # ======================= Visualize =========================================
    rr.init("segmentation_image_example", spawn=True)
    datatypes.visualize(
        segmentation_image, entity_path="/segmentation_image/updated"
    )
    datatypes.visualize(
        binary_segmentation_image, entity_path="/segmentation_image/binary"
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(segmentation_image)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized SegmentationImage: {deserialized}")
    logger.info(f"Round-trip successful: {segmentation_image == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    segmentation_image_example()
