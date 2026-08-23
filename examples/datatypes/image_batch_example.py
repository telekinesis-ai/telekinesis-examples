"""Demonstrates the Telekinesis ImageBatch datatype."""

import time
from pathlib import Path

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def image_batch_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    root = Path(__file__).parent

    image_1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    image_2 = datatypes.Image.from_path(root / "data/sample.jpg").to_numpy()
    images = [image_1, image_2]
    image_batch = datatypes.ImageBatch(images)
    logger.info(f"Created ImageBatch: {image_batch}")

    # ======================= Inspect ===========================================
    logger.info(f"Number of images in batch: {len(image_batch)}")
    logger.info(f"shapes={image_batch.shapes}")
    logger.info(f"dtypes={image_batch.dtypes}")
    logger.info(f"compressions={image_batch.compressions}")

    # ======================= Operations =========================================
    image_batch_copy = image_batch.copy()
    logger.info(f"Copied ImageBatch: {image_batch_copy}")

    image_batch_numpy = image_batch.to_numpy(copy=True)
    logger.info(f"NumPy ImageBatch: shapes={[arr.shape for arr in image_batch_numpy]}")

    index = 1
    image_at_index = image_batch[index]
    logger.info(f"Image at index {index}: {image_at_index}")

    sliced_batch = image_batch[0:1]
    logger.info(f"Sliced ImageBatch: {sliced_batch}")

    keep_mask = np.array([True, False])
    masked_batch = image_batch[keep_mask]
    logger.info(f"Masked ImageBatch: {masked_batch}")

    # Indexing returns a real Image, so its own methods remain available.
    gray_image = image_at_index.to_grayscale()
    logger.info(f"Grayscale image at index {index}: {gray_image}")
    gray_image.save_to_path(root / "data/grayscale_image.jpg")

    # ImageBatch has no setter for its contents; rebuild a new batch instead.
    updated_images = list(images)
    updated_images[0] = np.random.randint(0, 255, (1907, 512, 3), dtype=np.uint8)
    rebuilt_image_batch = datatypes.ImageBatch(updated_images)
    logger.info(f"Rebuilt ImageBatch: {rebuilt_image_batch}")

    # ======================= Visualize =========================================
    rr.init("image_batch_example", spawn=True)
    datatypes.visualize(image_batch, entity_path="/image_batch")
    datatypes.visualize(image_at_index, entity_path="/image_batch/image_1")
    datatypes.visualize(gray_image, entity_path="/image_batch/image_1/grayscale")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(image_batch)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized ImageBatch: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == image_batch}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    image_batch_example()
