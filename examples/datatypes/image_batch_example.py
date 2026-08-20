"""Demonstrates the Telekinesis ImageBatch datatype."""

import time
from pathlib import Path

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def image_batch_example():
    """Demonstrate creation, inspection, visualization, indexing, grayscale conversion, rebuilding, and serialization."""

    # ======================= Create ============================================
    ROOT_PATH = Path(__file__).parent
    image_1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    image_2 = datatypes.Image.from_path(ROOT_PATH / "data/sample.jpg").to_numpy()
    images = [image_1, image_2]
    image_batch = datatypes.ImageBatch(images)
    logger.info(f"Original ImageBatch: {image_batch}")

    # ======================= Inspect ===========================================
    logger.info(f"dtypes={image_batch.dtypes}, shapes={image_batch.shapes}, compressions={image_batch.compressions}")
    logger.info(f"NumPy array: {image_batch.to_numpy()}")

    # ======================= Visualize =========================================
    rr.init("image_batch_example", spawn=True)
    datatypes.visualize(image_batch, entity_path="/ImageBatch")

    # ======================= Index =============================================
    index = 1
    image_at_index = image_batch[index]
    logger.info(f"Image at index {index}: {image_at_index}")
    datatypes.visualize(image_at_index, entity_path="/ImageBatch/Image_1")

    # ======================= Grayscale =========================================
    gray_image = image_at_index.to_grayscale()
    logger.info(f"Grayscale image at index {index}: {gray_image}")
    datatypes.visualize(gray_image, entity_path="/ImageBatch/Image_1/Grayscale")
    gray_image.save_to_path(ROOT_PATH / "data/grayscale_image.jpg")

    # ======================= Rebuild ===========================================
    index = 0
    updated_image = np.random.randint(0, 255, (1907, 512, 3), dtype=np.uint8)
    images[index] = updated_image
    image_batch = datatypes.ImageBatch(images)
    logger.info(f"Rebuilt ImageBatch at index 0: {image_batch}")

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
