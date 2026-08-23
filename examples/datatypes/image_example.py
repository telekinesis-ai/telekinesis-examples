"""Demonstrates the Telekinesis Image datatype."""

import time
from pathlib import Path

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def image_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    root = Path(__file__).parent

    data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image = datatypes.Image(data)
    logger.info(f"Created Image: {image}")

    buffer = data.tobytes()
    image_from_raw_buffer = datatypes.Image.from_raw_buffer(buffer, shape=data.shape, dtype=data.dtype)
    logger.info(f"Image created from raw buffer: {image_from_raw_buffer}")

    encoded_buffer = (root / "data/sample.jpg").read_bytes()
    image_from_encoded_buffer = datatypes.Image.from_encoded_buffer(encoded_buffer)
    logger.info(f"Image created from encoded buffer: {image_from_encoded_buffer}")

    image_from_path = datatypes.Image.from_path(root / "data/sample.jpg")
    logger.info(f"Image created from path: {image_from_path}")

    url = "https://assets.telekinesis.ai/examples/v1/images/screws_standing.jpg"
    image_from_url = datatypes.Image.from_url(url)
    logger.info(f"Image created from URL: {image_from_url}")

    # ======================= Inspect ===========================================
    logger.info(f"compression={image.compression}")
    logger.info(f"data={image.data}")
    logger.info(f"shape={image.shape}")
    logger.info(f"height={image.height}")
    logger.info(f"width={image.width}")
    logger.info(f"channels={image.channels}")
    logger.info(f"dtype={image.dtype}")

    # ======================= Operations =========================================
    image.data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    logger.info(f"Updated Image: {image}")

    gray_image = image_from_path.to_grayscale()
    logger.info(f"Grayscale image: {gray_image}")

    bgr_image = image_from_path.to_bgr()
    logger.info(f"BGR image: {bgr_image}")

    rgb_image = bgr_image.to_rgb()
    logger.info(f"RGB image: {rgb_image}")

    image_batch = image.expand_dims()
    logger.info(f"Expanded to ImageBatch: {image_batch}")

    image_copy = image.copy()
    logger.info(f"Copied Image: {image_copy}")

    image_numpy = image.to_numpy(copy=True)
    logger.info(f"NumPy Image:\n{image_numpy}")

    output_path = root / "data/output_image.jpg"
    gray_image.save_to_path(output_path)
    logger.info(f"Image saved to: {output_path}")

    mean_pixel_value = np.mean(image)
    flipped_image = np.flipud(image)
    logger.info(f"Mean pixel value: {mean_pixel_value}")
    logger.info(f"Flipped shape={flipped_image.shape}, dtype={flipped_image.dtype}")

    # ======================= Visualize =========================================
    rr.init("image_example", spawn=True)
    datatypes.visualize(image, entity_path="/image")
    datatypes.visualize(image_from_path, entity_path="/image/from_path")
    datatypes.visualize(image_from_url, entity_path="/image/from_url")
    datatypes.visualize(gray_image, entity_path="/image/grayscale")
    datatypes.visualize(bgr_image, entity_path="/image/bgr")
    datatypes.visualize(rgb_image, entity_path="/image/rgb")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(image)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized Image: {deserialized}")
    logger.info(f"Round-trip successful: {image == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    image_example()
