"""Demonstrates the Telekinesis Image datatype."""

import time
from pathlib import Path

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def image_example():
    """Demonstrate creation, access, conversions, saving, NumPy interop, and serialization."""

    # ======================= Create ============================================
    data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    logger.info(f"Input data: shape={data.shape}, dtype={data.dtype}")

    image = datatypes.Image(data)

    logger.info(f"Created Image: {image}")

    # ======================= Inspect ===========================================
    logger.info(
        f"shape={image.data.shape}, "
        f"dtype={image.data.dtype}, "
        f"height={image.height}, "
        f"width={image.width}, "
        f"channels={image.channels}, "
        f"compression={image.compression}"
    )
    logger.info(f"Image data: {image.data}")
    logger.info(f"NumPy array: {image.to_numpy()}")

    # ======================= Visualize =========================================
    rr.init("image_example", spawn=True)
    datatypes.visualize(image, entity_path="/Image")

    # ======================= Update ============================================
    image.data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    logger.info(f"Updated Image: {image}")
    datatypes.visualize(image, entity_path="/Image")

    # ======================= Load From Path ====================================
    root = Path(__file__).parent
    image_from_path = datatypes.Image.from_path(root / "data/sample.jpg")

    logger.info(f"shape={image_from_path.shape}, dtype={image_from_path.dtype}")
    datatypes.visualize(image_from_path, entity_path="/ImageFromPath")

    # ======================= Load From URL =====================================
    url = "https://assets.telekinesis.ai/examples/v1/images/screws_standing.jpg"
    image_from_url = datatypes.Image.from_url(url)

    logger.info(f"shape={image_from_url.shape}, dtype={image_from_url.dtype}")
    datatypes.visualize(image_from_url, entity_path="/ImageFromURL")

    # ======================= Convert To Grayscale ==============================
    gray_image = image_from_path.to_grayscale()

    logger.info(f"shape={gray_image.shape}, dtype={gray_image.dtype}")
    datatypes.visualize(gray_image, entity_path="/GrayImage")

    # ======================= Convert To BGR ====================================
    bgr_image = image_from_path.to_bgr()

    logger.info(f"shape={bgr_image.shape}, dtype={bgr_image.dtype}")
    datatypes.visualize(bgr_image, entity_path="/BGRImage")

    # ======================= Convert To RGB ====================================
    rgb_image = bgr_image.to_rgb()

    logger.info(f"shape={rgb_image.shape}, dtype={rgb_image.dtype}")
    datatypes.visualize(rgb_image, entity_path="/RGBImage")

    # ======================= Expand Dims =======================================
    image_batch = image.expand_dims()

    logger.info(f"shapes={image_batch.shapes}, dtypes={image_batch.dtypes}")

    # ======================= Save ==============================================
    output_path = root / "data/output_image.jpg"
    gray_image.save_to_path(path=output_path)

    logger.info(f"Image saved to: {output_path}")

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

    logger.info(f"Deserialized Image: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == image}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    image_example()
