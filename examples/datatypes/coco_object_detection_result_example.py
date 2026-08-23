"""Demonstrates the Telekinesis COCOObjectDetectionResult datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def coco_object_detection_result_example():
    """Demonstrate creation, access, mask/polygon conversion, alternate constructors, and serialization."""

    # ======================= Create ============================================
    # Segmentation is always stored canonically as encoded COCO RLE, regardless
    # of input format. The plain constructor accepts an already-encoded RLE
    # dict directly; `mask_to_rle` is used here to build one from a mask (see
    # `from_mask`/`from_polygon` below for convenience constructors that do
    # this rasterization/encoding step for you).
    H, W = 720, 1280
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[0:10, 0:10] = 1
    result = datatypes.COCOObjectDetectionResult(
        image_id=7,
        category_id=1,
        image_height=H,
        image_width=W,
        score=0.95,
        bbox=[0, 0, 10, 10],
        segmentation=datatypes.COCOObjectDetectionResult.mask_to_rle(mask),
    )
    logger.info(f"Original COCOObjectDetectionResult: {result}")

    # ======================= Inspect ===========================================
    logger.info(
        f"image_id={result.image_id}, "
        f"category_id={result.category_id}, "
        f"image_height={result.image_height}, "
        f"image_width={result.image_width}, "
        f"score={result.score}"
    )
    logger.info(f"bbox={result.bbox}")
    logger.info(f"segmentation={result.segmentation}")

    # ======================= Visualize =========================================
    rr.init("coco_object_detection_result_example", spawn=True)
    datatypes.visualize(result, entity_path="/COCOObjectDetectionResult")

    # ======================= To Mask / To Polygon ================================
    as_mask = result.to_mask()
    logger.info(f"Segmentation as polygon: {result.to_polygon()}")
    logger.info(f"Segmentation as mask: shape={as_mask.shape}, dtype={as_mask.dtype}")

    # ======================= From Polygon / From Mask ============================
    # Convenience constructors that rasterize/encode a raw polygon or mask to
    # RLE before construction, instead of building the RLE dict yourself.
    polygon_result = datatypes.COCOObjectDetectionResult.from_polygon(
        image_id=7,
        category_id=2,
        image_height=H,
        image_width=W,
        score=0.82,
        polygon=[[20, 20, 25, 20, 25, 25, 20, 25]],
        bbox=[20, 20, 5, 5],
    )
    logger.info(f"Built from polygon: {polygon_result}")
    datatypes.visualize(polygon_result, entity_path="/COCOObjectDetectionResultFromPolygon")

    mask_for_result = np.zeros((H, W), dtype=np.uint8)
    mask_for_result[100:200, 150:400] = 1
    mask_result = datatypes.COCOObjectDetectionResult.from_mask(
        image_id=7,
        category_id=3,
        score=0.71,
        mask=mask_for_result,
    )
    logger.info(f"Built from mask: {mask_result}")
    datatypes.visualize(mask_result, entity_path="/COCOObjectDetectionResultFromMask")

    # ======================= Serialize / Deserialize ============================
    start = time.perf_counter()
    serialized = datatypes.serialize(result)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized COCOObjectDetectionResult: {deserialized}")
    logger.info(f"Round-trip successful: {result == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    coco_object_detection_result_example()
