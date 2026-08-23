"""Demonstrates the Telekinesis COCOObjectDetectionResult datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def coco_object_detection_result_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    # Segmentation is always stored canonically as encoded COCO RLE, regardless
    # of input format. The plain constructor accepts an already-encoded RLE
    # dict directly; `mask_to_rle` builds one from a mask here.
    image_height, image_width = 720, 1280
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    mask[0:10, 0:10] = 1
    result = datatypes.COCOObjectDetectionResult(
        image_id=7,
        category_id=1,
        image_height=image_height,
        image_width=image_width,
        score=0.95,
        bbox=[0, 0, 10, 10],
        segmentation=datatypes.COCOObjectDetectionResult.mask_to_rle(mask),
    )
    logger.info(f"Created COCOObjectDetectionResult: {result}")

    result_from_polygon = datatypes.COCOObjectDetectionResult.from_polygon(
        image_id=7,
        category_id=2,
        image_height=image_height,
        image_width=image_width,
        score=0.82,
        polygon=[[20, 20, 25, 20, 25, 25, 20, 25]],
        bbox=[20, 20, 5, 5],
    )
    logger.info(f"COCOObjectDetectionResult created from polygon: {result_from_polygon}")

    mask_for_result = np.zeros((image_height, image_width), dtype=np.uint8)
    mask_for_result[100:200, 150:400] = 1
    result_from_mask = datatypes.COCOObjectDetectionResult.from_mask(
        image_id=7,
        category_id=3,
        score=0.71,
        mask=mask_for_result,
    )
    logger.info(f"COCOObjectDetectionResult created from mask: {result_from_mask}")

    # ======================= Inspect ===========================================
    logger.info(f"image_id={result.image_id}")
    logger.info(f"category_id={result.category_id}")
    logger.info(f"image_height={result.image_height}")
    logger.info(f"image_width={result.image_width}")
    logger.info(f"score={result.score}")
    logger.info(f"bbox={result.bbox}")
    logger.info(f"segmentation={result.segmentation}")

    # ======================= Operations =========================================
    result_as_mask = result.as_mask()
    logger.info(
        f"Segmentation as mask: shape={result_as_mask['segmentation'].shape}, "
        f"dtype={result_as_mask['segmentation'].dtype}"
    )

    result_as_polygon = result.as_polygon()
    logger.info(f"Segmentation as polygon: {result_as_polygon['segmentation']}")

    # Mixin helpers shared across all COCO segmentation datatypes.
    mask_from_rle = datatypes.COCOObjectDetectionResult.rle_to_mask(result.segmentation)
    logger.info(f"Mask decoded via rle_to_mask: shape={mask_from_rle.shape}, dtype={mask_from_rle.dtype}")

    polygon_from_rle = datatypes.COCOObjectDetectionResult.rle_to_polygon(result.segmentation)
    logger.info(f"Polygon decoded via rle_to_polygon: {polygon_from_rle}")

    rle_from_polygon = datatypes.COCOObjectDetectionResult.polygon_to_rle(
        [[20, 20, 25, 20, 25, 25, 20, 25]], height=image_height, width=image_width
    )
    logger.info(f"RLE encoded via polygon_to_rle: {rle_from_polygon}")

    # ======================= Visualize =========================================
    rr.init("coco_object_detection_result_example", spawn=True)
    datatypes.visualize(result, entity_path="/coco_object_detection_result")
    datatypes.visualize(result_from_polygon, entity_path="/coco_object_detection_result/from_polygon")
    datatypes.visualize(result_from_mask, entity_path="/coco_object_detection_result/from_mask")

    # ======================= Serialize / Deserialize ===========================
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
