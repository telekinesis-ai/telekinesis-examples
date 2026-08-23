"""Demonstrates the Telekinesis COCOObjectDetectionResults datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def coco_object_detection_results_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    # Segmentation is always stored canonically as encoded COCO RLE, regardless
    # of input format. `mask_to_rle` builds one from a mask here.
    image_height, image_width = 720, 1280
    mask_0 = np.zeros((image_height, image_width), dtype=np.uint8)
    mask_0[0:10, 0:10] = 1
    mask_1 = np.zeros((image_height, image_width), dtype=np.uint8)
    mask_1[20:25, 20:25] = 1
    results = datatypes.COCOObjectDetectionResults(
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        image_heights=np.array([image_height, image_height], dtype=np.int32),
        image_widths=np.array([image_width, image_width], dtype=np.int32),
        scores=np.array([0.95, 0.82], dtype=np.float32),
        bboxes=np.array([[0, 0, 10, 10], [20, 20, 5, 5]], dtype=np.float32),
        segmentations=[
            datatypes.COCOObjectDetectionResults.mask_to_rle(mask_0),
            datatypes.COCOObjectDetectionResults.mask_to_rle(mask_1),
        ],
    )
    logger.info(f"Created COCOObjectDetectionResults: {results}")

    results_from_polygons = datatypes.COCOObjectDetectionResults.from_polygons(
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        image_heights=np.array([image_height, image_height], dtype=np.int32),
        image_widths=np.array([image_width, image_width], dtype=np.int32),
        scores=np.array([0.95, 0.82], dtype=np.float32),
        bboxes=np.array([[0, 0, 10, 10], [20, 20, 5, 5]], dtype=np.float32),
        polygons=[
            [[0, 2, 10, 0, 10, 10, 0, 10]],
            [[22, 20, 25, 20, 25, 25, 20, 25]],
        ],
    )
    logger.info(f"COCOObjectDetectionResults created from polygons: {results_from_polygons}")

    results_from_masks = datatypes.COCOObjectDetectionResults.from_masks(
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        scores=np.array([0.95, 0.82], dtype=np.float32),
        masks=[mask_0, mask_1],
    )
    logger.info(f"COCOObjectDetectionResults created from masks: {results_from_masks}")

    # ======================= Inspect ===========================================
    logger.info(f"Number of results in batch: {len(results)}")
    logger.info(f"image_ids={results.image_ids}")
    logger.info(f"category_ids={results.category_ids}")
    logger.info(f"image_heights={results.image_heights}")
    logger.info(f"image_widths={results.image_widths}")
    logger.info(f"scores={results.scores}")
    logger.info(f"bboxes={results.bboxes}")
    logger.info(f"segmentations={results.segmentations}")

    # ======================= Operations =========================================
    index = 0
    first_result = results[index]
    logger.info(f"COCOObjectDetectionResult at index {index}: {first_result}")

    sliced_results = results[0:1]
    logger.info(f"Sliced COCOObjectDetectionResults: {sliced_results}")

    keep_mask = np.array([True, False])
    masked_results = results[keep_mask]
    logger.info(f"Masked COCOObjectDetectionResults: {masked_results}")

    results_as_masks = results.as_masks()
    logger.info(f"Segmentations as masks: shapes={[m.shape for m in results_as_masks['segmentations']]}")

    results_as_polygons = results.as_polygons()
    logger.info(f"Segmentations as polygons: {results_as_polygons['segmentations']}")

    # Mixin helpers shared across all COCO segmentation datatypes.
    mask_from_rle = datatypes.COCOObjectDetectionResults.rle_to_mask(results.segmentations[0])
    logger.info(f"Mask 0 decoded via rle_to_mask: shape={mask_from_rle.shape}, dtype={mask_from_rle.dtype}")

    polygon_from_rle = datatypes.COCOObjectDetectionResults.rle_to_polygon(results.segmentations[0])
    logger.info(f"Polygon 0 decoded via rle_to_polygon: {polygon_from_rle}")

    rle_from_polygon = datatypes.COCOObjectDetectionResults.polygon_to_rle(
        [[0, 2, 10, 0, 10, 10, 0, 10]], height=image_height, width=image_width
    )
    logger.info(f"RLE encoded via polygon_to_rle: {rle_from_polygon}")

    # ======================= Visualize =========================================
    rr.init("coco_object_detection_results_example", spawn=True)
    datatypes.visualize(results, entity_path="/coco_object_detection_results")
    datatypes.visualize(first_result, entity_path="/coco_object_detection_results/first_result")
    datatypes.visualize(
        results_from_polygons, entity_path="/coco_object_detection_results/from_polygons"
    )
    datatypes.visualize(results_from_masks, entity_path="/coco_object_detection_results/from_masks")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(results)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized COCOObjectDetectionResults: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == results}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    coco_object_detection_results_example()
