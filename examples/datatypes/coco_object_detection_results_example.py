"""Demonstrates the Telekinesis COCOObjectDetectionResults datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def coco_object_detection_results_example():
    """Demonstrate creation, access, indexing, mask/polygon conversion, alternate constructors, and serialization."""

    # ======================= Create ============================================
    # Segmentation is always stored canonically as encoded COCO RLE, regardless
    # of input format. The plain constructor accepts a list of already-encoded
    # RLE dicts directly; `mask_to_rle` is used here to build them from masks
    # (see `from_masks`/`from_polygons` below for batch convenience
    # constructors that do this rasterization/encoding step for you).
    H, W = 720, 1280
    mask0 = np.zeros((H, W), dtype=np.uint8)
    mask0[0:10, 0:10] = 1
    mask1 = np.zeros((H, W), dtype=np.uint8)
    mask1[20:25, 20:25] = 1
    bbox_results = datatypes.COCOObjectDetectionResults(
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        image_heights=np.array([H, H], dtype=np.int32),
        image_widths=np.array([W, W], dtype=np.int32),
        scores=np.array([0.95, 0.82], dtype=np.float32),
        bboxes=np.array([[0, 0, 10, 10], [20, 20, 5, 5]], dtype=np.float32),
        segmentations=[
            datatypes.COCOObjectDetectionResults.mask_to_rle(mask0),
            datatypes.COCOObjectDetectionResults.mask_to_rle(mask1),
        ],
    )
    logger.info(f"Original COCOObjectDetectionResults: {bbox_results}")

    # ======================= Inspect ===========================================
    logger.info(f"Number of results in batch: {len(bbox_results)}")
    logger.info(
        f"image_ids={bbox_results.image_ids}, "
        f"category_ids={bbox_results.category_ids}, "
        f"image_heights={bbox_results.image_heights}, "
        f"image_widths={bbox_results.image_widths}, "
        f"scores={bbox_results.scores}"
    )
    logger.info(f"bboxes={bbox_results.bboxes}")
    logger.info(f"segmentations={bbox_results.segmentations}")

    # ======================= Visualize =========================================
    rr.init("coco_object_detection_results_example", spawn=True)
    datatypes.visualize(bbox_results, entity_path="/COCOObjectDetectionResults")

    # ======================= Index =============================================
    index = 0
    first_result = bbox_results[index]
    logger.info(
        f"ObjectDetectionResult at index {index}: "
        f"image_id={first_result.image_id}, "
        f"category_id={first_result.category_id}, "
        f"bbox={first_result.bbox}, "
        f"score={first_result.score}"
    )
    datatypes.visualize(first_result, entity_path="/COCOObjectDetectionResults/FirstResult")

    # ======================= To Mask / To Polygon ================================
    as_mask = first_result.to_mask()
    logger.info(f"Segmentation 0 as RLE: {first_result.segmentation}")
    logger.info(f"Segmentation 0 as polygon: {first_result.to_polygon()}")
    logger.info(f"Segmentation 0 as mask: shape={as_mask.shape}, dtype={as_mask.dtype}")

    # `to_masks`/`to_polygons` are the batch equivalents.
    logger.info(f"All masks: shapes={[m.shape for m in bbox_results.to_masks()]}")
    logger.info(f"All polygons: {bbox_results.to_polygons()}")

    # ======================= From Polygon / From Mask ============================
    # Convenience constructors that rasterize/encode raw polygons or masks to
    # RLE before construction, instead of building the RLE dicts yourself.
    # `from_masks` also derives `image_heights`/`image_widths` from each
    # mask's shape.
    bbox_segmentation_results = datatypes.COCOObjectDetectionResults.from_polygons(
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        image_heights=np.array([H, H], dtype=np.int32),
        image_widths=np.array([W, W], dtype=np.int32),
        scores=np.array([0.95, 0.82], dtype=np.float32),
        bboxes=np.array([[0, 0, 10, 10], [20, 20, 5, 5]], dtype=np.float32),
        polygons=[
            [[0, 2, 10, 0, 10, 10, 0, 10]],
            [[22, 20, 25, 20, 25, 25, 20, 25]],
        ],
    )
    logger.info(f"Built from polygons: {bbox_segmentation_results}")
    datatypes.visualize(
        bbox_segmentation_results, entity_path="/COCOObjectDetectionResultsFromPolygons"
    )

    segmentation_only_results = datatypes.COCOObjectDetectionResults.from_masks(
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        scores=np.array([0.95, 0.82], dtype=np.float32),
        masks=[mask0, mask1],
    )
    logger.info(f"Built from masks: {segmentation_only_results}")
    datatypes.visualize(
        segmentation_only_results, entity_path="/COCOObjectDetectionResultsFromMasks"
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(bbox_results)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized COCOObjectDetectionResults: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == bbox_results}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    coco_object_detection_results_example()
