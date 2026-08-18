"""Demonstrates the Telekinesis COCOObjectDetectionResults datatype."""

import time

import rerun as rr
import numpy as np
from loguru import logger

from telekinesis import datatypes

def coco_object_detection_results_example():
    """Demonstrate creation, indexing, segmentation conversion, visualization, and serialization."""

    # ======================= Case 1: Bounding Boxes Only =======================
    H, W = 720, 1280
    bbox_results = datatypes.COCOObjectDetectionResults(
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        image_heights=np.array([H, H], dtype=np.int32),
        image_widths=np.array([W, W], dtype=np.int32),
        scores=np.array([0.95, 0.82], dtype=np.float32),
        bboxes=np.array(
            [[0, 0, 10, 10], [20, 20, 5, 5]],
            dtype=np.float32,
        ),
    )
    logger.info(f"Original COCOObjectDetectionResults: {bbox_results}")

    # ======================= Inspect ===========================================
    image_ids = bbox_results.image_ids
    category_ids = bbox_results.category_ids
    image_heights = bbox_results.image_heights
    image_widths = bbox_results.image_widths
    scores = bbox_results.scores
    bboxes = bbox_results.bboxes
    segmentations = bbox_results.segmentations

    logger.info(f"Number of results in batch: {len(bboxes)}")
    logger.info(
        f"image_ids={image_ids}, "
        f"category_ids={category_ids}, "
        f"image_heights={image_heights}, "
        f"image_widths={image_widths}, "
        f"scores={scores}"
    )
    logger.info(f"Bboxes: {bboxes}")
    logger.info(f"Segmentations: {segmentations}")

    # ======================= Index =============================================
    index = 0
    detection_at_index = bbox_results[index]
    image_id = detection_at_index.image_id
    category_id = detection_at_index.category_id
    bbox = detection_at_index.bbox
    score = detection_at_index.score

    logger.info(
        f"ObjectDetectionResult at index {index}: "
        f"image_id={image_id}, "
        f"category_id={category_id}, "
        f"bbox={bbox}, "
        f"score={score}"
    )

    # ======================= Visualize =========================================
    rr.init("coco_object_detection_results_example", spawn=True)
    datatypes.visualize(bbox_results, entity_path="/COCOObjectDetectionResults")

    first_result = bbox_results[0]
    datatypes.visualize(first_result, entity_path="/COCOObjectDetectionResults/FirstResult")

    # ======================= Case 2: Bbox + Segmentation =======================
    bbox_segmentation_results = datatypes.COCOObjectDetectionResults(
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        image_heights=np.array([H, H], dtype=np.int32),
        image_widths=np.array([W, W], dtype=np.int32),
        scores=np.array([0.95, 0.82], dtype=np.float32),
        bboxes=np.array([[0, 0, 10, 10], [20, 20, 5, 5]], dtype=np.float32),
        segmentations=[
            [[0, 2, 10, 0, 10, 10, 0, 10]],
            [[22, 20, 25, 20, 25, 25, 20, 25]],
        ],
    )
    datatypes.visualize(
        bbox_segmentation_results, entity_path="/COCOObjectDetectionResultsWithSegmentation"
    )

    # ======================= Case 3: Segmentation Only =========================
    segmentation_only_results = datatypes.COCOObjectDetectionResults(
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        image_heights=np.array([H, H], dtype=np.int32),
        image_widths=np.array([W, W], dtype=np.int32),
        scores=np.array([0.95, 0.82], dtype=np.float32),
        segmentations=[
            [[0, 2, 10, 0, 10, 10, 0, 10]],
            [[22, 20, 25, 20, 25, 25, 20, 25]],
        ],
    )
    datatypes.visualize(
        segmentation_only_results, entity_path="/COCOObjectDetectionResultsWithOnlySegmentation"
    )

    # ======================= Convert Segmentation ==============================
    as_rle = bbox_segmentation_results.segmentations[0]
    as_polygon = bbox_segmentation_results.get_segmentation(0, "polygon")
    as_mask = bbox_segmentation_results.get_segmentation(0, "mask")
    logger.info(f"Segmentation 0 as RLE: {as_rle}")
    logger.info(f"Segmentation 0 as polygon: {as_polygon}")
    logger.info(f"Segmentation 0 as mask: shape={as_mask.shape}, dtype={as_mask.dtype}")

    mask_back_to_rle = datatypes.COCOObjectDetectionResults.convert_segmentation(
        as_mask, "mask", "rle"
    )
    polygon_to_mask = datatypes.COCOObjectDetectionResults.convert_segmentation(
        as_polygon,
        datatypes.SegmentationFormat.POLYGON,
        datatypes.SegmentationFormat.MASK,
        height=H,
        width=W,
    )
    logger.info(f"Mask converted back to RLE: {mask_back_to_rle}")
    logger.info(
        f"Polygon converted to mask: shape={polygon_to_mask.shape}, dtype={polygon_to_mask.dtype}"
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
