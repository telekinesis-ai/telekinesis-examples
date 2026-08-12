"""
Example script to demonstrate usage of COCOObjectDetectionResults datatype.
"""

import time

import rerun as rr
import numpy as np
from loguru import logger

from telekinesis import datatypes


def coco_object_detection_results_example():
    """
    Example function to demonstrate usage of COCOObjectDetectionResults datatype.
        - Create a COCOObjectDetectionResults batch
        - Visualize the batch and individual results

    """
    H, W = 720, 1280
    # Note: ObjectDetectionResults can be defined in three ways:
    # 1. Only bounding boxes
    # 2. Bounding boxes + segmentation (polygon or RLE)
    # 3. Only segmentation (polygon or RLE)

    # Case 1: Bounding boxes only
    my_bbox_results = datatypes.COCOObjectDetectionResults(
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

    logger.info(f"Original COCOObjectDetectionResults: {my_bbox_results}")

    # Access uderlying data
    image_ids = my_bbox_results.image_ids
    category_ids = my_bbox_results.category_ids
    image_heights = my_bbox_results.image_heights
    image_widths = my_bbox_results.image_widths
    scores = my_bbox_results.scores
    bboxes = my_bbox_results.bboxes
    segmentations = my_bbox_results.segmentations

    logger.info(f"Number of results in batch: {len(my_bbox_results.bboxes)}")
    logger.info(f"Underlying image_ids data: {image_ids}")
    logger.info(f"Underlying category_ids: {category_ids}")
    logger.info(f"Underlying image_heights: {image_heights}")
    logger.info(f"Underlying image_widths: {image_widths}")
    logger.info(f"Underlying scores: {scores}")
    logger.info(f"Underlying bboxes: {bboxes}")
    logger.info(f"Underlying segmentations: {segmentations}")

    # Access individual results via indexing
    index = 0
    my_detection_result_at_index = my_bbox_results[index]
    logger.info(f"ObjectDetectionResult at index {index}: {my_detection_result_at_index}")

    my_detection_result_at_index_image_id = my_detection_result_at_index.image_id
    my_detection_result_at_index_category_id = my_detection_result_at_index.category_id
    my_detection_result_at_index_bbox = my_detection_result_at_index.bbox
    my_detection_result_at_index_score = my_detection_result_at_index.score
    logger.info(
        f"ObjectDetectionResult at index {index}: "
        f"image_id={my_detection_result_at_index_image_id}, "
        f"category_id={my_detection_result_at_index_category_id}, "
        f"bbox={my_detection_result_at_index_bbox}, "
        f"score={my_detection_result_at_index_score}"
    )

    # Visualize the batch of results
    rr.init("coco_object_detection_results_example", spawn=True)
    datatypes.visualize(my_bbox_results, entity_path="/COCOObjectDetectionResults")

    # Access individual results via indexing, returns COCOObjectDetectionResult object
    result = my_bbox_results[0]
    datatypes.visualize(
        result,
        entity_path="/COCOObjectDetectionResults/FirstResult"
    )

    # Case 2: With bounding boxes and segmentation (polygon)
    my_bbox_segmenetation_results = datatypes.COCOObjectDetectionResults(
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        image_heights=np.array([H, H], dtype=np.int32),
        image_widths=np.array([W, W], dtype=np.int32),
        scores=np.array([0.95, 0.82], dtype=np.float32),
        bboxes=np.array([[0, 0, 10, 10], [20, 20, 5, 5]], dtype=np.float32),
        segmentations=[
            [[0, 2, 10, 0, 10, 10, 0, 10]],  # polygon for bbox [0,0,10,10]
            [[22, 20, 25, 20, 25, 25, 20, 25]],  # polygon for bbox [20,20,5,5]
        ],
    )
    datatypes.visualize(
        my_bbox_segmenetation_results, entity_path="/COCOObjectDetectionResultsWithSegmentation"
    )

    # Case 3: With only segmentation (polygon)
    my_object_detection_results_3 = datatypes.COCOObjectDetectionResults(
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        image_heights=np.array([H, H], dtype=np.int32),
        image_widths=np.array([W, W], dtype=np.int32),
        scores=np.array([0.95, 0.82], dtype=np.float32),
        segmentations=[
            [[0, 2, 10, 0, 10, 10, 0, 10]],  # polygon for bbox [0,0,10,10]
            [[22, 20, 25, 20, 25, 25, 20, 25]],  # polygon for bbox [20,20,5,5]
        ],
    )
    datatypes.visualize(
        my_object_detection_results_3, entity_path="/COCOObjectDetectionResultsWithOnlySegmentation"
    )

    # Convert to different segmentation formats
    # Segmentations are stored internally as canonical RLE; get_segmentation
    # converts a stored entry to "rle", "polygon", or "mask" on demand.
    as_rle = my_bbox_segmenetation_results.segmentations[0]
    as_polygon = my_bbox_segmenetation_results.get_segmentation(0, "polygon")
    as_mask = my_bbox_segmenetation_results.get_segmentation(0, "mask")
    logger.info(f"Segmentation 0 as RLE: {as_rle}")
    logger.info(f"Segmentation 0 as polygon: {as_polygon}")
    logger.info(f"Segmentation 0 as mask: shape={as_mask.shape}, dtype={as_mask.dtype}")

    # convert_segmentation is the same conversion, usable standalone on any
    # segmentation without needing an instance to look it up from
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

    # Serialize to pyarrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_bbox_results)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized COCOObjectDetectionResults: {deserialized}")
    logger.info(
        f"Deserialized COCOObjectDetectionResults data matches Original: {deserialized == my_bbox_results}"
    )

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    coco_object_detection_results_example()
