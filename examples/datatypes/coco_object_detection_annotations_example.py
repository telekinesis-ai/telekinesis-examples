"""
Example script to demonstrate usage of COCOObjectDetectionAnnotations datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def object_detection_annotations_example():
    """
    Example function to demonstrate usage of COCOObjectDetectionAnnotations datatype.
     - Create an COCOObjectDetectionAnnotations data
     - Print the original data
    """
    # Create an COCOObjectDetectionAnnotations data
    H, W = 720, 1280
    my_annotations = datatypes.COCOObjectDetectionAnnotations(
        ids=np.array([0, 1], dtype=np.int32),
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        bboxes=np.array([[0, 0, 10, 10], [20, 20, 5, 5]], dtype=np.float32),
        image_heights=np.array([H, H], dtype=np.int32),
        image_widths=np.array([W, W], dtype=np.int32),
        segmentations=[
            [[0, 2, 10, 0, 10, 10, 0, 10]],  # polygon for bbox [0,0,10,10]
            [[22, 20, 25, 20, 25, 25, 20, 25]],  # polygon for bbox [20,20,5,5]
        ],
    )
    logger.info(f"Original COCOObjectDetectionAnnotations: {my_annotations}")

    # Access the grouped underlying annotations data
    my_annotations_ids = my_annotations.ids
    my_annotations_image_ids = my_annotations.image_ids
    my_annotations_category_ids = my_annotations.category_ids
    my_annotations_bboxes = my_annotations.bboxes
    my_annotations_image_heights = my_annotations.image_heights
    my_annotations_image_widths = my_annotations.image_widths
    my_annotations_segmentations = my_annotations.segmentations

    logger.info(f"Number of annotations in batch: {len(my_annotations)}")
    logger.info(f"Underlying ids data: {my_annotations_ids}")
    logger.info(f"Underlying image_ids data: {my_annotations_image_ids}")
    logger.info(f"Underlying category_ids data: {my_annotations_category_ids}")
    logger.info(f"Underlying bboxes data: {my_annotations_bboxes}")
    logger.info(f"Underlying image_heights data: {my_annotations_image_heights}")
    logger.info(f"Underlying image_widths data: {my_annotations_image_widths}")
    logger.info(f"Underlying segmentations data: {my_annotations_segmentations}")

    logger.info("Visualizing with Rerun...")
    rr.init("object_detection_example", spawn=True)
    datatypes.visualize(
        my_annotations, 
        entity_path="/COCOObjectDetectionAnnotations"
    )

    # Indexing with an int returns a new COCOObjectDetectionAnnotation
    index = 0
    my_single_annotation = my_annotations[index]
    logger.info(
        f"Single ObjectDetectionAnnotation at index {index}: "
        f"{my_single_annotation}"
    )
    
    my_single_annotation_id = my_single_annotation.id
    my_single_annotation_image_id = my_single_annotation.image_id
    my_single_annotation_category_id = my_single_annotation.category_id
    my_single_annotation_bbox = my_single_annotation.bbox

    logger.info(f"Single annotation id: {my_single_annotation_id}")
    logger.info(f"Single annotation image_id: {my_single_annotation_image_id}")   
    logger.info(f"Single annotation category_id: {my_single_annotation_category_id}")
    logger.info(f"Single annotation bbox: {my_single_annotation_bbox}")
    logger.info(f"Single annotation segmentation: {my_single_annotation.segmentation}")

    datatypes.visualize(
        my_single_annotation, 
        entity_path="/SingleObjectDetectionAnnotation"
    )

    # Convert segmentation to different formats
    as_rle = my_annotations.segmentations[0]
    as_polygon = my_annotations.get_segmentation(0, "polygon")
    as_mask = my_annotations.get_segmentation(0, "mask")
    logger.info(f"Segmentation 0 as RLE: {as_rle}")
    logger.info(f"Segmentation 0 as polygon: {as_polygon}")
    logger.info(f"Segmentation 0 as mask: shape={as_mask.shape}, dtype={as_mask.dtype}")

    # Convert_segmentation is the same conversion, usable standalone on any
    mask_back_to_rle = datatypes.COCOObjectDetectionAnnotations.convert_segmentation(
        as_mask, "mask", "rle"
    )
    logger.info(f"Mask converted back to RLE: {mask_back_to_rle}")

    # Create from binary masks
    masks = [
        my_annotations.get_segmentation(i, datatypes.SegmentationFormat.MASK)
        for i in range(len(my_annotations))
    ]
    from_masks = datatypes.COCOObjectDetectionAnnotations.from_binary_masks(
        ids=my_annotations.ids,
        image_ids=my_annotations.image_ids,
        category_ids=my_annotations.category_ids,
        bboxes=my_annotations.bboxes,
        masks=masks,
    )
    logger.info(f"Built from binary masks: {from_masks}")

    # Serialize to pyarrow and back
    serialization_start_time = time.perf_counter()
    serialized = datatypes.serialize(my_annotations)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(f"Deserialized COCOObjectDetectionAnnotations: {deserialized}")
    logger.info(f"Deserialized COCOObjectDetectionAnnotations data matches Original: {deserialized == my_annotations}")

    logger.info(f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms")
    logger.info(f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms")


if __name__ == "__main__":
    object_detection_annotations_example()
