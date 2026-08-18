"""Demonstrates the Telekinesis COCOObjectDetectionAnnotations datatype."""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes

def object_detection_annotations_example():
    """Demonstrate creation, access, indexing, segmentation format conversion, construction from masks, and serialization."""

    # ======================= Create ============================================
    H, W = 720, 1280
    annotations = datatypes.COCOObjectDetectionAnnotations(
        ids=np.array([0, 1], dtype=np.int32),
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        bboxes=np.array([[0, 0, 10, 10], [20, 20, 5, 5]], dtype=np.float32),
        image_heights=np.array([H, H], dtype=np.int32),
        image_widths=np.array([W, W], dtype=np.int32),
        segmentations=[
            [[0, 2, 10, 0, 10, 10, 0, 10]],
            [[22, 20, 25, 20, 25, 25, 20, 25]],
        ],
    )
    logger.info(f"Original COCOObjectDetectionAnnotations: {annotations}")

    # ======================= Inspect ===========================================
    logger.info(f"Number of annotations in batch: {len(annotations)}")
    logger.info(
        f"ids={annotations.ids}, "
        f"image_ids={annotations.image_ids}, "
        f"category_ids={annotations.category_ids}, "
        f"image_heights={annotations.image_heights}, "
        f"image_widths={annotations.image_widths}"
    )
    logger.info(f"bboxes={annotations.bboxes}")
    logger.info(f"segmentations={annotations.segmentations}")

    # ======================= Visualize =========================================
    rr.init("object_detection_example", spawn=True)
    datatypes.visualize(annotations, entity_path="/COCOObjectDetectionAnnotations")

    # ======================= Index =============================================
    index = 0
    single = annotations[index]
    logger.info(f"Single ObjectDetectionAnnotation at index {index}: {single}")
    logger.info(
        f"id={single.id}, "
        f"image_id={single.image_id}, "
        f"category_id={single.category_id}, "
        f"bbox={single.bbox}"
    )
    logger.info(f"segmentation={single.segmentation}")
    datatypes.visualize(single, entity_path="/SingleObjectDetectionAnnotation")

    # ======================= Convert ===========================================
    as_rle = annotations.segmentations[0]
    as_polygon = annotations.get_segmentation(0, "polygon")
    as_mask = annotations.get_segmentation(0, "mask")
    logger.info(f"Segmentation 0 as RLE: {as_rle}")
    logger.info(f"Segmentation 0 as polygon: {as_polygon}")
    logger.info(f"Segmentation 0 as mask: shape={as_mask.shape}, dtype={as_mask.dtype}")

    mask_back_to_rle = datatypes.COCOObjectDetectionAnnotations.convert_segmentation(
        as_mask, "mask", "rle"
    )
    logger.info(f"Mask converted back to RLE: {mask_back_to_rle}")

    # ======================= From Masks ========================================
    masks = [
        annotations.get_segmentation(i, datatypes.SegmentationFormat.MASK)
        for i in range(len(annotations))
    ]
    from_masks = datatypes.COCOObjectDetectionAnnotations.from_binary_masks(
        ids=annotations.ids,
        image_ids=annotations.image_ids,
        category_ids=annotations.category_ids,
        bboxes=annotations.bboxes,
        masks=masks,
    )
    logger.info(f"Built from binary masks: {from_masks}")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(annotations)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized COCOObjectDetectionAnnotations: {deserialized}")
    logger.info(f"Round-trip successful: {annotations == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    object_detection_annotations_example()
