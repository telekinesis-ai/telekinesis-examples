"""Demonstrates the Telekinesis COCOObjectDetectionAnnotation datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def coco_object_detection_annotation_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    # Segmentation semantics: iscrowd=False -> polygon (stored natively, not
    # converted), iscrowd=True -> encoded COCO RLE. The two representations are
    # never silently converted between each other.
    image_height, image_width = 720, 1280
    annotation = datatypes.COCOObjectDetectionAnnotation(
        id=0,
        image_id=7,
        category_id=1,
        bbox=[0, 0, 10, 10],
        area=100.0,
        iscrowd=False,
        segmentation=[[0, 2, 10, 0, 10, 10, 0, 10]],
    )
    logger.info(f"Created COCOObjectDetectionAnnotation: {annotation}")

    # ======================= Inspect ===========================================
    logger.info(f"id={annotation.id}")
    logger.info(f"image_id={annotation.image_id}")
    logger.info(f"category_id={annotation.category_id}")
    logger.info(f"bbox={annotation.bbox}")
    logger.info(f"area={annotation.area}")
    logger.info(f"iscrowd={annotation.iscrowd}")
    logger.info(f"segmentation={annotation.segmentation}")

    # ======================= Operations =========================================
    annotation_as_mask = annotation.as_mask(image_height=image_height, image_width=image_width)
    logger.info(
        f"Segmentation as mask: shape={annotation_as_mask['segmentation'].shape}, "
        f"dtype={annotation_as_mask['segmentation'].dtype}"
    )

    # `area`/`iscrowd` are labeling decisions that can't be derived from a mask,
    # so there's no dedicated "from mask" constructor (unlike
    # `COCOObjectDetectionResult.from_mask`). Build the RLE segmentation via the
    # shared `COCOSegmentationMixin.mask_to_rle` helper and pass the rest of the
    # fields directly.
    crowd_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    crowd_mask[100:200, 150:400] = 1
    crowd_annotation = datatypes.COCOObjectDetectionAnnotation(
        id=1,
        image_id=7,
        category_id=2,
        bbox=[150, 100, 250, 100],
        area=float(crowd_mask.sum()),
        iscrowd=True,
        segmentation=datatypes.COCOObjectDetectionAnnotation.mask_to_rle(crowd_mask),
    )
    logger.info(f"Crowd annotation built from a mask: {crowd_annotation}")

    crowd_annotation_as_mask = crowd_annotation.as_mask(
        image_height=image_height, image_width=image_width
    )
    logger.info(
        f"Crowd segmentation as mask: shape={crowd_annotation_as_mask['segmentation'].shape}, "
        f"dtype={crowd_annotation_as_mask['segmentation'].dtype}"
    )

    # Mixin helpers shared across all COCO segmentation datatypes.
    mask_from_rle = datatypes.COCOObjectDetectionAnnotation.rle_to_mask(crowd_annotation.segmentation)
    logger.info(f"Mask decoded via rle_to_mask: shape={mask_from_rle.shape}, dtype={mask_from_rle.dtype}")

    polygon_from_rle = datatypes.COCOObjectDetectionAnnotation.rle_to_polygon(
        crowd_annotation.segmentation
    )
    logger.info(f"Polygon decoded via rle_to_polygon: {polygon_from_rle}")

    rle_from_polygon = datatypes.COCOObjectDetectionAnnotation.polygon_to_rle(
        annotation.segmentation, height=image_height, width=image_width
    )
    logger.info(f"RLE encoded via polygon_to_rle: {rle_from_polygon}")

    # ======================= Visualize =========================================
    rr.init("coco_object_detection_annotation_example", spawn=True)
    datatypes.visualize(annotation, entity_path="/coco_object_detection_annotation")
    datatypes.visualize(crowd_annotation, entity_path="/coco_object_detection_annotation/crowd")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(annotation)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized COCOObjectDetectionAnnotation: {deserialized}")
    logger.info(f"Round-trip successful: {annotation == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    coco_object_detection_annotation_example()
