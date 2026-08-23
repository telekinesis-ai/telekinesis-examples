"""Demonstrates the Telekinesis COCOObjectDetectionAnnotation datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def coco_object_detection_annotation_example():
    """Demonstrate creation, access, mask rasterization, and serialization."""

    # ======================= Create ============================================
    # Segmentation semantics: iscrowd=False -> polygon (stored natively, not
    # converted), iscrowd=True -> encoded COCO RLE. The two representations are
    # never silently converted between each other.
    H, W = 720, 1280
    annotation = datatypes.COCOObjectDetectionAnnotation(
        id=0,
        image_id=7,
        category_id=1,
        bbox=[0, 0, 10, 10],
        area=100.0,
        iscrowd=False,
        segmentation=[[0, 2, 10, 0, 10, 10, 0, 10]],
    )
    logger.info(f"Original COCOObjectDetectionAnnotation: {annotation}")

    # ======================= Inspect ===========================================
    logger.info(
        f"id={annotation.id}, "
        f"image_id={annotation.image_id}, "
        f"category_id={annotation.category_id}, "
        f"bbox={annotation.bbox}, "
        f"area={annotation.area}, "
        f"iscrowd={annotation.iscrowd}"
    )
    logger.info(f"segmentation={annotation.segmentation}")

    # ======================= Visualize =========================================
    rr.init("coco_object_detection_annotation_example", spawn=True)
    datatypes.visualize(annotation, entity_path="/COCOObjectDetectionAnnotation")

    # ======================= To Mask =============================================
    # `to_mask` takes the target size explicitly since the annotation doesn't
    # store one. For an `iscrowd=True` (RLE) entry, the passed size must match
    # the RLE's own embedded size.
    mask = annotation.to_mask(image_height=H, image_width=W)
    logger.info(f"Segmentation as mask: shape={mask.shape}, dtype={mask.dtype}")

    # ======================= From Mask ===========================================
    # `area`/`iscrowd` are labeling decisions that can't be derived from a mask,
    # so there's no dedicated "from mask" constructor (unlike
    # `COCOObjectDetectionResult.from_mask`). Build the RLE segmentation via the
    # shared `COCOSegmentationMixin.mask_to_rle` helper and pass the rest of the
    # fields directly.
    crowd_mask = np.zeros((H, W), dtype=np.uint8)
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
    datatypes.visualize(crowd_annotation, entity_path="/CrowdObjectDetectionAnnotation")

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
