"""Demonstrates the Telekinesis COCOObjectDetectionAnnotations datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def object_detection_annotations_example():
    """Demonstrate creation, access, indexing, mask rasterization, and serialization."""

    # ======================= Create ============================================
    # Segmentation semantics: iscrowd=False -> polygon (stored natively, not
    # converted), iscrowd=True -> encoded COCO RLE. The two representations are
    # never silently converted between each other.
    H, W = 720, 1280
    annotations = datatypes.COCOObjectDetectionAnnotations(
        ids=np.array([0, 1], dtype=np.int32),
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([1, 2], dtype=np.int32),
        bboxes=np.array([[0, 0, 10, 10], [20, 20, 5, 5]], dtype=np.float32),
        areas=np.array([100.0, 25.0], dtype=np.float32),
        iscrowds=np.array([False, False]),
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
        f"areas={annotations.areas}, "
        f"iscrowds={annotations.iscrowds}"
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
        f"bbox={single.bbox}, "
        f"area={single.area}, "
        f"iscrowd={single.iscrowd}"
    )
    logger.info(f"segmentation={single.segmentation}")
    datatypes.visualize(single, entity_path="/SingleObjectDetectionAnnotation")

    # ======================= To Mask =============================================
    # Neither the single annotation nor the batch stores an image size, so
    # `to_mask`/`to_masks` take the target size explicitly. For an `iscrowd=True`
    # (RLE) entry, the passed size must match the RLE's own embedded size.
    single_mask = single.to_mask(image_height=H, image_width=W)
    logger.info(f"Segmentation 0 as mask: shape={single_mask.shape}, dtype={single_mask.dtype}")

    logger.info(
        f"All masks: shapes={[m.shape for m in annotations.to_masks(image_heights=[H, H], image_widths=[W, W])]}"
    )

    # ======================= From Mask ===========================================
    # `area`/`iscrowd` are labeling decisions that can't be derived from a mask,
    # so there's no dedicated "from mask" constructor for annotations (unlike
    # `COCOObjectDetectionResult.from_mask`). Build the RLE segmentation via the
    # shared `COCOSegmentationMixin.mask_to_rle` helper and pass the rest of the
    # fields directly.
    crowd_mask = np.zeros((H, W), dtype=np.uint8)
    crowd_mask[100:200, 150:400] = 1
    crowd_rle = datatypes.COCOObjectDetectionAnnotations.mask_to_rle(crowd_mask)
    crowd_annotation = datatypes.COCOObjectDetectionAnnotation(
        id=2,
        image_id=7,
        category_id=3,
        bbox=[150, 100, 250, 100],
        area=float(crowd_mask.sum()),
        iscrowd=True,
        segmentation=crowd_rle,
    )
    logger.info(f"Crowd annotation built from a mask: {crowd_annotation}")
    datatypes.visualize(crowd_annotation, entity_path="/CrowdObjectDetectionAnnotation")

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
