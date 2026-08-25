"""Demonstrates the Telekinesis COCOObjectDetectionAnnotations datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def coco_object_detection_annotations_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    # Segmentation semantics: iscrowd=False -> polygon (stored natively, not
    # converted), iscrowd=True -> encoded COCO RLE. 
    image_height, image_width = 720, 1280
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
    logger.info(f"Created COCOObjectDetectionAnnotations: {annotations}")

    mask_0 = np.zeros((image_height, image_width), dtype=np.uint8)
    mask_0[100:200, 150:400] = 1
    mask_1 = np.zeros((image_height, image_width), dtype=np.uint8)
    mask_1[300:340, 500:540] = 1
    annotations_from_masks = datatypes.COCOObjectDetectionAnnotations.from_masks(
        ids=np.array([2, 3], dtype=np.int32),
        image_ids=np.array([7, 7], dtype=np.int32),
        category_ids=np.array([3, 4], dtype=np.int32),
        bboxes=np.array([[150, 100, 250, 100], [500, 300, 40, 40]], dtype=np.float32),
        areas=np.array([float(mask_0.sum()), float(mask_1.sum())], dtype=np.float32),
        masks=[mask_0, mask_1],
    )
    logger.info(f"COCOObjectDetectionAnnotations created from masks: {annotations_from_masks}")

    # ======================= Inspect ===========================================
    logger.info(f"Number of annotations in batch: {len(annotations)}")
    logger.info(f"ids={annotations.ids}")
    logger.info(f"image_ids={annotations.image_ids}")
    logger.info(f"category_ids={annotations.category_ids}")
    logger.info(f"bboxes={annotations.bboxes}")
    logger.info(f"areas={annotations.areas}")
    logger.info(f"iscrowds={annotations.iscrowds}")
    logger.info(f"segmentations={annotations.segmentations}")

    # ======================= Operations =========================================
    index = 0
    single_annotation = annotations[index]
    logger.info(f"COCOObjectDetectionAnnotation at index {index}: {single_annotation}")

    sliced_annotations = annotations[0:1]
    logger.info(f"Sliced COCOObjectDetectionAnnotations: {sliced_annotations}")

    keep_mask = np.array([True, False])
    masked_annotations = annotations[keep_mask]
    logger.info(f"Masked COCOObjectDetectionAnnotations: {masked_annotations}")
    
    single_as_mask = single_annotation.as_mask(image_height=image_height, image_width=image_width)
    logger.info(
        f"Segmentation 0 as mask: shape={single_as_mask['segmentation'].shape}, "
        f"dtype={single_as_mask['segmentation'].dtype}"
    )

    annotations_as_masks = annotations.as_masks(
        image_heights=[image_height, image_height], image_widths=[image_width, image_width]
    )
    logger.info(f"All masks: shapes={[m.shape for m in annotations_as_masks['segmentations']]}")

    crowd_mask = np.zeros((image_height, image_width), dtype=np.uint8)
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

    mask_from_rle = datatypes.COCOObjectDetectionAnnotations.rle_to_mask(crowd_rle)
    logger.info(f"Mask decoded via rle_to_mask: shape={mask_from_rle.shape}, dtype={mask_from_rle.dtype}")

    polygon_from_rle = datatypes.COCOObjectDetectionAnnotations.rle_to_polygon(crowd_rle)
    logger.info(f"Polygon decoded via rle_to_polygon: {polygon_from_rle}")

    rle_from_polygon = datatypes.COCOObjectDetectionAnnotations.polygon_to_rle(
        single_annotation.segmentation, height=image_height, width=image_width
    )
    logger.info(f"RLE encoded via polygon_to_rle: {rle_from_polygon}")

    # ======================= Visualize =========================================
    rr.init("coco_object_detection_annotations_example", spawn=True)
    datatypes.visualize(annotations, entity_path="/coco_object_detection_annotations")
    datatypes.visualize(single_annotation, entity_path="/coco_object_detection_annotations/single")
    datatypes.visualize(crowd_annotation, entity_path="/coco_object_detection_annotations/crowd")
    datatypes.visualize(
        annotations_from_masks, entity_path="/coco_object_detection_annotations/from_masks"
    )

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
    coco_object_detection_annotations_example()
