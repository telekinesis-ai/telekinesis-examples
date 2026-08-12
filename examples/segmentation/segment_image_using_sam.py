"""
Demonstrates segmentation using SAM (Segment Anything Model).

This example:
- Downloads an example image.
- Segments objects using SAM with bounding box prompts.
- Processes and visualizes the segmentation masks using Rerun.
"""

import numpy as np
import requests
import cv2
from loguru import logger
import rerun as rr
import rerun.blueprint as rrb
from pycocotools import mask as mask_utils
from datatypes import datatypes
from telekinesis import cornea, pupil

def segment_image_using_sam_example():
    """Segments an image using the Segment Anything Model (SAM)."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/pedestrians.jpg"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    bboxes = [[40, 70, 330, 414]]
    annotations = cornea.segment_image_using_sam(
        image=image, bboxes=bboxes, mask_threshold=0.5
    )
    annotations_list = annotations.to_list()
    logger.success(f"Segmented {len(annotations_list)} objects.")

    # ===================== Visualization  (Optional) ======================
    visualize(image, bboxes, annotations_list)


def fetch_image(image_url: str) -> datatypes.Image:
    """
    Downloads an image from a given URL and returns it as a telekinesis.datatypes.Image object.
    """
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image_bgr = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR,
    )
    image = datatypes.Image(image=image_bgr, color_model="BGR")
    image = pupil.convert_image_color_space(
        image, source_color_space="BGR", target_color_space="RGB"
    )
    logger.success(f"Loaded image from {image_url}")
    return image

def visualize(image: datatypes.Image, bboxes: list, annotations_list: list) -> None:
    """Visualizes the original image and SAM segmentation results using Rerun."""
    rr.init("cornea_sam_segmentation", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Input", origin="input"),
                rrb.Spatial2DView(name="Bboxes & Segments", origin="segmented"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    in_np_image = image.to_numpy()
    h, w = in_np_image.shape[:2]
    rr.log("input", rr.Image(in_np_image))
    rr.log("segmented/image", rr.Image(in_np_image))

    masks = []
    segmentation_img = np.zeros((h, w), dtype=np.uint16)
    ann_bboxes, class_ids = [], []

    for idx, ann in enumerate(annotations_list):
        label = idx + 1
        mask_i = np.zeros((h, w), dtype=np.uint8)
        if "mask" in ann and isinstance(ann["mask"], np.ndarray):
            m = ann["mask"]
            mask_i = (m > 0.5).astype(np.uint8) if m.dtype.kind in ("f", "b") else (m > 0).astype(np.uint8)
        elif "segmentation" in ann and ann["segmentation"]:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                mask_dec = mask_utils.decode(seg)
                if mask_dec.ndim == 3:
                    mask_dec = mask_dec[:, :, 0]
                mask_i = (mask_dec > 0).astype(np.uint8)
            elif isinstance(seg, list) and len(seg) > 0:
                temp = np.zeros((h, w), dtype=np.uint8)
                for poly in (seg if isinstance(seg[0], list) else [seg]):
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(temp, [pts], 1)
                mask_i = (temp > 0).astype(np.uint8)

        if mask_i.sum() == 0:
            continue

        masks.append(mask_i)
        segmentation_img[mask_i > 0] = label
        if "bbox" in ann:
            ann_bboxes.append(list(ann["bbox"]))
            class_ids.append(label)

    rr.log("segmented/masks", rr.SegmentationImage(segmentation_img))
    if ann_bboxes:
        rr.log("segmented/boxes", rr.Boxes2D(
            array=np.asarray(ann_bboxes, dtype=np.float32),
            array_format=rr.Box2DFormat.XYWH,
            class_ids=np.asarray(class_ids, dtype=np.int32),
        ))

if __name__ == "__main__":
    segment_image_using_sam_example()
