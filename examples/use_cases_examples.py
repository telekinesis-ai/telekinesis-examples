"""Use case examples for Telekinesis SDK packages.

This module provides a CLI and registry for running end-to-end use-case examples
that demonstrate how to use the main Telekinesis packages in real-world scenarios.

Each example is named after a use case and is grouped by package.

Available packages for use-case examples:
    - cornea (computer vision, segmentation, detection, etc.)

Currently, only use cases for the `cornea` package are implemented/focused.
Other package use cases will be added in the future.

Usage:
    python use_cases_examples.py --list
    python use_cases_examples.py --example <example_name>
"""
import argparse
import difflib
import pathlib
from loguru import logger
import numpy as np
from pycocotools import mask as mask_utils
import cv2

import rerun as rr
from rerun import blueprint as rrb

from datatypes import io
from telekinesis import cornea

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = ROOT_DIR / "telekinesis-data"
# ================================================================================
#                           Cornea
# ================================================================================


# Conveyor Tracking
def conveyor_tracking_using_sam_example():
    """
    Conveyor Tracking: Segment objects on a conveyor belt.

    Loads an image, and segments the box using SAM.
    Visualizes the results using Rerun.
    """ 
    # Load image
    image_path = DATA_DIR / "images/conveyor_tracking.png"
    image = io.load_image(image_path, keep_alpha=False)
    logger.info(f"Loaded image shape: {image.to_numpy().shape}")

    # Define a bounding box: (x, y, width, height)
    height, width = image.to_numpy().shape[:2]
    x_min = width // 12
    y_min = height // 10
    x_max = width // 1.5
    y_max = height // 1.5
    bounding_box = [x_min, y_min, x_max, y_max]

    # Segment using SAM
    result = cornea.segment_image_using_sam(image=image, 
                                           bboxes=[bounding_box])
    annotations = result.to_list()

    # Rerun visualization
    rr.init("conveyor_tracking_using_sam", spawn=False)
    try:
        rr.connect()
    except Exception as e:
        # If connection fails, attempt to spawn a new Rerun viewer window.
        rr.spawn()
    
    # Blueprint
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

    # --- Logging images ---
    image = image.to_numpy()
    rr.log("input/image", rr.Image(image=image))
    rr.log("segmented/image", rr.Image(image=image))

    h, w = image.shape[:2]
    masks = []
    masks_with_ids = []
    segmentation_img = np.zeros((h, w), dtype=np.uint16)

    # --- boxes: extract from annotations (preferred) ---
    ann_bboxes = []
    class_ids = []

    for idx, ann in enumerate(annotations):
        label = idx + 1
        mask_i = np.zeros((h, w), dtype=np.uint8)

        if "mask" in ann and isinstance(ann["mask"], np.ndarray):
            m = ann["mask"]
            # if float prob mask, threshold at 0.5
            if m.dtype.kind in ("f", "b"):
                mask_i = (m > 0.5).astype(np.uint8)
            else:
                mask_i = (m > 0).astype(np.uint8)

        elif "segmentation" in ann and ann["segmentation"]:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                mask_dec = mask_utils.decode(seg)
                if mask_dec.ndim == 3:
                    mask_dec = mask_dec[:, :, 0]
                mask_i = (mask_dec > 0).astype(np.uint8)
            elif isinstance(seg, list) and len(seg) > 0:
                temp = np.zeros((h, w), dtype=np.uint8)
                polys = seg if isinstance(seg[0], list) else [seg]
                for poly in polys:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(temp, [pts], 1)
                mask_i = (temp > 0).astype(np.uint8)

        # skip empty masks
        if mask_i.sum() == 0:
            print(f"Skipping annotation {idx} with label {label} due to empty mask.")
            continue

        masks.append(mask_i)
        masks_with_ids.append((label, mask_i))
        segmentation_img[mask_i > 0] = label

        bbox = ann.get("bbox", None)
        if bbox is None:
            continue

        ann_bboxes.append(list(bbox))
        class_ids.append(label)
        

    # --- overlay segmentation in segmented view ---
    rr.log("segmented/masks", rr.SegmentationImage(segmentation_img))

    # --- boxes ---
    if ann_bboxes:
        rr.log(
            "segmented/boxes",
            rr.Boxes2D(
                array=np.asarray(ann_bboxes, dtype=np.float32),
                array_format=rr.Box2DFormat.XYWH,
                class_ids=np.asarray(class_ids, dtype=np.int32),
            ),
        )
   
# Depalletizing
def depalletizing_using_sam_example():
    """
    Depalletizing: Segment objects on a pallet using SAM.

    Loads an image, crops it, and segments the box using SAM.

    Visualizes the results using Rerun.
    """
    
    # Load image
    image_path = DATA_DIR / "images/depalletizing.png"
    image = io.load_image(image_path)
    logger.info(f"Loaded image shape: {image.to_numpy().shape}")

    # Define a bounding box: (x, y, width, height)
    bounding_box = [170, 370, 360, 500]

    # Segment using SAM
    result = cornea.segment_image_using_sam(
        image=image,
        bboxes=[bounding_box],
    )
    annotations = result.to_list()  

    # Rerun visualization
    rr.init("depalletizing_using_sam", spawn=False)
    try:
        rr.connect()
    except Exception as e:
        # If connection fails, attempt to spawn a new Rerun viewer window.
        rr.spawn()
    
    # Blueprint
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

    # --- Logging images ---
    image = image.to_numpy()
    rr.log("input/image", rr.Image(image=image))
    rr.log("segmented/image", rr.Image(image=image))

    # Create empty masks
    h, w = image.shape[:2]
    masks = []
    masks_with_ids = []
    segmentation_img = np.zeros((h, w), dtype=np.uint16)

    # --- boxes: extract from annotations (preferred) ---
    ann_bboxes = []
    class_ids = []

    for idx, ann in enumerate(annotations):
        label = idx + 1
        mask_i = np.zeros((h, w), dtype=np.uint8)

        if "mask" in ann and isinstance(ann["mask"], np.ndarray):
            m = ann["mask"]
            # if float prob mask, threshold at 0.5
            if m.dtype.kind in ("f", "b"):
                mask_i = (m > 0.5).astype(np.uint8)
            else:
                mask_i = (m > 0).astype(np.uint8)

        elif "segmentation" in ann and ann["segmentation"]:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                mask_dec = mask_utils.decode(seg)
                if mask_dec.ndim == 3:
                    mask_dec = mask_dec[:, :, 0]
                mask_i = (mask_dec > 0).astype(np.uint8)
            elif isinstance(seg, list) and len(seg) > 0:
                temp = np.zeros((h, w), dtype=np.uint8)
                polys = seg if isinstance(seg[0], list) else [seg]
                for poly in polys:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(temp, [pts], 1)
                mask_i = (temp > 0).astype(np.uint8)

        # skip empty masks
        if mask_i.sum() == 0:
            print(f"Skipping annotation {idx} with label {label} due to empty mask.")
            continue

        masks.append(mask_i)
        masks_with_ids.append((label, mask_i))
        segmentation_img[mask_i > 0] = label

        bbox = ann.get("bbox", None)
        if bbox is None:
            continue

        ann_bboxes.append(list(bbox))
        class_ids.append(label)
        

    # --- overlay segmentation in segmented view ---
    rr.log("segmented/masks", rr.SegmentationImage(segmentation_img))

    # --- boxes ---
    if ann_bboxes:
        rr.log(
            "segmented/boxes",
            rr.Boxes2D(
                array=np.asarray(ann_bboxes, dtype=np.float32),
                array_format=rr.Box2DFormat.XYWH,
                class_ids=np.asarray(class_ids, dtype=np.int32),
            ),
        )
       
# Bin Picking
def bin_picking_using_sam_example():
    """
    Bin Picking: Segment objects in a bin using SAM.

    Loads an image, and segments the box using SAM.
    Visualizes the results using Rerun.
    """

    # Load image
    image_path = DATA_DIR / "images/bin_picking_metal_2.jpg"
    image = io.load_image(image_path)
    logger.info(f"Loaded image shape: {image.to_numpy().shape}")

    # Define a bounding box: (x_min, y_min, x_max, y_max)
    bounding_box = [550, 260, 680, 350]

    # Segment using SAM
    result = cornea.segment_image_using_sam(
        image=image,
        bboxes=[bounding_box],
    )
    annotations = result.to_list()

    # Rerun visualization
    rr.init("bin_picking_using_sam", spawn=False)
    try:
        rr.connect()
    except Exception as e:
        rr.spawn()

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

    # --- Logging images ---
    image = image.to_numpy()
    rr.log("input/image", rr.Image(image=image))
    rr.log("segmented/image", rr.Image(image=image))

    # Create empty masks
    h, w = image.shape[:2]
    masks = []
    masks_with_ids = []
    segmentation_img = np.zeros((h, w), dtype=np.uint16)

    # --- boxes: extract from annotations (preferred) ---
    ann_bboxes = []
    class_ids = []

    for idx, ann in enumerate(annotations):
        label = idx + 1
        mask_i = np.zeros((h, w), dtype=np.uint8)

        if "mask" in ann and isinstance(ann["mask"], np.ndarray):
            m = ann["mask"]
            if m.dtype.kind in ("f", "b"):
                mask_i = (m > 0.5).astype(np.uint8)
            else:
                mask_i = (m > 0).astype(np.uint8)

        elif "segmentation" in ann and ann["segmentation"]:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                mask_dec = mask_utils.decode(seg)
                if mask_dec.ndim == 3:
                    mask_dec = mask_dec[:, :, 0]
                mask_i = (mask_dec > 0).astype(np.uint8)
            elif isinstance(seg, list) and len(seg) > 0:
                temp = np.zeros((h, w), dtype=np.uint8)
                polys = seg if isinstance(seg[0], list) else [seg]
                for poly in polys:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(temp, [pts], 1)
                mask_i = (temp > 0).astype(np.uint8)

        if mask_i.sum() == 0:
            print(f"Skipping annotation {idx} with label {label} due to empty mask.")
            continue

        masks.append(mask_i)
        masks_with_ids.append((label, mask_i))
        segmentation_img[mask_i > 0] = label

        bbox = ann.get("bbox", None)
        if bbox is None:
            continue

        ann_bboxes.append(list(bbox))
        class_ids.append(label)

    rr.log("segmented/masks", rr.SegmentationImage(segmentation_img))

    if ann_bboxes:
        rr.log(
            "segmented/boxes",
            rr.Boxes2D(
                array=np.asarray(ann_bboxes, dtype=np.float32),
                array_format=rr.Box2DFormat.XYWH,
                class_ids=np.asarray(class_ids, dtype=np.int32),
            ),
        )

# Ground Segmentation
def ground_segmentation_using_sam_example():
    """
    Ground Segmentation: Segment ground regions in an image using SAM.

    Loads an image, defines a bounding box for the ground, and segments it using SAM.
    Visualizes the results using Rerun.
    """
    # Load image
    image_path = DATA_DIR / "images/warehouse_ground.jpg"
    image = io.load_image(image_path)
    logger.info(f"Loaded image shape: {image.to_numpy().shape}")

    # Define a bounding box: (x_min, y_min, x_max, y_max)
    bounding_box = [3, 294, 794, 499] 

    # Segment using SAM
    result = cornea.segment_image_using_sam(
        image=image,
        bboxes=[bounding_box],
    )
    annotations = result.to_list()

    # Rerun visualization
    rr.init("ground_segmentation_using_sam", spawn=False)
    try:
        rr.connect()
    except Exception as e:
        rr.spawn()

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

    # --- Logging images ---
    image = image.to_numpy()
    rr.log("input/image", rr.Image(image=image))
    rr.log("segmented/image", rr.Image(image=image))

    # Create empty masks
    h, w = image.shape[:2]
    masks = []
    masks_with_ids = []
    segmentation_img = np.zeros((h, w), dtype=np.uint16)

    # --- boxes: extract from annotations (preferred) ---
    ann_bboxes = []
    class_ids = []

    for idx, ann in enumerate(annotations):
        label = idx + 1
        mask_i = np.zeros((h, w), dtype=np.uint8)

        if "mask" in ann and isinstance(ann["mask"], np.ndarray):
            m = ann["mask"]
            if m.dtype.kind in ("f", "b"):
                mask_i = (m > 0.5).astype(np.uint8)
            else:
                mask_i = (m > 0).astype(np.uint8)

        elif "segmentation" in ann and ann["segmentation"]:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                mask_dec = mask_utils.decode(seg)
                if mask_dec.ndim == 3:
                    mask_dec = mask_dec[:, :, 0]
                mask_i = (mask_dec > 0).astype(np.uint8)
            elif isinstance(seg, list) and len(seg) > 0:
                temp = np.zeros((h, w), dtype=np.uint8)
                polys = seg if isinstance(seg[0], list) else [seg]
                for poly in polys:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(temp, [pts], 1)
                mask_i = (temp > 0).astype(np.uint8)

        if mask_i.sum() == 0:
            print(f"Skipping annotation {idx} with label {label} due to empty mask.")
            continue

        masks.append(mask_i)
        masks_with_ids.append((label, mask_i))
        segmentation_img[mask_i > 0] = label

        bbox = ann.get("bbox", None)
        if bbox is None:
            continue

        ann_bboxes.append(list(bbox))
        class_ids.append(label)

    rr.log("segmented/masks", rr.SegmentationImage(segmentation_img))

    if ann_bboxes:
        rr.log(
            "segmented/boxes",
            rr.Boxes2D(
                array=np.asarray(ann_bboxes, dtype=np.float32),
                array_format=rr.Box2DFormat.XYWH,
                class_ids=np.asarray(class_ids, dtype=np.int32),
            ),
        )

# Pedestrian Segmentation
def pedestrian_segmentation_using_sam_example():
    """
    Pedestrian Segmentation: Segment pedestrian regions in an image using SAM.

    Loads an image, defines a bounding box for the pedestrian, and segments it using SAM.
    Visualizes the results using Rerun.
    """
    # Load image
    image_path = DATA_DIR / "images/pedestrians.jpg"
    image = io.load_image(image_path)
    logger.info(f"Loaded image shape: {image.to_numpy().shape}")

    # Define a bounding box: (x_min, y_min, x_max, y_max)
    bounding_box = [40, 70, 330, 414] 

    # Segment using SAM
    result = cornea.segment_image_using_sam(
        image=image,
        bboxes=[bounding_box],
    )
    annotations = result.to_list()

    # Rerun visualization
    rr.init("pedestrian_segmentation_using_sam", spawn=False)
    try:
        rr.connect()
    except Exception as e:
        rr.spawn()

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

    # --- Logging images ---
    image = image.to_numpy()
    rr.log("input/image", rr.Image(image=image))
    rr.log("segmented/image", rr.Image(image=image))

    # Create empty masks
    h, w = image.shape[:2]
    masks = []
    masks_with_ids = []
    segmentation_img = np.zeros((h, w), dtype=np.uint16)

    # --- boxes: extract from annotations (preferred) ---
    ann_bboxes = []
    class_ids = []

    for idx, ann in enumerate(annotations):
        label = idx + 1
        mask_i = np.zeros((h, w), dtype=np.uint8)

        if "mask" in ann and isinstance(ann["mask"], np.ndarray):
            m = ann["mask"]
            if m.dtype.kind in ("f", "b"):
                mask_i = (m > 0.5).astype(np.uint8)
            else:
                mask_i = (m > 0).astype(np.uint8)

        elif "segmentation" in ann and ann["segmentation"]:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                mask_dec = mask_utils.decode(seg)
                if mask_dec.ndim == 3:
                    mask_dec = mask_dec[:, :, 0]
                mask_i = (mask_dec > 0).astype(np.uint8)
            elif isinstance(seg, list) and len(seg) > 0:
                temp = np.zeros((h, w), dtype=np.uint8)
                polys = seg if isinstance(seg[0], list) else [seg]
                for poly in polys:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(temp, [pts], 1)
                mask_i = (temp > 0).astype(np.uint8)

        if mask_i.sum() == 0:
            print(f"Skipping annotation {idx} with label {label} due to empty mask.")
            continue

        masks.append(mask_i)
        masks_with_ids.append((label, mask_i))
        segmentation_img[mask_i > 0] = label

        bbox = ann.get("bbox", None)
        if bbox is None:
            continue

        ann_bboxes.append(list(bbox))
        class_ids.append(label)

    rr.log("segmented/masks", rr.SegmentationImage(segmentation_img))

    if ann_bboxes:
        rr.log(
            "segmented/boxes",
            rr.Boxes2D(
                array=np.asarray(ann_bboxes, dtype=np.float32),
                array_format=rr.Box2DFormat.XYWH,
                class_ids=np.asarray(class_ids, dtype=np.int32),
            ),
        )

# PCB Segmentation
def pcb_segmentation_using_sam_example():
    """
    PCB Segmentation: Segment PCB regions in an image using SAM.

    Loads an image, defines a bounding box for the PCB, and segments it using SAM.
    Visualizes the results using Rerun.
    """
    # Load image
    image_path = DATA_DIR / "images/pcb.jpg"
    image = io.load_image(image_path)
    logger.info(f"Loaded image shape: {image.to_numpy().shape}")

    # Define a bounding box: (x_min, y_min, x_max, y_max)
    bounding_box = [1185, 1407, 1645, 1690] 

    # Segment using SAM
    result = cornea.segment_image_using_sam(
        image=image,
        bboxes=[bounding_box],
    )
    annotations = result.to_list()

    # Rerun visualization
    rr.init("pcb_segmentation_using_sam", spawn=False)
    try:
        rr.connect()
    except Exception as e:
        rr.spawn()

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

    # --- Logging images ---
    image = image.to_numpy()
    rr.log("input/image", rr.Image(image=image))
    rr.log("segmented/image", rr.Image(image=image))

    # Create empty masks
    h, w = image.shape[:2]
    masks = []
    masks_with_ids = []
    segmentation_img = np.zeros((h, w), dtype=np.uint16)

    # --- boxes: extract from annotations (preferred) ---
    ann_bboxes = []
    class_ids = []

    for idx, ann in enumerate(annotations):
        label = idx + 1
        mask_i = np.zeros((h, w), dtype=np.uint8)

        if "mask" in ann and isinstance(ann["mask"], np.ndarray):
            m = ann["mask"]
            if m.dtype.kind in ("f", "b"):
                mask_i = (m > 0.5).astype(np.uint8)
            else:
                mask_i = (m > 0).astype(np.uint8)

        elif "segmentation" in ann and ann["segmentation"]:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                mask_dec = mask_utils.decode(seg)
                if mask_dec.ndim == 3:
                    mask_dec = mask_dec[:, :, 0]
                mask_i = (mask_dec > 0).astype(np.uint8)
            elif isinstance(seg, list) and len(seg) > 0:
                temp = np.zeros((h, w), dtype=np.uint8)
                polys = seg if isinstance(seg[0], list) else [seg]
                for poly in polys:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(temp, [pts], 1)
                mask_i = (temp > 0).astype(np.uint8)

        if mask_i.sum() == 0:
            print(f"Skipping annotation {idx} with label {label} due to empty mask.")
            continue

        masks.append(mask_i)
        masks_with_ids.append((label, mask_i))
        segmentation_img[mask_i > 0] = label

        bbox = ann.get("bbox", None)
        if bbox is None:
            continue

        ann_bboxes.append(list(bbox))
        class_ids.append(label)

    rr.log("segmented/masks", rr.SegmentationImage(segmentation_img))

    if ann_bboxes:
        rr.log(
            "segmented/boxes",
            rr.Boxes2D(
                array=np.asarray(ann_bboxes, dtype=np.float32),
                array_format=rr.Box2DFormat.XYWH,
                class_ids=np.asarray(class_ids, dtype=np.int32),
            ),
        )

# Forklift Segmentation
def forklift_segmentation_using_sam_example():
    """
    Forklift Segmentation: Segment forklifts in an image using SAM.

    Loads an image, defines a bounding box for the forklift, and segments it using SAM.
    Visualizes the results using Rerun.
    """
    # Load image
    image_path = DATA_DIR / "images/forklift.jpg"
    image = io.load_image(image_path)
    logger.info(f"Loaded image shape: {image.to_numpy().shape}")

    # Define a bounding box: (x_min, y_min, x_max, y_max)
    bounding_box = [18, 216, 303, 389] 

    # Segment using SAM
    result = cornea.segment_image_using_sam(
        image=image,
        bboxes=[bounding_box],
    )
    annotations = result.to_list()

    # Rerun visualization
    rr.init("forklift_segmentation_using_sam", spawn=False)
    try:
        rr.connect()
    except Exception as e:
        rr.spawn()

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

    # --- Logging images ---
    image = image.to_numpy()
    rr.log("input/image", rr.Image(image=image))
    rr.log("segmented/image", rr.Image(image=image))

    # Create empty masks
    h, w = image.shape[:2]
    masks = []
    masks_with_ids = []
    segmentation_img = np.zeros((h, w), dtype=np.uint16)

    # --- boxes: extract from annotations (preferred) ---
    ann_bboxes = []
    class_ids = []

    for idx, ann in enumerate(annotations):
        label = idx + 1
        mask_i = np.zeros((h, w), dtype=np.uint8)

        if "mask" in ann and isinstance(ann["mask"], np.ndarray):
            m = ann["mask"]
            if m.dtype.kind in ("f", "b"):
                mask_i = (m > 0.5).astype(np.uint8)
            else:
                mask_i = (m > 0).astype(np.uint8)

        elif "segmentation" in ann and ann["segmentation"]:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                mask_dec = mask_utils.decode(seg)
                if mask_dec.ndim == 3:
                    mask_dec = mask_dec[:, :, 0]
                mask_i = (mask_dec > 0).astype(np.uint8)
            elif isinstance(seg, list) and len(seg) > 0:
                temp = np.zeros((h, w), dtype=np.uint8)
                polys = seg if isinstance(seg[0], list) else [seg]
                for poly in polys:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(temp, [pts], 1)
                mask_i = (temp > 0).astype(np.uint8)

        if mask_i.sum() == 0:
            print(f"Skipping annotation {idx} with label {label} due to empty mask.")
            continue

        masks.append(mask_i)
        masks_with_ids.append((label, mask_i))
        segmentation_img[mask_i > 0] = label

        bbox = ann.get("bbox", None)
        if bbox is None:
            continue

        ann_bboxes.append(list(bbox))
        class_ids.append(label)

    rr.log("segmented/masks", rr.SegmentationImage(segmentation_img))

    if ann_bboxes:
        rr.log(
            "segmented/boxes",
            rr.Boxes2D(
                array=np.asarray(ann_bboxes, dtype=np.float32),
                array_format=rr.Box2DFormat.XYWH,
                class_ids=np.asarray(class_ids, dtype=np.int32),
            ),
        )


def get_example_dict():
    """Return mapping of available use-case example names to callables.

    These are placeholders; actual use-case implementations will be added
    later. Keep the keys lower-case so the CLI can match them case-insensitively.
    """
    return {
        # Map the example name to the actual function implementation
        "conveyor_tracking_using_sam": conveyor_tracking_using_sam_example,
        "depalletizing_using_sam": depalletizing_using_sam_example,
        "bin_picking_using_sam": bin_picking_using_sam_example,
        "ground_segmentation_using_sam": ground_segmentation_using_sam_example,
        "pcb_segmentation_using_sam": pcb_segmentation_using_sam_example,
        "pedestrian_segmentation_using_sam": pedestrian_segmentation_using_sam_example,
        "forklift_segmentation_using_sam": forklift_segmentation_using_sam_example,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run use-case examples (stubs)")
    parser.add_argument("--example", type=str, help="Name of the example to run")
    parser.add_argument("--list", action="store_true", help="List all available examples")
    return parser.parse_args()


def main():
    args = parse_args()
    example_dict = get_example_dict()

    if args.list:
        logger.info("Available use-case examples:")
        for name in sorted(example_dict.keys()):
            logger.info(f"  - {name}")
        return

    if not args.example:
        logger.error("Please provide --example or use --list to see available examples.")
        raise SystemExit(1)

    name = args.example.lower()
    if name not in example_dict:
        logger.error(f"Example '{name}' not found.")
        close_matches = difflib.get_close_matches(name, example_dict.keys(), n=3, cutoff=0.4)
        if close_matches:
            logger.error("Did you mean one of these?")
            for m in close_matches:
                logger.error(f"  - {m}")
        raise SystemExit(1)

    logger.info(f"Running use-case example: {name}")
    try:
        example_dict[name]()
    except NotImplementedError:
        logger.warning(f"Example '{name}' is a stub. Implement the function to run it.")


if __name__ == "__main__":
    main()
