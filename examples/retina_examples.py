"""
    This example demonstrates how to use the Retina Skill Group for image object detection operations, 
    including classic computer vision techniques (like Hough Circle Transform and contour detection) 
    as well as modern neural network-based detectors (like YOLOX, RF-DETR, QWEN VLM, and Grounding DINO).
    
    It also shows how to visualize results using Rerun.
"""


import argparse
import difflib
import pathlib
import numpy as np

from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import retina
from datatypes import io

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()

DATA_DIR = ROOT_DIR / "telekinesis-data"


# -------------------------
# Circle examples
# -------------------------

def detect_circle_using_classic_hough_example():
    """
    Detect circles using the classic Hough Circle Transform.

    Runs Hough circle detection on a grayscale image and returns
    coco-style annotations.

    The annotations are used for visualization overlays.
    """
    # ===================== Operation ==========================================
    # Load image
    filepath = str(DATA_DIR / "images" / "metal_gears.jpg")
    image = io.load_image(filepath=filepath, as_gray=True)
    logger.success(f"Loaded image from {filepath}")

    # Detect circles
    annotations = retina.detect_circle_using_classic_hough(
        image=image,
        inverse_resolution_ratio=1,
        min_distance=50,
        min_radius=40,
        max_radius=60,
        canny_detector_upper_threshold=300,
        accumulator_threshold=30,
    )

    # Access results
    annotations = annotations.to_list()
    logger.success(f"Detected {len(annotations)} circles using classic Hough transform.")

    # ===================== Visualization  (Optional) ===========================
    
    image_np = image.to_numpy()

    # Extract circles and bboxes form annotations
    circles = []
    bboxes = []
    for annotation in annotations:
        bboxes.append(annotation["bbox"])  # [x, y, w, h]
        circle_dict = annotation["geometry"]
        cx, cy = circle_dict["center"]
        r = circle_dict["radius"]
        circles.append((float(cx), float(cy), float(r)))

    # Intialize Rerun and send blueprint
    rr.init("classic_hough_circle_detector_example", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="image"),
                rrb.Spatial2DView(name="Detection", origin="detection"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Log original image
    rr.log("image", rr.Image(image_np))

    # Log overlay image (same as input, annotations will be overlaid using rerun primitives)
    rr.log("detection", rr.Image(image_np))

    # Build circle polylines using LineStrips2D
    def circle_polyline_2d(center_xy, radius, n=128):
        cx, cy = center_xy
        t = np.linspace(0, 2 * np.pi, n, endpoint=True)
        pts = np.stack([cx + radius * np.cos(t), cy + radius * np.sin(t)], axis=1)
        return pts

    circle_polylines = [circle_polyline_2d((cx, cy), r) for cx, cy, r in circles]
    circle_labels = [f"Circle {i} (r={int(r)})" for i, (cx, cy, r) in enumerate(circles)]

    # Log circle outlines as LineStrips2D on overlay image
    rr.log(
        "detection/circles",
        rr.LineStrips2D(
            circle_polylines,
            colors=[[0, 255, 0]] ,
            radii=[1] ,
            labels=circle_labels,
        ),
    )

    # Log bounding boxes as Boxes2D on overlay image
    box_labels = [f"Box {i}" for i in range(len(bboxes))]
    rr.log(
        "detection/bboxes",
        rr.Boxes2D(
            array = bboxes, 
            array_format=rr.Box2DFormat.XYWH, 
            colors=[[0, 255, 0]], 
            labels=box_labels, 
            radii=[1] 
        ),
    )


# -------------------------
# Edge examples
# -------------------------

def detect_contours_example():
    """
    Detect contours using a contour-based detector.

    Extracts contours from the input image and returns
    coco-style annotations. 

    The annotations are used for visualization overlays.
    """

    # ===================== Operation ==========================================
    # Load image
    filepath = str(DATA_DIR / "images" / "nuts_scattered_filtered_gaussian.png")
    image = io.load_image(filepath=filepath, as_binary=True)
    logger.success(f"Loaded image from {filepath}")

    # Detect circles
    annotations = retina.detect_contours(
        image=image,
        retrieval_mode="retrieve_list",
        approx_method="chain_approximate_simple",
        min_area=200,
        max_area=100000,
    )

    # Access results
    annotations = annotations.to_list()
    logger.debug(f"Detected {len(annotations)} contours using contour detector.")

    # ===================== Visualization  (Optional) ======================

    image_np = image.to_numpy()

    # Extract contours and bboxes form annotations
    contour_polylines = []
    bboxes = []
    for annotation in annotations:
        contour_dict = annotation["geometry"]
        points = contour_dict["points"]
        if not points:
            continue
        contour_polylines.append(np.array(points, dtype=np.float32))
        bboxes.append(annotation["bbox"])

    # Intialize Rerun and send blueprint
    rr.init("detect_contours_example", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="image"),
                rrb.Spatial2DView(name="Detection", origin="detection"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Log original image
    rr.log("image", rr.Image(image_np))

    # Log overlay image (same as input, annotations will be overlaid using rerun primitives)
    rr.log("detection", rr.Image(image_np))

    # Log countours as LineStrips2D on overlay image
    contour_labels = [f"Contour {i}" for i in range(len(contour_polylines))]
    rr.log(
        "detection/contours",
        rr.LineStrips2D(
            contour_polylines,
            colors=[[0, 255, 0]],
            radii=[2],
            labels=contour_labels,
        ),
    )

    # Log bounding boxes using Boxes2D on overlay image
    box_labels = [f"Box {i}" for i in range(len(bboxes))]
    rr.log(
        "detection/bboxes",
        rr.Boxes2D(
            array=bboxes,
            array_format=rr.Box2DFormat.XYWH,
            colors=[[0, 255, 0]],
            labels=box_labels,
            radii=[2],
        ),
    )


# -------------------------
# Neural Network examples
# -------------------------

def detect_objects_using_yolox_example():
    """
    Detect objects using YOLOX.

    Runs YOLOX object detection on an image and returns COCO-like annotations
    with category names from the COCO 80-class label set.

    The annotations and categories are used for visualization overlays.
    """

    # ===================== Operation ==========================================

    #load image
    filepath = str(DATA_DIR / "images" / "warehouse_2.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Detect Objects
    annotations, categories = retina.detect_objects_using_yolox(
        image=image,
        score_threshold=0.80,
        nms_threshold=0.45,
    )

    # Access results
    annotations = annotations.to_list()
    categories = categories.to_list()
    logger.success(f"YOLOX detected {len(annotations)} objects.")

    # ===================== Visualization  (Optional) ======================
    
    image_np = image.to_numpy()

    # Build categories_map
    categories_map = {category["id"]: category["name"] for category in categories}

    # Extract objects form annotations
    bboxes = []
    colors = []
    labels = []
    radii = []
    colors_list = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for idx, ann in enumerate(annotations):
        color = colors_list[idx % len(colors_list)]
        label = categories_map.get(ann.get("category_id", 0), "")
        score = ann.get("score", 0.0)
        bboxes.append(ann["bbox"])          # [x, y, w, h]
        colors.append(color)                  # (r,g,b)
        labels.append(f"{label} {score:.2f}")
        radii.append(2)

    # Intialize Rerun and send blueprint
    rr.init("detect_objects_using_yolox_example", spawn=True)
    
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="image"),
                rrb.Spatial2DView(name="Detection", origin="detection"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Log original image
    rr.log("image", rr.Image(image_np))

    # Log overlay image (same as input, annotations will be overlaid using rerun primitives)
    rr.log("detection", rr.Image(image_np))

    # Log bounding boxes as Boxes2D on overlay image
    rr.log(
        "detection/bboxes",
        rr.Boxes2D(
            array=np.array(bboxes, dtype=np.float32),
            array_format=rr.Box2DFormat.XYWH,
            colors=np.array(colors, dtype=np.uint8),
            labels=labels,
            radii=radii,
        ),
    )


def detect_objects_using_rfdetr_example():
    """
    Detect objects using RF-DETR.

    Runs RF-DETR object detection on an image and returns COCO-like annotations
    with category names from the COCO 80-class label set.

    The annotations and categories are used for visualization overlays.
    """

    # ===================== Operation ==========================================

    #load image
    filepath = str(DATA_DIR / "images" / "warehouse_1.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")


    annotations, categories = retina.detect_objects_using_rfdetr(
        image=image,
        score_threshold=0.5,
    )

    # Access results
    annotations = annotations.to_list()
    categories = categories.to_list()  
    logger.success(f"RF-DETR detected {len(annotations)} objects.")

    # ===================== Visualization  (Optional) ======================
    
    image_np = image.to_numpy()

    # Build categories_map
    categories_map = {category["id"]: category["name"] for category in categories}

    # Extract objects form annotations
    bboxes = []
    colors = []
    labels = []
    radii = []
    colors_list = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for idx, ann in enumerate(annotations):
        color = colors_list[idx % len(colors_list)]
        label = categories_map.get(ann.get("category_id", 0), "")
        score = ann.get("score", 0.0)
        bboxes.append(ann["bbox"])          # [x, y, w, h]
        colors.append(color)                  # (r,g,b)
        labels.append(f"{label}{score:.2f}")
        radii.append(2)

    # Intialize Rerun and send blueprint
    rr.init("detect_objects_using_rfdetr_example", spawn=True)

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="image"),
                rrb.Spatial2DView(name="Detection", origin="detection"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Log original image
    rr.log("image", rr.Image(image_np))

    # Log overlay image (same as input, annotations will be overlaid using rerun primitives)
    rr.log("detection", rr.Image(image_np))

    # Log bounding boxes as Boxes2D on overlay image
    rr.log(
        "detection/bboxes",
        rr.Boxes2D(
            array=np.array(bboxes, dtype=np.float32),
            array_format=rr.Box2DFormat.XYWH,
            colors=np.array(colors, dtype=np.uint8),
            labels=labels,
            radii=radii,
        ),
    )

# -------------------------
# VLM examples
# -------------------------


def detect_objects_using_qwen_example():
    """
    Detect objects using QWEN VLM.

    Objects, mentioned in the prompt, are detected in an RGB image and a list of COCO-like annotations is returned. 

    The annotations are used for visualization overlays.
    """

    # ===================== Operation ==========================================

    #load image
    filepath = str(DATA_DIR / "images" / "warehouse_1.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Detect Objects
    annotations = retina.detect_objects_using_qwen(
        image=image,
        objects_to_detect="person .",
        model_name="Qwen/Qwen3-VL-4B-Instruct",
    )

    # Access results
    annotations = annotations.to_list()  
    logger.success(f"Applied QWEN object detection on the given image. Detected {len(annotations)} objects.")

    # ===================== Visualization  (Optional) ======================
    
    image_np = image.to_numpy()

    # Extract objects form annotations
    bboxes = []
    colors = []
    labels = []
    radii = []
    colors_list = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for idx, ann in enumerate(annotations):
        color = colors_list[idx % len(colors_list)]
        bboxes.append(ann["bbox"])          # [x, y, w, h]
        colors.append(color)                  # (r,g,b)
        labels.append(idx)
        radii.append(2)

    # Intialize Rerun and send blueprint
    rr.init("detect_objects_using_qwen_example", spawn=True)
    
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="image"),
                rrb.Spatial2DView(name="Detection", origin="detection"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Log original image
    rr.log("image", rr.Image(image_np))

    # Log overlay image (same as input, annotations will be overlaid using rerun primitives)
    rr.log("detection", rr.Image(image_np))

    # Log bounding boxes as Boxes2D on overlay image
    rr.log(
        "detection/bboxes",
        rr.Boxes2D(
            array=np.array(bboxes, dtype=np.float32),
            array_format=rr.Box2DFormat.XYWH,
            colors=np.array(colors, dtype=np.uint8),
            labels=labels,
            radii=radii,
        ),
    )


def detect_objects_using_grounding_dino_example():
    """
    Detect objects using Grounding DINO (zero-shot).

    Uses a free-form text prompt to detect objects in an RGB image.
    Returns COCO-like annotations with bounding boxes.

    The annotations and categories are used for visualization overlays.
    """

    # ===================== Operation ==========================================

    # Load image
    filepath = str(DATA_DIR / "images" / "palletizing.jpg")
    image = io.load_image(filepath=filepath)
    logger.success(f"Loaded image from {filepath}")

    # Detect Objects
    annotations, categories = retina.detect_objects_using_grounding_dino(
        image=image,
        text="cartons .",
        box_threshold=0.5,
        text_threshold=0.5,
    )

    # Access results
    annotations = annotations.to_list()
    categories = categories.to_list()
    logger.success(f"Grounding DINO detected {len(annotations)} objects.")

    # ===================== Visualization  (Optional) ======================
    
    image_np = image.to_numpy()

    # Build categories_map
    categories_map = {category["id"]: category["name"] for category in categories}

    # Extract objects form annotations
    bboxes = []
    colors = []
    labels = []
    radii = []
    colors_list = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for idx, ann in enumerate(annotations):
        color = colors_list[idx % len(colors_list)]
        label = categories_map.get(ann.get("category_id", 0), "")
        score = ann.get("score", 0.0)
        bboxes.append(ann["bbox"])          # [x, y, w, h]
        colors.append(color)                  # (r,g,b)
        labels.append(f"{label} {score:.2f}")
        radii.append(2)

    # Intialize Rerun and send blueprint
    rr.init("detect_objects_using_grounding_dino_example", spawn=True)
    
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="image"),
                rrb.Spatial2DView(name="Detection", origin="detection"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )

    # Log original image
    rr.log("image", rr.Image(image_np))

    # Log overlay image (same as input, annotations will be overlaid using rerun primitives)
    rr.log("detection", rr.Image(image_np))

    # Log bounding boxes as Boxes2D on overlay image
    rr.log(
        "detection/bboxes",
        rr.Boxes2D(
            array=np.array(bboxes, dtype=np.float32),
            array_format=rr.Box2DFormat.XYWH,
            colors=np.array(colors, dtype=np.uint8),
            labels=labels,
            radii=radii,
        ),
    )


def get_example_dict():
    """Returns a dictionary mapping example names (without _example suffix) to their functions."""
    return {

        # Circle Examples
        "detect_circle_using_classic_hough": detect_circle_using_classic_hough_example,

        # Edge Examples
        "detect_contours": detect_contours_example, 

        # Neural Network Examples
        "detect_objects_using_yolox": detect_objects_using_yolox_example,
        "detect_objects_using_rfdetr": detect_objects_using_rfdetr_example,

        # Vlm Examples
        "detect_objects_using_qwen": detect_objects_using_qwen_example,
        "detect_objects_using_grounding_dino": detect_objects_using_grounding_dino_example,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run retina examples")
    parser.add_argument(
        "--example",
        type=str,
        help="Name of the example to run (without _example suffix) or use --list to see all available examples",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available examples"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    example_dict = get_example_dict()

    if args.list:
        logger.info("Available examples:")
        for example_name in sorted(example_dict.keys()):
            logger.info(f"  - {example_name}")
        return

    if not args.example:
        logger.error(
            "Please provide --example or use --list to see all available examples."
        )
        raise SystemExit(1)

    args.example = args.example.lower()

    if args.example not in example_dict:
        logger.error(f"Example '{args.example}' not found.")

        # Find the one that is nearest in name
        close_matches = difflib.get_close_matches(
            args.example, example_dict.keys(), n=3, cutoff=0.4
        )

        if close_matches:
            logger.error(f"Did you mean one of these?")
            for match in close_matches:
                logger.error(f"  - {match}")

        raise SystemExit(1)

    logger.info(f"Running {args.example} example...")
    example_dict[args.example]()
    logger.success(f"Example {args.example} completed.")


if __name__ == "__main__":
    main()

