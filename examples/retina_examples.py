import argparse
import difflib
import pathlib
import numpy as np

from loguru import logger
import rerun as rr

from telekinesis import retina
from datatypes import io

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = ROOT_DIR / "telekinesis-data"

# -------------------------
# Circle examples
# -------------------------


def detect_circle_using_classic_hough_example():
    """
    Detect circles using the classic Hough Circle Transform.

    Runs Hough circle detection on a grayscale image, then converts the
    returned annotations into simple circle + bounding-box overlays for visualization.
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

    circles = []
    bboxes = []

    for annotation in annotations:
        bboxes.append(annotation["bbox"])  # [x, y, w, h]
        circle_dict = annotation["geometry"]
        cx, cy = circle_dict["center"]
        r = circle_dict["radius"]
        circles.append((float(cx), float(cy), float(r)))

    rr.init("classic_hough_circle_detector_example", spawn=True)

    # Log original image
    rr.log("input_image", rr.Image(image_np))

    # Log overlay image (same as input, annotations will be overlaid using rerun primitives)
    rr.log("overlay_image", rr.Image(image_np))

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
        "overlay_image/circles",
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
        "overlay_image/bboxes",
        rr.Boxes2D(array = bboxes, array_format=rr.Box2DFormat.XYWH, colors=[[0, 255, 0]], labels=box_labels, radii=[1] ),
    )



def get_example_dict():
    """Returns a dictionary mapping example names (without _example suffix) to their functions."""
    return {
        # Circle Examples
        "detect_circle_using_classic_hough": detect_circle_using_classic_hough_example,
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

