"""
Demonstrates pyramid downsampling transformation.

This example:
- Downloads an example image.
- Applies pyramid downsampling multiple times.
- Visualizes the result using Rerun.
"""

import numpy as np
import requests
import cv2
from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from datatypes import datatypes
from telekinesis import pupil


def transform_image_using_pyramid_downsampling_example():
    """Applies pyramid downsampling transformation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/gearbox.png"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.transform_image_using_pyramid_downsampling(
        image=image,
        scale_factor=0.5,
    )
    filtered_image_1 = pupil.transform_image_using_pyramid_downsampling(
        image=filtered_image,
        scale_factor=0.5,
    )
    filtered_image_2 = pupil.transform_image_using_pyramid_downsampling(
        image=filtered_image_1,
        scale_factor=0.5,
    )

    filtered_image_np = filtered_image.to_numpy()
    filtered_image_np_1 = filtered_image_1.to_numpy()
    filtered_image_np_2 = filtered_image_2.to_numpy()
    logger.success(
        "Applied pyramid downsampling. Transformed output image shapes: {}, {}, {}",
        filtered_image_np.shape,
        filtered_image_np_1.shape,
        filtered_image_np_2.shape,
    )

    # ===================== Visualization  (Optional) ======================
    visualize(image, filtered_image_np, filtered_image_np_1, filtered_image_np_2)


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


def visualize(image: datatypes.Image, filtered_1, filtered_2, filtered_3) -> None:
    """Visualizes the images using Rerun."""
    rr.init("transform_image_using_pyramid_downsampling", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(name="Original", origin="input"),
                rrb.Spatial2DView(name="Level 1", origin="filtered_image_1"),
                rrb.Spatial2DView(name="Level 2", origin="filtered_image_2"),
                rrb.Spatial2DView(name="Level 3", origin="filtered_image_3"),
            ),
            rrb.SelectionPanel(),
            rrb.TimePanel(),
        ),
        make_active=True,
    )
    image_np = image.to_numpy()
    rr.log("input", rr.Image(image_np))
    rr.log("filtered_image_1", rr.Image(filtered_1))
    rr.log("filtered_image_2", rr.Image(filtered_2))
    rr.log("filtered_image_3", rr.Image(filtered_3))


if __name__ == "__main__":
    transform_image_using_pyramid_downsampling_example()
