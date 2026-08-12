"""
Demonstrates PCA calculation on a binary mask.

This example:
- Downloads an example binary image.
- Computes PCA to find principal components.
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


def calculate_image_pca_example():
    """Computes PCA on a binary mask."""
    # ===================== Load Image ==========================================
    image_url = "https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/examples/v1/images/can_vertical_6_mask.png"
    image = fetch_image(image_url)

    # ===================== Run Skill ==========================================
    centroid, eigenvectors, eigenvalues, angle = pupil.calculate_image_pca(
        image=image
    )

    centroid_np = centroid.to_numpy()
    eigenvectors_np = eigenvectors.to_numpy()
    eigenvalues_np = eigenvalues.to_numpy()
    angle_val = angle.value

    logger.success(
        "Computed PCA. Centroid: ({}, {}), Angle: {} degrees",
        centroid_np[0],
        centroid_np[1],
        angle_val,
    )

    # ===================== Visualization  (Optional) ======================
    visualize(image, centroid_np, eigenvectors_np, eigenvalues_np)


def fetch_image(image_url: str) -> datatypes.Image:
    """
    Downloads a binary image from a given URL and returns it as a telekinesis.datatypes.Image object.
    """
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image_bgr = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE,
    )
    image = datatypes.Image(image=image_bgr, color_model="L")
    logger.success(f"Loaded image from {image_url}")
    return image


def visualize(image: datatypes.Image, centroid, eigenvectors, eigenvalues) -> None:
    """Visualizes the binary mask with PCA results using Rerun."""
    rr.init("calculate_image_pca", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Input", origin="input"),
                rrb.Spatial2DView(name="Output", origin="output"),
            )
        )
    )

    binary_mask = image.to_numpy()
    rr.log("input", rr.Image(binary_mask))
    rr.log("output/image", rr.Image(binary_mask))

    if centroid is not None and eigenvectors is not None and eigenvalues is not None:
        cx, cy = int(centroid[0]), int(centroid[1])

        # Draw centroid
        rr.log(
            "output/centroid",
            rr.Points2D(
                positions=[[cx, cy]],
                colors=[[0, 255, 0]],
                radii=[8],
            ),
        )

        # Compute scale for visualization
        scale = (
            float(np.sqrt(np.real(eigenvalues[0])))
            if np.any(eigenvalues)
            else 30.0
        )

        ex = float(np.real(eigenvectors[0, 0])) * scale
        ey = float(np.real(eigenvectors[1, 0])) * scale

        pt2 = [cx + ex, cy + ey]

        # Draw principal axis
        rr.log(
            "output/principal_axis",
            rr.LineStrips2D(
                [[[cx, cy], pt2]],
                colors=[[255, 0, 0]],
                radii=[2],
            ),
        )


if __name__ == "__main__":
    calculate_image_pca_example()
