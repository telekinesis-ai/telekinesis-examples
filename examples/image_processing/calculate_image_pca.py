"""
Demonstrates PCA calculation on a binary mask.

This example:
- Downloads an example binary image.
- Computes PCA to find principal components.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def calculate_image_pca_example():
    """Computes PCA on a binary mask."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/can_vertical_6_mask.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    centroid, eigenvectors, eigenvalues, angle = pupil.calculate_image_pca(
        image=image
    )

    logger.success(
        "Computed PCA. Centroid: {}, Angle: {} degrees",
        centroid.data,
        angle.data
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("calculate_image_pca_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Mask")
    datatypes.visualize(centroid, entity_path="2-Centroid")
    datatypes.visualize(eigenvectors, entity_path="3-Eigenvectors")
    datatypes.visualize(eigenvalues, entity_path="4-Eigenvalues")
    datatypes.visualize(angle, entity_path="5-Angle")
    
if __name__ == "__main__":
    calculate_image_pca_example()
