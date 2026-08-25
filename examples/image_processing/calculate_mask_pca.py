"""Demonstrates PCA calculation on a binary mask."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes, cornea


def calculate_mask_pca_example():
    """Computes PCA on a binary mask."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/can_vertical_6_mask.png"
    image = datatypes.Image.from_url(image_url)
    mask = cornea.segment_image_using_otsu_threshold(image=image)

    # ===================== Run Skill ==========================================
    centroid, eigenvectors, eigenvalues, angle = pupil.calculate_mask_pca(
        mask=mask
    )

    # ===================== Log ================================================
    logger.success(f"Calculated PCA of {image}")
    logger.success(f"Result: centroid={centroid}, angle={angle}")

    # ===================== Visualization  (Optional) ======================
    rr.init("calculate_mask_pca_example", spawn=True)
    datatypes.visualize(image, eigenvectors, entity_path="1-mask")
    datatypes.visualize(centroid, entity_path="2-centroid")
    datatypes.visualize(eigenvectors, entity_path="3-eigenvectors")
    datatypes.visualize(eigenvalues, entity_path="4-eigenvalues")
    datatypes.visualize(angle, entity_path="5-angle")
    
if __name__ == "__main__":
    calculate_mask_pca_example()
