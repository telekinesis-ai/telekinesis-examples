"""
Demonstrates blob thinning (skeletonization) transformation.

This example:
- Downloads an example image.
- Applies blob thinning operation.
- Visualizes the result using Rerun.
"""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def transform_mask_using_blob_thinning_example():
    """Applies blob thinning transformation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/handwriting_mask.png"
    image = datatypes.Image.from_url(image_url)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.transform_mask_using_blob_thinning(
        image=image,
        thinning_type="thinning guohall",
    )

    logger.success(
        "Applied blob thinning. Output image shape: {}",
        filtered_image.shape,
    )

    # ===================== Visualization  (Optional) ======================
    rr.init("transform_mask_using_blob_thinning_example", spawn=True)
    datatypes.visualize(image, entity_path="1-Original")
    datatypes.visualize(filtered_image, entity_path="2-Thinned")

if __name__ == "__main__":
    transform_mask_using_blob_thinning_example()
