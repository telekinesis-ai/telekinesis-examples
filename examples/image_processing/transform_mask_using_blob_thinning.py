"""Demonstrates blob thinning (skeletonization) transformation."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes, cornea


def transform_mask_using_blob_thinning_example():
    """Applies blob thinning transformation."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/handwriting_mask.png"
    image = datatypes.Image.from_url(image_url)

    mask = cornea.segment_image_using_otsu_threshold(image=image)

    # ===================== Run Skill ==========================================
    filtered_image = pupil.transform_mask_using_blob_thinning(
        mask=mask,
        thinning_type="thinning guohall",
    )

    # ===================== Log ================================================
    logger.success(f"Applied blob thinning on {image}")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("transform_mask_using_blob_thinning_example", spawn=True)
    datatypes.visualize(image, entity_path="1-original")
    datatypes.visualize(filtered_image, entity_path="2-thinned")

if __name__ == "__main__":
    transform_mask_using_blob_thinning_example()
