"""
Demonstrates segmentation using SAM (Segment Anything Model).
"""

from loguru import logger
import rerun as rr

from telekinesis import cornea, datatypes

def segment_image_using_sam_example():
    """Segments an image using the Segment Anything Model (SAM)."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/pedestrians.jpg"
    image = datatypes.Image.from_url(url=image_url)

    # ===================== Run Skill ==========================================
    bboxes = [[40, 70, 330, 414]]
    segmentation_results = cornea.segment_image_using_sam(
        image=image, bboxes=bboxes, mask_threshold=0.5
    )

    # ===================== Log ================================================
    logger.success(f"Segmented {image} using SAM.")
    logger.success(f"Results: {segmentation_results}")
    logger.info(f"Number of segmented objects: {len(segmentation_results)}")

    # ===================== Visualization  (Optional) ======================
    rr.init("segment_image_using_sam_example", spawn=True)
    datatypes.visualize(image, segmentation_results, entity_path="/Image/overlayed_segmentations")


if __name__ == "__main__":
    segment_image_using_sam_example()
