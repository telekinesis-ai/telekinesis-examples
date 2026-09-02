"""
Demonstrates segmentation using SAM3 (Segment Anything Model 3).
"""

from loguru import logger
import rerun as rr
import rerun.blueprint as rrb

from telekinesis import cornea, datatypes

def segment_image_using_sam3_example():
    """Segments an image using the Segment Anything Model 3 (SAM3)."""
    # ===================== Load Image ==========================================
    image_url = "https://assets.telekinesis.ai/examples/v1/images/pedestrians.jpg"
    image = datatypes.Image.from_url(url=image_url)

    # ===================== Run Skill ==========================================
    objects = ["pedestrian"]
    segmentation_results = cornea.segment_image_using_sam3(
        image=image, objects=objects, threshold=0.5, mask_threshold=0.5
    )

    # ===================== Log ================================================
    logger.success(f"Segmented {image} using SAM3.")
    logger.success(f"Results: {segmentation_results}")
    logger.info(f"Number of segmented objects: {len(segmentation_results)}")

    # ===================== Visualization  (Optional) ======================
    # `category_id` is `0` for a box-only detection, or `i + 1` for a
    # detection matching `objects[i]` -- map it back to the matched
    # concept's name so boxes/masks show e.g. "pedestrian" instead of the
    # raw numeric "category=1".
    labels = [
        objects[category_id - 1] if category_id > 0 else "box prompt"
        for category_id in segmentation_results.category_ids.tolist()
    ]

    rr.init("segment_image_using_sam3_example", spawn=True)
    blueprint = rrb.Horizontal(
        rrb.Spatial2DView(origin="/input_image", name="Input"),
        rrb.Spatial2DView(origin="/segmented_image", name="Output"),
    )
    rr.send_blueprint(blueprint)
    datatypes.visualize(image, entity_path="/input_image")
    datatypes.visualize(
        image, segmentation_results, entity_path="/segmented_image", label=labels
    )


if __name__ == "__main__":
    segment_image_using_sam3_example()
