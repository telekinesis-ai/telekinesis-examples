"""
Demonstrates estimating a metric depth map from a rectified stereo image
pair using Fast-FoundationStereo.
"""

from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def estimate_depth_using_fast_foundation_stereo_example():
    """
    Estimates a metric depth map from a rectified stereo pair.
    """
    # ===================== Load Data ==========================================
    left_image_url = "https://assets.telekinesis.ai/examples/v1/images/foundationstereo_left.png"
    right_image_url = "https://assets.telekinesis.ai/examples/v1/images/foundationstereo_right.png"
    left_image = datatypes.Image.from_url(url=left_image_url)
    right_image = datatypes.Image.from_url(url=right_image_url)

    camera_calibration = datatypes.CameraCalibration(
        width=left_image.width,
        height=left_image.height,
        distortion_model="plumb_bob",
        distortion_parameters=[0.0, 0.0, 0.0, 0.0, 0.0],
        intrinsic_matrix=[
            433.68502808, 0.0, 420.71362305,
            0.0, 433.68502808, 242.72007751,
            0.0, 0.0, 1.0,
        ],
    )

    # ===================== Run Skill ==========================================
    depth_image = vitreous.estimate_depth_using_fast_foundation_stereo(
        left_image=left_image,
        right_image=right_image,
        camera_calibration=camera_calibration,
        baseline=0.01798470,  # meters
    )

    # ===================== Log ================================================
    logger.success(f"Estimated depth from {left_image} and {right_image}")
    logger.success(f"Results: {depth_image}")
    logger.info(f"Depth image shape: {depth_image.shape}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("estimate_depth_using_fast_foundation_stereo_example", spawn=True)
    datatypes.visualize(left_image, entity_path="/1-left_image")
    datatypes.visualize(right_image, entity_path="/2-right_image")
    datatypes.visualize(depth_image, entity_path="/3-depth_image")


if __name__ == "__main__":
    estimate_depth_using_fast_foundation_stereo_example()
