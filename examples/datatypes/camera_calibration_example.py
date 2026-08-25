"""Demonstrates the Telekinesis CameraCalibration datatype."""

import time

import rerun as rr
from loguru import logger

from telekinesis import datatypes

def camera_calibration_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    camera_calibration = datatypes.CameraCalibration(
        width=1280,
        height=720,
        distortion_model="plumb_bob",
        distortion_parameters=[-0.2, 0.1, 0.0, 0.0, 0.0],
        intrinsic_matrix=[1000.0, 0.0, 640.0, 0.0, 1000.0, 360.0, 0.0, 0.0, 1.0],
    )
    logger.info(f"Created CameraCalibration: {camera_calibration}") 

    # ======================= Inspect ===========================================
    logger.info(f"width={camera_calibration.width}")
    logger.info(f"height={camera_calibration.height}")
    logger.info(f"distortion_model={camera_calibration.distortion_model}")
    logger.info(f"distortion_parameters={camera_calibration.distortion_parameters}")
    logger.info(f"intrinsic_matrix=\n{camera_calibration.intrinsic_matrix}")

    # ======================= Operations =========================================
    intrinsic_matrix_view = camera_calibration.intrinsic_matrix
    intrinsic_matrix_view[0, 0] = -1.0
    logger.info(
        "Mutating the returned intrinsic_matrix does not affect the original: "
        f"intrinsic_matrix[0, 0]={camera_calibration.intrinsic_matrix[0, 0]}"
    )

    # ======================= Visualize =========================================
    rr.init("camera_calibration_example", spawn=True)
    datatypes.visualize(camera_calibration, entity_path="/camera_calibration")

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(camera_calibration)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized CameraCalibration: {deserialized}")
    logger.info(f"Round-trip successful: {camera_calibration == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    camera_calibration_example()
