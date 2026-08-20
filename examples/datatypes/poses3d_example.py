"""Demonstrates the Telekinesis Poses3D datatype."""

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def poses3d_example():
    """Demonstrate creation, access, and visualization."""

    # ======================= Create ============================================
    poses3d = datatypes.Poses3D(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]
    )
    logger.info(f"Original Poses3D: {poses3d}")

    # ======================= Inspect ===========================================
    logger.info(f"Underlying Poses3D data: {poses3d.data}")

    # ======================= Visualize =========================================
    rr.init("poses3d_example", spawn=True)
    datatypes.visualize(poses3d, entity_path="/Poses3D", label=["My Poses3D 0", "My Poses3D 1"])


if __name__ == "__main__":
    poses3d_example()
