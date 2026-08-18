"""Demonstrates the Telekinesis Poses2D datatype."""

from loguru import logger
import rerun as rr

from telekinesis import datatypes


def poses2d_example():
    """Demonstrate creation, access, and visualization."""

    # ======================= Create ============================================
    poses2d = datatypes.Poses2D([[1.0, 2.0, 0.5], [3.0, 4.0, 1.2]])
    logger.info(f"Original Poses2D: {poses2d}")

    # ======================= Inspect ===========================================
    data = poses2d.data
    logger.info(f"Underlying Poses2D data: {data}")

    # ======================= Visualize =========================================
    rr.init("poses2d_example", spawn=True)
    datatypes.visualize(poses2d, entity_path="/Poses2D", label=["My Poses2D 0", "My Poses2D 1"])


if __name__ == "__main__":
    poses2d_example()
