"""Demonstrates the Telekinesis OrientedBox3D datatype."""

import time

from loguru import logger
import rerun as rr

from telekinesis import datatypes

def oriented_box3d_example():
    """Demonstrate creation, access, visualization, translate/scale/rotate, and serialization."""

    # ======================= Create ============================================
    box = datatypes.OrientedBox3D([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    logger.info(f"Created OrientedBox3D: {box}")

    # ======================= Visualize =========================================
    rr.init("oriented_box3d_example", spawn=True)
    datatypes.visualize(
        box, entity_path="/OrientedBox3D/my_oriented_box3d", label="My Oriented Box3D"
    )

    # ======================= Inspect ===========================================
    logger.info(f"shape={box.shape}, dtype={box.dtype}, ndim={box.ndim}")
    logger.info(f"NumPy array: {box.to_numpy()}")
    logger.info(
        f"center={box.center}, volume={box.volume}, width={box.width}, "
        f"height={box.height}, depth={box.depth}"
    )
    logger.info(f"Quaternion [qx, qy, qz, qw]: {box.data[6:]}")

    # ======================= Update ============================================
    box.data = [2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    logger.info(f"Updated OrientedBox3D: {box}")
    datatypes.visualize(
        box, entity_path="/OrientedBox3D/my_updated_oriented_box3d", label="Updated Oriented Box3D"
    )

    # ======================= Translate =========================================
    translated_box = box.translate([1.0, 1.0, 1.0])

    logger.info(f"Translated center: {translated_box.center} (was {box.center})")
    datatypes.visualize(
        translated_box,
        entity_path="/OrientedBox3D/my_translated_oriented_box3d",
        label="Translated Oriented Box3D",
    )

    # ======================= Scale =============================================
    scaled_box = box.scale(1.5)

    logger.info(
        f"Scaled width, height, and depth: {scaled_box.width} x {scaled_box.height} x "
        f"{scaled_box.depth} (was {box.width} x {box.height} x {box.depth})"
    )
    datatypes.visualize(
        scaled_box, entity_path="/OrientedBox3D/my_scaled_oriented_box3d", label="Scaled Oriented Box3D"
    )

    # ======================= Rotate ============================================
    delta_quaternion = [0.0, 0.0, 0.70710678, 0.70710678]
    rotated_box = box.rotate(delta_quaternion)

    logger.info(f"Rotated quaternion: {rotated_box.data[6:]} (was {box.data[6:]})")
    rotated_box_display = rotated_box.translate([4.0, 0.0, 0.0])
    datatypes.visualize(
        rotated_box_display,
        entity_path="/OrientedBox3D/my_rotated_oriented_box3d",
        label="Rotated Oriented Box3D",
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(box)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized OrientedBox3D: {deserialized}")
    logger.info(f"Round-trip successful: {deserialized == box}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    oriented_box3d_example()
