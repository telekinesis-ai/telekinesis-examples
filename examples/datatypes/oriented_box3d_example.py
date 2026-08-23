"""Demonstrates the Telekinesis OrientedBox3D datatype."""

import time

from loguru import logger
import rerun as rr
from scipy.spatial.transform import Rotation

from telekinesis import datatypes

def oriented_box3d_example():
    """Demonstrate creation, access, visualization, format conversion, translate/scale/rotate, and serialization."""

    # ======================= Create ============================================
    # OrientedBox3D format is CXCYCZWHD = [cx, cy, cz, width, height, depth]
    # + rotation [roll_deg, pitch_deg, yaw_deg]
    box = datatypes.OrientedBox3D([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    logger.info(f"Created OrientedBox3D: {box}")

    # ======================= Inspect ===========================================
    logger.info(f"OrientedBox3D data: {box.data}")
    logger.info(
        f"dtype={box.dtype}, "
        f"ndim={box.ndim}, "
        f"shape={box.shape}, "
        f"dimensions={box.dimensions}, "
        f"volume={box.volume}, "
        f"center={box.center}, "
        f"rotation={box.rotation}"
    )

    # ======================= Visualize =========================================
    rr.init("oriented_box3d_example", spawn=True)
    datatypes.visualize(
        box, entity_path="/OrientedBox3D/oriented_box3d", label="My Oriented Box3D"
    )

    # ======================= Update ============================================
    box.data = [2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    logger.info(f"Updated OrientedBox3D: {box}")
    datatypes.visualize(
        box, entity_path="/OrientedBox3D/updated_oriented_box3d", label="Updated Oriented Box3D"
    )

    # ======================= Alternate Construction =============================
    # Only the center/dimensions portion is reinterpreted; the trailing
    # rotation entries pass through unchanged.
    xyzxyz_coords = [0.5, 1.0, 1.5, 3.5, 3.0, 2.5, 0.0, 0.0, 0.0]
    box_from_xyzxyz = datatypes.OrientedBox3D.from_xyzxyz(xyzxyz_coords)
    logger.info(f"OrientedBox3D created from xyzxyz format: {box_from_xyzxyz}")

    xyzxyz_view = box.as_xyzxyz()
    logger.info(f"OrientedBox3D converted to xyzxyz format: {xyzxyz_view}")

    xyzwhd_coords = [0.5, 1.0, 1.5, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0]
    box_from_xyzwhd = datatypes.OrientedBox3D.from_xyzwhd(xyzwhd_coords)
    logger.info(f"OrientedBox3D created from xyzwhd format: {box_from_xyzwhd}")

    xyzwhd_view = box.as_xyzwhd()
    logger.info(f"OrientedBox3D converted to xyzwhd format: {xyzwhd_view}")

    # ======================= NumPy Interop =====================================
    # Translate, scale, and rotate by operating on the underlying NumPy array directly.
    translated_data = box.data.copy()
    translated_data[:3] += [1.0, 1.0, 1.0]
    translated_box = datatypes.OrientedBox3D(translated_data)

    logger.info(f"Translated center: {translated_box.center} (was {box.center})")
    datatypes.visualize(
        translated_box,
        entity_path="/OrientedBox3D/translated_oriented_box3d",
        label="Translated Oriented Box3D",
    )

    scaled_data = box.data.copy()
    scaled_data[3:6] *= 1.5
    scaled_box = datatypes.OrientedBox3D(scaled_data)

    logger.info(f"Scaled dimensions: {scaled_box.dimensions} (was {box.dimensions})")
    datatypes.visualize(
        scaled_box, entity_path="/OrientedBox3D/scaled_oriented_box3d", label="Scaled Oriented Box3D"
    )

    delta_rotation_deg = [0.0, 0.0, 90.0]
    current_rot = Rotation.from_euler("xyz", box.rotation, degrees=True)
    delta_rot = Rotation.from_euler("xyz", delta_rotation_deg, degrees=True)
    composed_deg = (delta_rot * current_rot).as_euler("xyz", degrees=True)

    rotated_data = box.data.copy()
    rotated_data[6:9] = composed_deg
    rotated_box = datatypes.OrientedBox3D(rotated_data)

    logger.info(f"Rotated rotation: {rotated_box.rotation} (was {box.rotation})")
    display_data = rotated_box.data.copy()
    display_data[:3] += [4.0, 0.0, 0.0]
    rotated_box_display = datatypes.OrientedBox3D(display_data)
    datatypes.visualize(
        rotated_box_display,
        entity_path="/OrientedBox3D/rotated_oriented_box3d",
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
