"""Demonstrates the Telekinesis OrientedBox3D datatype."""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis import datatypes

def oriented_box3d_example():
    """Demonstrate creation, inspection, operations, visualization, and serialization."""

    # ======================= Create ============================================
    # OrientedBox3D format is CXCYCZWHD = [cx, cy, cz, width, height, depth]
    # + rotation [roll_deg, pitch_deg, yaw_deg] (Euler XYZ, in degrees)
    coords = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    oriented_box3d = datatypes.OrientedBox3D(coords)
    logger.info(f"Created OrientedBox3D: {oriented_box3d}")

    xyzxyz_coords = [0.5, 1.0, 1.5, 3.5, 3.0, 2.5, 0.0, 0.0, 0.0]
    oriented_box3d_from_xyzxyz = datatypes.OrientedBox3D.from_xyzxyz(xyzxyz_coords)
    logger.info(f"OrientedBox3D created from xyzxyz format: {oriented_box3d_from_xyzxyz}")

    xyzwhd_coords = [0.5, 1.0, 1.5, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0]
    oriented_box3d_from_xyzwhd = datatypes.OrientedBox3D.from_xyzwhd(xyzwhd_coords)
    logger.info(f"OrientedBox3D created from xyzwhd format: {oriented_box3d_from_xyzwhd}")

    # ======================= Inspect ===========================================
    logger.info(f"data={oriented_box3d.data}")
    logger.info(f"dtype={oriented_box3d.dtype}")
    logger.info(f"ndim={oriented_box3d.ndim}")
    logger.info(f"shape={oriented_box3d.shape}")
    logger.info(f"size={oriented_box3d.size}")
    logger.info(f"center={oriented_box3d.center}")
    logger.info(f"dimensions={oriented_box3d.dimensions}")
    logger.info(f"volume={oriented_box3d.volume}")
    logger.info(f"rotation={oriented_box3d.rotation}")

    # ======================= Operations =========================================
    updated_coords = [2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    oriented_box3d.data = updated_coords
    logger.info(f"Updated OrientedBox3D: {oriented_box3d}")

    xyzxyz_view = oriented_box3d.as_xyzxyz()
    logger.info(f"OrientedBox3D converted to xyzxyz format: {xyzxyz_view}")

    xyzwhd_view = oriented_box3d.as_xyzwhd()
    logger.info(f"OrientedBox3D converted to xyzwhd format: {xyzwhd_view}")

    oriented_box3d_copy = oriented_box3d.copy()
    logger.info(f"Copied OrientedBox3D: {oriented_box3d_copy}")

    # Returns the internal data as a NumPy array. If copy=True, returns a copy; otherwise, returns a view.
    oriented_box3d_numpy = oriented_box3d.to_numpy(copy=False)
    logger.info(f"NumPy OrientedBox3D:\n{oriented_box3d_numpy}")

    # Translate, scale, and rotate by operating on the underlying NumPy array directly.
    translation = [1.0, 1.0, 1.0]
    translated_data = oriented_box3d.data.copy()
    translated_data[:3] += translation
    translated_oriented_box3d = datatypes.OrientedBox3D(translated_data)
    logger.info(f"Translated OrientedBox3D: {translated_oriented_box3d}")

    scale_factors = [1.5, 1.5, 1.5]
    scaled_data = oriented_box3d.data.copy()
    scaled_data[3:6] *= np.asarray(scale_factors, dtype=np.float32)
    scaled_oriented_box3d = datatypes.OrientedBox3D(scaled_data)
    logger.info(f"Scaled OrientedBox3D: {scaled_oriented_box3d}")

    # `rotation` stores Euler-XYZ degrees natively, so a rotation delta is
    # applied by adding directly to the [roll_deg, pitch_deg, yaw_deg] slice.
    rotation_delta_deg = [0.0, 0.0, 90.0]
    rotated_data = oriented_box3d.data.copy()
    rotated_data[6:9] += np.asarray(rotation_delta_deg, dtype=np.float32)
    rotated_oriented_box3d = datatypes.OrientedBox3D(rotated_data)
    logger.info(f"Rotated OrientedBox3D: {rotated_oriented_box3d}")

    numpy_array = np.asarray(oriented_box3d)
    logger.info(f"NumPy array via __array__: {numpy_array}")

    # ======================= Visualize =========================================
    rr.init("oriented_box3d_example", spawn=True)
    datatypes.visualize(oriented_box3d, entity_path="/oriented_box3d/updated", label="Updated Oriented Box3D")
    datatypes.visualize(
        translated_oriented_box3d,
        entity_path="/oriented_box3d/translated",
        label="Translated Oriented Box3D",
    )
    datatypes.visualize(
        scaled_oriented_box3d, entity_path="/oriented_box3d/scaled", label="Scaled Oriented Box3D"
    )
    datatypes.visualize(
        rotated_oriented_box3d, entity_path="/oriented_box3d/rotated", label="Rotated Oriented Box3D"
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(oriented_box3d)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized OrientedBox3D: {deserialized}")
    logger.info(f"Round-trip successful: {oriented_box3d == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    oriented_box3d_example()
