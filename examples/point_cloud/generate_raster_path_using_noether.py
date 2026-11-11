"""
Demonstrates planning back-and-forth raster lines over a surface with the Noether plane-slicer.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def _plate_with_hole() -> datatypes.Mesh3D:
    """A flat 0.3 m x 0.2 m plate facing +Z, with a 0.06 m square hole."""
    step = 0.02
    columns, rows = 15, 10
    xs, ys = np.meshgrid(
        np.arange(columns + 1) * step, np.arange(rows + 1) * step
    )
    vertices = np.c_[xs.ravel(), ys.ravel(), np.zeros(xs.size)]
    triangles = []
    for row in range(rows):
        for column in range(columns):
            if 6 <= column < 9 and 4 <= row < 7:
                continue  # leave a hole in the middle
            a = row * (columns + 1) + column
            b = a + columns + 1
            triangles += [[a, a + 1, b + 1], [a, b + 1, b]]
    return datatypes.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=np.asarray(triangles, dtype=np.int32),
    )


def generate_raster_path_using_noether_example():
    """
    Plans a raster path over a plate with a hole.

    Slices the plate with parallel planes and places a waypoint every few
    centimetres along each slice. Lines that cross the hole are split into
    separate segments, so a tool can lift off between them.
    """
    # ===================== Load Data ==========================================
    mesh = _plate_with_hole()

    # ===================== Run Skill ==========================================
    surface_frames, segment_ids = vitreous.generate_raster_path_using_noether(
        mesh=mesh,
        line_spacing=0.04,
        point_spacing=0.02,
        raster_direction=[1.0, 0.0, 0.0],
    )

    # ===================== Log ================================================
    frames = np.asarray(surface_frames.data)
    ids = np.asarray(segment_ids.data)
    segments = np.unique(ids, axis=0)
    logger.success(f"Planned a raster path over {mesh}")
    logger.success(f"Results: {surface_frames}, {segment_ids}")
    logger.info(
        f"{len(frames)} waypoints on {len(np.unique(ids[:, 0]))} raster "
        f"lines, split into {len(segments)} segments by the hole"
    )
    logger.info(f"First waypoint frame (+Z is the surface normal):\n{np.round(frames[0], 4)}")

    # ===================== Visualization  (Optional) ===========================
    rr.init("generate_raster_path_using_noether_example", spawn=True)
    datatypes.visualize(mesh, entity_path="/1-surface")
    strips = [frames[(ids == key).all(axis=1), :3, 3] for key in segments]
    rr.log(
        "/2-raster_segments",
        rr.LineStrips3D(strips, radii=0.002),
        static=True,
    )
    rr.log(
        "/3-surface_normals",
        rr.Arrows3D(
            origins=frames[:, :3, 3],
            vectors=frames[:, :3, 2] * 0.02,
        ),
        static=True,
    )


if __name__ == "__main__":
    generate_raster_path_using_noether_example()
