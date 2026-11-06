"""Generate antipodal grasp poses from a hosted object mesh.

The sampler finds pairs of opposing surface contacts that fit within the
configured gripper opening. Returned poses use a UR-style TCP frame: +Z points
from the gripper toward the object and +Y is the finger-closing axis. Their
origins lie midway between the paired contacts.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes, vitreous

NUM_GRASPS = 20


def generate_grasps_using_antipodal_sampler_example():
    """Generate and visualize antipodal grasps for a connector mesh."""
    # ===================== Load Data ==========================================
    mesh_url = "https://assets.telekinesis.ai/examples/v1/meshes/male_pin_connector.obj"
    mesh = datatypes.Mesh3D.from_url(url=mesh_url, use_cache=True)

    # The hosted mesh is already in metres. This example places its object
    # frame at the world origin, so returned world-frame poses match the mesh frame.
    w_T_object = np.eye(4, dtype=np.float32)

    # ===================== Run Skill ==========================================
    grasps, scores, antipodal_candidates, ray_backend = (
        vitreous.generate_grasps_using_antipodal_sampler(
            mesh=mesh,
            num_grasps=NUM_GRASPS,
            w_T_object=w_T_object,
            antipodal_thresh=-0.95,
            max_pt_dist=0.08,
            min_pt_dist=0.005,
            max_sample_retries=500,
            seed=42,
            # Desired world-frame TCP +Z direction (approach from above).
            # Omit to use the object-frame -Z default.
            alignment_axis_W=[0.0, 0.0, -1.0],
            alignment_cost_weight=20.0,
            contact_cost_weight=1.0,
        )
    )

    # ===================== Log ================================================
    logger.success(f"Generated {len(grasps.data)} antipodal grasps for {mesh}")
    logger.success(f"Grasps: {grasps}")
    logger.info(f"Ray backend: {ray_backend.data}")
    logger.info(f"Scores (normalized cost in [0, 1], lower is better): {scores.data}")
    logger.info(f"Contact pairs (point_1, point_2, normal_1, normal_2): {antipodal_candidates}")
    if len(grasps.data):
        logger.info(f"Best returned grasp pose:\n{np.round(grasps.data[0], 4)}")
        logger.info(
            "Its paired contacts:\n"
            f"{np.round(antipodal_candidates.data[0, :2], 4)}"
        )
    else:
        logger.warning("No grasps satisfied the configured contact constraints.")

    # ===================== Visualization  (Optional) ===========================
    rr.init("generate_grasps_using_antipodal_sampler_example", spawn=True)
    rr.log(
        "/1-object_mesh",
        rr.Mesh3D(
            vertex_positions=mesh.vertex_positions,
            triangle_indices=mesh.triangle_indices,
            vertex_normals=mesh.vertex_normals,
            vertex_colors=mesh.vertex_colors,
        ),
        static=True,
    )

    for index, (grasp, score, candidate) in enumerate(
        zip(grasps.data, scores.data, antipodal_candidates.data)
    ):
        rr.set_time("grasp", sequence=index)
        datatypes.visualize(
            datatypes.Transform3D(grasp),
            entity_path="/2-selected_grasp",
            label=f"grasp {index}: {score:.3f}",
        )

        # candidate has shape (4, 3): point_1, point_2, normal_1, normal_2.
        contact_points = candidate[:2]
        contact_normals = candidate[2:]
        rr.log(
            "/3-selected_candidate/contacts",
            rr.Points3D(contact_points, radii=0.002),
        )
        rr.log(
            "/3-selected_candidate/pair",
            rr.LineStrips3D([contact_points]),
        )
        rr.log(
            "/3-selected_candidate/normals",
            rr.Arrows3D(
                origins=contact_points,
                vectors=contact_normals * 0.01,
            ),
        )


if __name__ == "__main__":
    generate_grasps_using_antipodal_sampler_example()
