"""
Demonstrates reconstructing a mesh from the depth images of several cameras using TSDF fusion.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes

DEPTH_IMAGE_URL = (
    "https://assets.telekinesis.ai/examples/v1/depth_images/metal_bin/"
    "golden_bin_view_{:02d}.png"
)
# Intrinsics of the depth camera, shared by all views.
INTRINSIC_MATRIX = np.array(
    [
        [892.5320, 0.0000, 484.0409],
        [0.0000, 892.5209, 300.8271],
        [0.0000, 0.0000, 1.0000],
    ]
)
# Top three rows of the camera-to-world pose of each view, from the robot's
# kinematics and hand-eye calibration. The world frame is the robot base, in
# metres.
CAMERA_POSE_TOP_ROWS = np.array(
    [
        [
            [0.999982, 0.005909, -0.001212, 0.157029],
            [0.005920, -0.999941, 0.009150, 0.707153],
            [-0.001157, -0.009157, -0.999957, 0.592811],
        ],
        [
            [0.999981, 0.005992, -0.001507, -0.058806],
            [0.006008, -0.999927, 0.010480, 0.706391],
            [-0.001444, -0.010489, -0.999944, 0.593981],
        ],
        [
            [0.999980, 0.005901, -0.002144, -0.274562],
            [0.005925, -0.999919, 0.011271, 0.705517],
            [-0.002077, -0.011283, -0.999934, 0.594985],
        ],
        [
            [0.645185, 0.006344, 0.764000, -0.196511],
            [-0.003957, -0.999924, 0.011645, 0.823788],
            [0.764016, -0.010536, -0.645111, 0.445925],
        ],
        [
            [0.645313, 0.006593, 0.763890, -0.196365],
            [-0.004057, -0.999919, 0.012057, 0.707785],
            [0.763907, -0.010880, -0.645234, 0.445049],
        ],
        [
            [0.645463, 0.006610, 0.763763, -0.196073],
            [-0.004336, -0.999915, 0.012318, 0.591654],
            [0.763779, -0.011263, -0.645380, 0.444058],
        ],
        [
            [0.844426, 0.004811, 0.535651, -0.062433],
            [-0.346844, -0.757130, 0.553583, 0.556552],
            [0.408220, -0.653247, -0.637671, 0.418854],
        ],
        [
            [0.999956, 0.001893, -0.009181, -0.302525],
            [0.008307, -0.632765, 0.774299, 0.419513],
            [-0.004344, -0.774341, -0.632753, 0.443692],
        ],
        [
            [0.999967, 0.002588, -0.007726, -0.140717],
            [0.007622, -0.632092, 0.774856, 0.420219],
            [-0.002878, -0.774889, -0.632091, 0.443089],
        ],
        [
            [0.999969, 0.004510, -0.006379, 0.020816],
            [0.007792, -0.633989, 0.773302, 0.421449],
            [-0.000556, -0.773329, -0.634005, 0.441574],
        ],
        [
            [0.999967, 0.004071, -0.007099, 0.182856],
            [0.008067, -0.636359, 0.771351, 0.422738],
            [-0.001377, -0.771383, -0.636370, 0.440008],
        ],
        [
            [0.836664, 0.004531, -0.547698, -0.058586],
            [0.359891, -0.758347, 0.543497, 0.556139],
            [-0.412883, -0.651836, -0.636112, 0.417604],
        ],
        [
            [0.641320, 0.007044, -0.767242, 0.076904],
            [0.011201, -0.999937, 0.000182, 0.591953],
            [-0.767192, -0.008711, -0.641358, 0.441740],
        ],
        [
            [0.641139, 0.006875, -0.767394, 0.076605],
            [0.011166, -0.999938, 0.000371, 0.707952],
            [-0.767344, -0.008807, -0.641176, 0.442779],
        ],
        [
            [0.641083, 0.006552, -0.767444, 0.076373],
            [0.010954, -0.999940, 0.000614, 0.824000],
            [-0.767394, -0.008800, -0.641116, 0.443779],
        ],
        [
            [0.841554, 0.005600, -0.540144, -0.057666],
            [-0.340476, -0.770805, -0.538458, 0.859143],
            [-0.419361, 0.637047, -0.646767, 0.420062],
        ],
        [
            [0.999992, 0.003858, 0.000883, 0.184077],
            [0.003178, -0.649876, -0.760033, 0.994317],
            [-0.002359, 0.760030, -0.649883, 0.444992],
        ],
        [
            [0.999991, 0.004052, 0.001151, 0.022246],
            [0.003509, -0.650144, -0.759803, 0.994120],
            [-0.002330, 0.759800, -0.650153, 0.445702],
        ],
        [
            [0.999991, 0.004126, 0.001072, -0.139612],
            [0.003498, -0.650453, -0.759538, 0.993762],
            [-0.002437, 0.759535, -0.650462, 0.446529],
        ],
        [
            [0.999992, 0.004006, 0.000803, -0.301331],
            [0.003216, -0.650673, -0.759351, 0.993369],
            [-0.002519, 0.759348, -0.650680, 0.447283],
        ],
        [
            [0.840019, 0.005723, 0.542527, -0.060964],
            [0.349132, -0.771111, -0.532442, 0.859650],
            [0.415302, 0.636675, -0.649746, 0.420525],
        ],
        [
            [0.999991, 0.003042, 0.003008, -0.059642],
            [0.003514, -0.183266, -0.983057, 0.979794],
            [-0.002440, 0.983059, -0.183275, 0.202631],
        ],
    ]
)


def reconstruct_mesh_using_tsdf_example():
    """
    Reconstructs the mesh of a metal bin from 22 depth images taken by a
    robot-held camera.

    The camera poses come from the robot and its hand-eye calibration and
    carry a few millimetres of error each. The skill first refines the poses
    so that the views agree, then fuses the depth images into one mesh,
    keeping only surface that at least two views measured.
    """
    # ===================== Load Data ==========================================
    # Depth in tenths of a millimetre; 0 marks pixels without a measurement.
    depth_images = [
        datatypes.DepthImage.from_url(DEPTH_IMAGE_URL.format(index), depth_scale=0.0001)
        for index in range(len(CAMERA_POSE_TOP_ROWS))
    ]
    camera_poses = np.tile(np.eye(4), (len(CAMERA_POSE_TOP_ROWS), 1, 1))
    camera_poses[:, :3] = CAMERA_POSE_TOP_ROWS
    # The bin with a 1 cm margin: [cx, cy, cz, width, height, depth, roll, pitch, yaw].
    region_of_interest = np.array(
        [
            -0.0609,
            0.7110,
            0.0570,
            0.6673,
            0.3683,
            0.1494,
            0.5037,
            0.2796,
            0.1838,
        ]
    )

    # ===================== Run Skill ==========================================
    # 2 mm voxels keep the example fast; 1 mm resolves finer detail.
    reconstructed_mesh, refined_poses = vitreous.reconstruct_mesh_using_tsdf(
        depth_images=depth_images,
        intrinsic_matrix=INTRINSIC_MATRIX,
        camera_to_world_transformation_matrices=camera_poses,
        region_of_interest_oriented_bounding_box=region_of_interest,
        downsampling_voxel_size=0.003,
        max_correspondence_distance=0.006,
        normal_search_radius=0.01,
        normal_max_neighbors=30,
        max_iterations=50,
        number_of_refinement_rounds=3,
        max_correction_translation=0.008,
        max_correction_rotation_degrees=2.5,
        fusion_voxel_size=0.002,
        truncation_distance=0.006,
        min_ray_surface_cosine=0.15,
        min_observation_weight=0.3,
        min_number_of_views=2,
    )

    # ===================== Log ================================================
    logger.success(
        f"Reconstructed a mesh from {len(depth_images)} depth images using TSDF fusion"
    )
    logger.success(f"Results: {reconstructed_mesh}")
    logger.info(
        f"Reconstructed mesh has {len(reconstructed_mesh.vertex_positions)} vertices "
        f"and {len(reconstructed_mesh.triangle_indices)} triangles"
    )
    # How far the refinement moved each camera.
    for index, (initial, refined) in enumerate(zip(camera_poses, refined_poses.data)):
        logger.info(
            f"Camera {index}: refinement moved it "
            f"{np.linalg.norm(refined[:3, 3] - initial[:3, 3]) * 1000:.2f} mm"
        )

    # ===================== Visualization  (Optional) ===========================
    rr.init("reconstruct_mesh_using_tsdf_example", spawn=True)
    # Back-project every measured pixel and place it with its initial pose.
    for index, (depth_image, camera_pose) in enumerate(zip(depth_images, camera_poses)):
        rows, columns = np.nonzero(depth_image.depth > 0)
        z = depth_image.depth[rows, columns]
        camera_points = np.column_stack(
            [
                (columns - INTRINSIC_MATRIX[0, 2]) * z / INTRINSIC_MATRIX[0, 0],
                (rows - INTRINSIC_MATRIX[1, 2]) * z / INTRINSIC_MATRIX[1, 1],
                z,
            ]
        )
        world_points = camera_points @ camera_pose[:3, :3].T + camera_pose[:3, 3]
        datatypes.visualize(
            datatypes.PointCloud(positions=world_points),
            entity_path=f"/1-input_depth_with_initial_poses/camera_{index}",
        )
    datatypes.visualize(reconstructed_mesh, entity_path="/2-tsdf_mesh")


if __name__ == "__main__":
    reconstruct_mesh_using_tsdf_example()
