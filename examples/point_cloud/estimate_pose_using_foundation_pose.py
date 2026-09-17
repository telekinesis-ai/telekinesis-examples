"""
Demonstrates estimating a 6-DOF object pose from an RGB-D frame and a CAD
mesh using FoundationPose.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import vitreous, datatypes


def estimate_pose_using_foundation_pose_example():
    """
    Registers a pose using the exported RGB-D frame, mesh, and mask.
    """
    # ===================== Load Data ==========================================
    rgb_image_url = "https://assets.telekinesis.ai/examples/v1/images/male_pin_connector_binpicking.png"
    rgb_image = datatypes.Image.from_url(rgb_image_url)
    depth_image_url = "https://assets.telekinesis.ai/examples/v1/depth_images/male_pin_connector_binpicking.png"
    depth_image = datatypes.DepthImage.from_url(depth_image_url, depth_scale=0.001)
    mask_url = "https://assets.telekinesis.ai/examples/v1/images/male_pin_connector_binpicking_mask.png"
    mask = datatypes.SegmentationImage.from_url(mask_url)
    # The exported OBJ is already in meters.
    mesh_url = "https://assets.telekinesis.ai/examples/v1/meshes/male_pin_connector.obj"
    mesh = datatypes.Mesh3D.from_url(mesh_url)
    camera_calibration = datatypes.CameraCalibration(
        width=848,
        height=480,
        distortion_model="plumb_bob",
        distortion_parameters=[0.0, 0.0, 0.0, 0.0, 0.0],
        intrinsic_matrix=[
            435.46856689453125, 0.0, 420.7252502441406,
            0.0, 434.5237731933594, 244.55136108398438,
            0.0, 0.0, 1.0,
        ],
    )

    # ===================== Run Skill ==========================================
    # Frame 1: register from scratch using the mask.
    pose, tracking_pose = vitreous.estimate_pose_using_foundation_pose(
        rgb_image=rgb_image,
        depth_image=depth_image,
        mesh=mesh,
        camera_calibration=camera_calibration,
        mask=mask,
        refine_iterations=2,
    )


    # ===================== Log ================================================
    logger.success(f"Estimated pose for {mesh}")
    logger.success(f"Results: {pose}")
    logger.info(f"Pose matrix:\n{pose.data}")

    # ===================== Visualization  (Optional) ===========================
    # Back-project the depth frame into a colored point cloud in the same
    # camera-frame coordinates as `pose`, so the estimated pose can be seen
    # in context against the actual scene geometry.
    raw_point_cloud = vitreous.convert_depth_image_to_point_cloud(
        depth_image=depth_image,
        intrinsic_matrix=camera_calibration.intrinsic_matrix,
    )

    # Pixels with no depth return (0, ~14% of this frame) back-project to
    # exactly (0, 0, 0) - the camera origin - since x/y/z all scale with a
    # zero depth. Left in, that's a ~58k-point spike sitting on top of the
    # camera that dominates rerun's auto view-fit and makes the real,
    # tiny, far-away geometry look squashed/misplaced. Drop them.
    valid = depth_image.depth.reshape(-1) > 0
    point_cloud = datatypes.PointCloud(
        positions=raw_point_cloud.positions[valid],
        colors=rgb_image.data.reshape(-1, 3)[valid],
    )

    # `mesh`'s vertices are in the mesh's own (uncentered) coordinate
    # system, so apply `pose` (the object-to-camera transform) to move
    # them into the point cloud's camera frame for a correct overlay.
    rotation, translation = pose.data[:3, :3], pose.data[:3, 3]
    mesh_in_camera_frame = datatypes.Mesh3D(
        vertex_positions=mesh.vertex_positions @ rotation.T + translation,
        triangle_indices=mesh.triangle_indices,
        vertex_normals=(
            mesh.vertex_normals @ rotation.T if mesh.has_vertex_normals else None
        ),
    )

    rr.init("estimate_pose_using_foundation_pose_example", spawn=True)
    # Keep the image plane near the camera so it does not obscure the mesh.
    rr.log(
        "/1-camera",
        rr.Pinhole(
            image_from_camera=camera_calibration.intrinsic_matrix,
            resolution=[camera_calibration.width, camera_calibration.height],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=0.05,
        ),
    )
    datatypes.visualize(rgb_image, entity_path="/1-camera/rgb_image")
    datatypes.visualize(depth_image, entity_path="/1-camera/depth_image")
    datatypes.visualize(mask, entity_path="/1-camera/mask")
    datatypes.visualize(point_cloud, entity_path="/5-point_cloud")
    rr.log(
        "/5-point_cloud/pose",
        rr.Arrows3D(
            origins=np.tile(translation, (3, 1)),
            vectors=rotation.T * 0.2,  # 2 cm axes
            colors=[[255, 0, 0], [0, 255, 0], [0, 128, 255]],
            radii=0.0005,
            labels=["X", "Y", "Z"],
        ),
    )
    rr.log(
        "/5-point_cloud/mesh",
        rr.Mesh3D(
            vertex_positions=mesh_in_camera_frame.vertex_positions,
            triangle_indices=mesh_in_camera_frame.triangle_indices,
            albedo_factor=[255, 190, 30, 255],
        ),
    )



if __name__ == "__main__":
    estimate_pose_using_foundation_pose_example()
