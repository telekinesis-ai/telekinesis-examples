"""
Demonstrates estimating multiple 6-DOF object poses from one RGB-D frame
using the multi-object FoundationPose backend.
"""

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes, vitreous


OBJECT_COUNT = 2


def estimate_multi_object_pose_using_foundation_pose_example():
    """Register multiple masked objects against one shared RGB-D frame."""
    # ===================== Load Data ==========================================
    rgb_image_url = "https://assets.telekinesis.ai/examples/v1/images/male_pin_connector_binpicking.png"
    rgb_image = datatypes.Image.from_url(rgb_image_url)
    depth_image_url = "https://assets.telekinesis.ai/examples/v1/depth_images/male_pin_connector_binpicking.png"
    depth_image = datatypes.DepthImage.from_url(
        depth_image_url, depth_scale=0.001
    )
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
            435.46856689453125,
            0.0,
            420.7252502441406,
            0.0,
            434.5237731933594,
            244.55136108398438,
            0.0,
            0.0,
            1.0,
        ],
    )

    # Repeat the available object asset to demonstrate the multi-object input.
    # Each mesh corresponds to the mask at the same list index.
    meshes = [mesh] * OBJECT_COUNT
    masks = [mask] * OBJECT_COUNT

    # ===================== Run Skill ==========================================
    poses, scores, statuses = (
        vitreous.estimate_multi_object_pose_using_foundation_pose(
            rgb_image=rgb_image,
            depth_image=depth_image,
            meshes=meshes,
            masks=masks,
            camera_calibration=camera_calibration,
        )
    )

    # ===================== Log ================================================
    logger.success(f"Estimated {len(poses)} object poses")
    logger.info(f"Pose matrices:\n{poses.data}")
    logger.info(f"Scores:\n{scores.data}")
    logger.info(f"Statuses:\n{statuses.data}")

    # ===================== Visualization (Optional) ===========================
    raw_point_cloud = vitreous.convert_depth_image_to_point_cloud(
        depth_image=depth_image,
        intrinsic_matrix=camera_calibration.intrinsic_matrix,
    )
    valid = depth_image.depth.reshape(-1) > 0
    point_cloud = datatypes.PointCloud(
        positions=raw_point_cloud.positions[valid],
        colors=rgb_image.data.reshape(-1, 3)[valid],
    )

    rr.init(
        "estimate_multi_object_pose_using_foundation_pose_example",
        spawn=True,
    )
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
    datatypes.visualize(point_cloud, entity_path="/5-point_cloud")

    for object_index in range(OBJECT_COUNT):
        pose = poses[object_index]
        object_mesh = meshes[object_index]
        rotation = pose.data[:3, :3]
        translation = pose.data[:3, 3]
        mesh_in_camera_frame = datatypes.Mesh3D(
            vertex_positions=(
                object_mesh.vertex_positions @ rotation.T + translation
            ),
            triangle_indices=object_mesh.triangle_indices,
            vertex_normals=(
                object_mesh.vertex_normals @ rotation.T
                if object_mesh.has_vertex_normals
                else None
            ),
        )
        object_path = f"/5-point_cloud/objects/{object_index}"

        datatypes.visualize(
            masks[object_index],
            entity_path=f"/1-camera/masks/{object_index}",
        )
        rr.log(
            f"{object_path}/pose",
            rr.Arrows3D(
                origins=np.tile(translation, (3, 1)),
                vectors=rotation.T * 0.2,
                colors=[[255, 0, 0], [0, 255, 0], [0, 128, 255]],
                radii=0.0005,
                labels=["X", "Y", "Z"],
            ),
        )
        rr.log(
            f"{object_path}/mesh",
            rr.Mesh3D(
                vertex_positions=mesh_in_camera_frame.vertex_positions,
                triangle_indices=mesh_in_camera_frame.triangle_indices,
                albedo_factor=[255, 190, 30, 255],
            ),
        )


if __name__ == "__main__":
    estimate_multi_object_pose_using_foundation_pose_example()
