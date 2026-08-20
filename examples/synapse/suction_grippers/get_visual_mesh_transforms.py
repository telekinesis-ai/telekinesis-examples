"""
Demonstrates how to read the world transforms of every visual mesh of a suction
gripper kinematics model for the Synapse SDK.

Usage:
    python get_visual_mesh_transforms.py
"""

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import piab
from telekinesis.synapse import utils


def main() -> None:
    """Reads the visual mesh world transform of every link of a Piab gripper."""

    #===================== Create Gripper ======================================
    gripper = piab.PiabPiCobotElectric()

    # ==================== Run Skill ===========================================
    transforms = gripper.get_visual_mesh_transforms(base_transform=np.eye(4))
    logger.info(f"Number of links with visual meshes: {len(transforms)}")

    # =================== Visualization (Optional) ==============================
    rr.init(f"telekinesis_synapse_{type(gripper).__name__}", spawn=True)
    gripper.visualize_rerun(recording_stream=rr.get_global_data_recording())

    for name, transform in transforms.items():
        pose = utils.transformation_matrix_to_pose(transform, rot_type="deg")
        logger.success(f"{name}: pose [m, deg] = {pose}")

        origin = transform[:3, 3]
        rr.log(f"/visual_mesh_frames/{name}",
               rr.Arrows3D(origins=[origin, origin, origin],
                           vectors=[transform[:3, 0] * 0.02,
                                    transform[:3, 1] * 0.02,
                                    transform[:3, 2] * 0.02],
                           colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))


if __name__ == "__main__":
    main()
