"""
Demonstrates how to read the base-frame transforms of every link in a gripper
kinematics model for the Synapse SDK.

Supports all.

Usage:
    python get_link_transforms.py
"""

import rerun as rr
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import onrobot
from telekinesis.synapse import utils


def main() -> None:
    """Reads the base-frame transform of every link of an OnRobot gripper."""

    #===================== Create Gripper ======================================
    gripper = onrobot.OnRobotRG6()

    # ==================== Run Skill ===========================================
    transforms = gripper.get_link_transforms()
    logger.info(f"Number of frames: {len(transforms)}")

    # =================== Visualization (Optional) ==============================
    rr.init(f"telekinesis_synapse_{type(gripper).__name__}", spawn=True)
    gripper.visualize_rerun(recording_stream=rr.get_global_data_recording())

    for name, transform in transforms.items():
        pose = utils.transformation_matrix_to_pose(transform, rot_type="deg")
        logger.success(f"{name}: pose [m, deg] = {pose}")

        origin = transform[:3, 3]
        rr.log(f"/frames/{name}",
               rr.Arrows3D(origins=[origin, origin, origin],
                           vectors=[transform[:3, 0] * 0.02,
                                    transform[:3, 1] * 0.02,
                                    transform[:3, 2] * 0.02],
                           colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))

if __name__ == "__main__":
    main()
