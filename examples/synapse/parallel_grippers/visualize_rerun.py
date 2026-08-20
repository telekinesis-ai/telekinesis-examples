"""
Demonstrates how to visualize the meshes of a gripper kinematics model in
Rerun for the Synapse SDK.

Usage:
    python visualize_rerun.py
"""

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main() -> None:
    """Visualizes a Robotiq gripper in Rerun while translating it along +X."""

    #===================== Create Gripper ======================================
    gripper = robotiq.Robotiq2F85()

    # =================== Visualization (Optional) ==============================
    # The first call uploads the static meshes; subsequent calls only update
    # the transforms.
    rr.init(f"telekinesis_synapse_{type(gripper).__name__}", spawn=True)
    gripper.visualize_rerun(recording_stream=rr.get_global_data_recording())
    logger.success("Gripper meshes logged at the world origin.")

    # ==================== Run Skill ===========================================
    for step in range(10):
        tcp_transform = np.eye(4)
        tcp_transform[0, 3] = 0.01 * step
        gripper.visualize_rerun(tcp_transform=tcp_transform,
                                recording_stream=rr.get_global_data_recording())

    logger.success("Gripper transform updated over 10 steps.")


if __name__ == "__main__":
    main()
