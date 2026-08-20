"""
Demonstrates how to visualize the meshes of a gripper kinematics model in
Rerun for the Synapse SDK.

Supports all.

Usage:
    python visualize_rerun.py
"""

import rerun as rr

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


if __name__ == "__main__":
    main()
