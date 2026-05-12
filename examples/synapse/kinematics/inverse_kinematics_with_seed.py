"""
Inverse Kinematics with an explicit seed for the Synapse SDK.

``inverse_kinematics`` is defined on the abstract manipulator and supports
all robots.

Universal Robots (UR10e) is used here purely for illustration.
The solver runs purely on the kinematic model, so this example does not
connect to hardware and no ``--ip`` is required.

Usage:
    python inverse_kinematics_with_seed.py
"""

import numpy as np
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Solve IK with an explicit initial joint seed. Supports all robots."""

    # Create the robot (no connect required — IK runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Pick a known-reachable configuration and derive its forward-kinematic pose
    q_deg = np.asarray([0, -90 - 5.7, -90, 0, 90, 0], dtype=float)
    target_pose = robot.forward_kinematics(q_deg)

    # Perturb the configuration to use as an IK seed
    q_init = q_deg + np.random.normal(loc=0.0, scale=30.0, size=q_deg.shape)

    # Solve IK with the explicit seed and report the result
    try:
        q = robot.inverse_kinematics(
            target_pose=target_pose,
            q_init=q_init,
            solver="multi_start_clik",
        )
        logger.success(f"IK solution: {q}")
    except (RuntimeError, TypeError, ValueError) as e:
        logger.error(f"IK failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
