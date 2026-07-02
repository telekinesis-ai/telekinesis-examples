"""
Collision-free Inverse Kinematics example for the Synapse SDK.

Passing ``collision_free_ik=True`` discards solutions that self-collide,
so the returned configuration is guaranteed to be self-collision-free
(where multiple joint solutions exist for the same TCP pose, a colliding
one is rejected in favour of a clear one).

Universal Robots (UR10e) is used here purely for illustration; the same API works for all supported robots.

Usage:
    python inverse_kinematics_collision_free.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Solve IK with self-collision filtering enabled."""

    # Create the robot (no connect required — IK runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Solve IK for a fixed target pose [x, y, z, rx, ry, rz] (m, deg), rejecting
    # solutions that self-collide.
    target_pose = [0.5, 0.2, 0.3, 180.0, 0.0, 0.0]
    try:
        q = robot.inverse_kinematics(
            target_pose=target_pose,
            collision_free_ik=True,
        )
        logger.success(f"Collision-free IK solution: {q}")
    except (RuntimeError, TypeError, ValueError) as e:
        logger.error(f"IK failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
