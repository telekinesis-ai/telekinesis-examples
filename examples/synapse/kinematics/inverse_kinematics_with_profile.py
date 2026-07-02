"""
Inverse Kinematics with profiling diagnostics for the Synapse SDK.

Pass ``profile=True`` to return a ``(q, timing)`` tuple, where ``timing`` reports
the total solve time (``total_s``), the number of seeds tried
(``num_seeds_tried``), the winning seed index (``winning_seed_index``), and the
separate residual linear and angular errors (``linear_error_norm_meters`` and
``angular_error_norm_rad``).

Universal Robots (UR10e) is used here purely for illustration; the same API works for all supported robots.

Usage:
    python inverse_kinematics_with_profile.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Solve IK with ``profile=True`` and log the timing diagnostics."""

    # Create the robot (no connect required — IK runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Solve IK with profiling enabled to return (q, timing) and log both
    target_pose = [0.5, 0.2, 0.3, 180.0, 0.0, 0.0]
    try:
        q, timing = robot.inverse_kinematics(
            target_pose=target_pose,
            profile=True,
        )
        logger.success(f"IK solution: {q}")
        logger.info(f"Total time: {timing['total_s']:.4f} s")
        logger.info(f"Seeds tried: {timing['num_seeds_tried']}")
        logger.info(f"Winning seed: {timing['winning_seed_index']}")
        logger.info(f"Linear error (m):  {timing['linear_error_norm_meters']:.6f}")
        logger.info(f"Angular error (rad): {timing['angular_error_norm_rad']:.6f}")
    except (RuntimeError, TypeError, ValueError) as e:
        logger.error(f"IK failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
