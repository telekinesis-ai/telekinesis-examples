"""
Commands a circular-arc move from the current TCP pose to an offset target pose using servo_circular.

Supports Universal Robots (UR).

Usage:
    python servo_circular.py [--ip <ROBOT_IP>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Drive a circular arc from the current TCP pose to an offset target."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    # Live: subscribes to the robot's state topic and redraws as it moves.
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)

        #===================== Prepare Target ==========================================
        # Target pose: 2 cm out in Y and 2 cm down in Z from the current pose.
        current = robot.get_cartesian_pose()
        target = list(current)
        target[1] += 0.02
        target[2] -= 0.02

        # ==================== Run Skill ============================================
        logger.warning(
            f"About to move real robot along a circular arc from {current} to {target}. "
            "Make sure it's safe to move there."
        )
        logger.info(f"servo_circular target: {target}")

        # Command the circular servo move, then stop streaming to end the motion.
        robot.servo_circular(
            pose=target,
            speed=0.1,
            acceleration=0.1,
            blend=0.0,
        )

        # In a real application, you would typically stream continuously until some
        # condition is met (e.g. a certain time has elapsed, or a sensor triggers).
        time.sleep(2.0)
        robot.servo_stop()
        logger.success("servo_circular complete.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR10e servo_circular example")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    args = parser.parse_args()

    main(ip=args.ip)
