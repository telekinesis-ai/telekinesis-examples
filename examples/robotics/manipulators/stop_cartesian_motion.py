"""
Commands an asynchronous Cartesian move and interrupts it mid-trajectory with stop_cartesian_motion.

Supports Universal Robots (UR), Epson, and Isaac Sim.

Usage:
    python stop_cartesian_motion.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None, prim_path: str | None) -> None:
    """Start an async Cartesian move and interrupt it with stop_cartesian_motion."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    # Live: subscribes to the robot's state topic and redraws as it moves.
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)
        elif prim_path:
            robot.connect(simulation_prim_path=prim_path)

        #===================== Prepare Target ==========================================
        # Get initial Cartesian pose [x, y, z, rx, ry, rz] (m, deg)
        actual_pose = robot.get_cartesian_pose()
        target_pose = list(actual_pose)
        target_pose[2] += 0.15  # Asynchronous +15 cm move along Z

        # ==================== Run Skill ============================================
        robot.set_cartesian_pose(
            cartesian_pose=target_pose,
            speed=0.25,
            acceleration=0.5,
            asynchronous=True,
        )

        # Let the move run briefly, then interrupt it
        time.sleep(0.3)
        robot.stop_cartesian_motion(stopping_speed=0.25)
        logger.info("Stopped Cartesian motion.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interrupt an async Cartesian move with stop_cartesian_motion")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/ur10e"')
    args = parser.parse_args()

    main(ip=args.ip, prim_path=args.prim_path)
