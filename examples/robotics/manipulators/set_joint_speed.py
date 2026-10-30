"""
Set the default joint speed, then make a joint move that uses it.

Supports Universal Robots (UR), Epson, virtual, and Isaac Sim.

Usage:
    python set_joint_speed.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None, prim_path: str | None) -> None:
    """Set the default joint speed and move with it."""

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
            robot.set_joint_positions(robot.default_joint_configuration)

        # ==================== Run Skill ============================================
        # Joint speed of the leading axis [deg/s] for joint moves called without one
        robot.set_joint_speed(speed=30)
        logger.info("Default joint speed set to 30 deg/s.")

        #===================== Move With Defaults ==========================================
        target_joint_positions = list(robot.get_joint_positions())
        target_joint_positions[0] += 20  # +20 deg on joint 0

        # Called without speed/acceleration, so the defaults set above are used
        robot.set_joint_positions(joint_positions=target_joint_positions)
        logger.success(f"Moved to target joint positions: {target_joint_positions}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set the default joint speed and move with it.")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/ur10e"')
    args = parser.parse_args()

    main(ip=args.ip, prim_path=args.prim_path)
