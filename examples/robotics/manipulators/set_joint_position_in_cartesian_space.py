"""
Move to a target joint configuration along a Cartesian motion trajectory.

Supports Universal Robots (UR), Epson, virtual, and Isaac Sim.

Usage:
    python set_joint_position_in_cartesian_space.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None, prim_path: str | None) -> None:
    """Move to a target joint configuration along a Cartesian trajectory."""

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

        #===================== Prepare Target ==========================================
        # Target: current joint configuration with the base joint rotated
        target_joint_positions = robot.get_joint_positions().copy()
        target_joint_positions[0] += 5

        # ==================== Run Skill ============================================
        robot.set_joint_position_in_cartesian_space(
            joint_positions=target_joint_positions,
            speed=1.05,
            acceleration=1.4,
        )
        logger.info(f"Moved to target joint positions: {target_joint_positions}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move to a target joint configuration along a Cartesian trajectory")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/ur10e"')
    args = parser.parse_args()

    main(ip=args.ip, prim_path=args.prim_path)
