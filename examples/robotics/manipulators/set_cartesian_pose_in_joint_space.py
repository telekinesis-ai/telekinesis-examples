"""
Move the TCP to a target pose along a joint-space-linear trajectory.

Supports Universal Robots (UR), Epson, virtual, and Isaac Sim.

Usage:
    python set_cartesian_pose_in_joint_space.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None, prim_path: str | None) -> None:
    """Move the TCP to a target pose along a joint-space-linear trajectory."""

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
        current_cartesian_pose = robot.get_cartesian_pose()
        target_cartesian_pose = current_cartesian_pose.copy()
        target_cartesian_pose[2] += 0.1  # Move 10 cm up in Z

        # ==================== Run Skill ============================================
        robot.set_cartesian_pose_in_joint_space(
            cartesian_pose=target_cartesian_pose,
            speed=60,
            acceleration=80,
        )
        logger.info(f"Moved to target Cartesian pose: {target_cartesian_pose}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move the TCP to a target pose along a joint-space trajectory")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/ur10e"')
    args = parser.parse_args()

    main(ip=args.ip, prim_path=args.prim_path)
