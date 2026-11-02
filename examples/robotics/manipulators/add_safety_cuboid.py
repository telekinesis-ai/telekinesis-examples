"""
Restrict the TCP to an axis-aligned cuboid, then try moves inside and outside it.

Supports Epson, virtual, and Isaac Sim.

Usage:
    python add_safety_cuboid.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import epson


def main(ip: str | None, prim_path: str | None) -> None:
    """Add a safety cuboid around the current TCP and check that moves outside it are rejected."""

    #===================== Create Robot ==========================================
    robot = epson.EpsonCX4A601S(name='EpsonCX4A601S')

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
        # Allowed TCP workspace: a 20 cm cube centred on the current TCP position [m]
        x, y, z = robot.get_cartesian_pose()[:3]
        min_corner = [x - 0.1, y - 0.1, z - 0.1]
        max_corner = [x + 0.1, y + 0.1, z + 0.1]
        robot.add_safety_cuboid(min_corner=min_corner, max_corner=max_corner)
        logger.info(f"Safety cuboid set: min={min_corner}, max={max_corner}")

        # A move that stays inside the cuboid is executed
        inside_pose = robot.get_cartesian_pose().copy()
        inside_pose[2] += 0.05
        robot.set_cartesian_pose(cartesian_pose=inside_pose, speed=0.1, acceleration=0.5)
        logger.success(f"Moved inside the cuboid to: {inside_pose}")

        # A move that leaves the cuboid is rejected
        outside_pose = robot.get_cartesian_pose().copy()
        outside_pose[2] += 0.2
        try:
            robot.set_cartesian_pose(cartesian_pose=outside_pose, speed=0.1, acceleration=0.5)
            logger.warning(f"Move outside the cuboid was not rejected: {outside_pose}")
        except (ValueError, RuntimeError) as e:
            logger.success(f"Move outside the cuboid rejected: {e}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restrict the TCP workspace with a safety cuboid")
    parser.add_argument("--ip", type=str, default=None,
                         help="Epson controller IP address for real hardware, e.g. 192.168.0.1")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/epson_cx4a601s"')
    args = parser.parse_args()

    main(ip=args.ip, prim_path=args.prim_path)
