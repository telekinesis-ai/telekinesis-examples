"""
Attaches a gripper and visualizes it, then detaches it from the Rerun visualization again.

Supports Universal Robots (UR), Epson, virtual, and Isaac Sim.

Passing ``--gripper_prim_path`` connects the simulated gripper, so attach_tool()
also fixes it to the arm's flange in Isaac Sim. Note that detach_tool() affects
the visualization only: a tool assembled onto the arm in the simulation stays
assembled, because the simulation offers no way to take it back off.

Usage:
    python detach_tool.py [--ip <ROBOT_IP>]
    python detach_tool.py --prim_path <PRIM_PATH> [--gripper_prim_path <GRIPPER_PRIM_PATH>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import schunk


def main(ip: str | None,
         prim_path: str | None,
         gripper_prim_path: str | None) -> None:
    """Attach a gripper, visualize it, then detach it again."""

    #===================== Create Robot and Gripper =============================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    gripper = schunk.SchunkEGU50()

    # ==================== Visualization (Optional) ================================
    robot.visualize_rerun()

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)
        elif prim_path:
            robot.connect(simulation_prim_path=prim_path)
            robot.set_joint_positions(robot.default_joint_configuration)

        #===================== Connect Gripper (Isaac Sim) ============================
        # attach_tool() fixes the tool to the flange in the simulation only when
        # the tool reports the ISAACSIM protocol, which it does after connecting
        # to its own prim. On hardware the gripper needs no connection here: the
        # attachment is for co-visualization only.
        if gripper_prim_path:
            gripper.connect(simulation_prim_path=gripper_prim_path)
        elif ip:
            logger.warning("Running on real hardware: attach_tool() registers the tool "
                           "for co-visualization only. Nothing is physically attached — "
                           "mount the gripper on the flange yourself.")
        elif prim_path:
            logger.warning("Connected to Isaac Sim without --gripper_prim_path: the "
                           "gripper is only drawn in Rerun, not fixed to the arm in the "
                           "simulation. Pass --gripper_prim_path to assemble it.")

        # ==================== Run Skill ============================================
        robot.attach_tool(gripper)
        time.sleep(2)
        logger.info("Gripper attached and visualized.")

        # detach_tool() clears the tool from the Rerun recording. It never undoes
        # a physical mount, nor an Isaac Sim assembly.
        if ip:
            logger.warning("Running on real hardware: detach_tool() clears the "
                           "co-visualization only. The physically mounted gripper is "
                           "untouched — remove it from the flange yourself.")
        elif gripper_prim_path:
            logger.warning("Isaac Sim: detach_tool() clears the visualization only. The "
                           "gripper stays fixed to the arm in the simulation — it offers "
                           "no way to take an assembled tool back off.")

        robot.detach_tool()
        logger.success("Gripper detached from visualization.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        if gripper.is_connected:
            gripper.disconnect()
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detach a tool from the robot")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/ur10e"')
    parser.add_argument("--gripper_prim_path", type=str, default=None,
                         help='Isaac Sim gripper prim path, e.g. "/World/schunk_egu50". '
                              'Required to fix the tool to the arm in simulation.')
    args = parser.parse_args()

    main(ip=args.ip,
         prim_path=args.prim_path,
         gripper_prim_path=args.gripper_prim_path)
