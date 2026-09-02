"""
Attaches an OnRobot RG6 gripper to a UR10e, registers its TCP, and visualizes it in Rerun.

Supports Universal Robots (UR), Epson, virtual, and Isaac Sim.

On hardware the attachment is for co-visualization only. In Isaac Sim the tool
is additionally fixed to the arm's flange in the simulation, but only if the
tool is itself connected to the simulation — pass ``--gripper_prim_path`` for
that. Without it the gripper is drawn in Rerun and never assembled onto the arm.

Usage:
    python attach_tool.py [--ip <ROBOT_IP>]
    python attach_tool.py --prim_path <PRIM_PATH> [--gripper_prim_path <GRIPPER_PRIM_PATH>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot


def main(ip: str | None,
         prim_path: str | None,
         gripper_prim_path: str | None) -> None:
    """Attach an OnRobot RG6 gripper to a UR10e and visualize in Rerun."""

    #===================== Create Robot and Gripper =============================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    gripper = onrobot.OnRobotRG6()

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)

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
        # UR robots declare simulation_mount_link = "wrist_3_link", so the
        # default mount frame is a real rigid-body link the simulation accepts.
        robot.attach_tool(gripper)
        robot.add_tcp(name="gripper_tip",
                      transform=[0.0, 0.0, 0.18, 0.0, 0.0, 0.0],
                      set_active=True)
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        if gripper.is_connected:
            gripper.disconnect()
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attach a gripper to the robot and visualize it")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/ur10e"')
    parser.add_argument("--gripper_prim_path", type=str, default=None,
                         help='Isaac Sim gripper prim path, e.g. "/World/onrobot_rg6". '
                              'Required to fix the tool to the arm in simulation.')
    args = parser.parse_args()

    main(ip=args.ip,
         prim_path=args.prim_path,
         gripper_prim_path=args.gripper_prim_path)
