"""
Start an asynchronous joint move, pause it mid-motion, then continue it.

Supports Epson, virtual, and Isaac Sim.

Usage:
    python pause_and_continue_motion.py [--ip <ROBOT_IP>] [--prim_path <PRIM_PATH>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import epson


def main(ip: str | None, prim_path: str | None) -> None:
    """Start an async joint move, pause it with pause_motion and resume it with continue_motion."""

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

        #===================== Prepare Target ==========================================
        target_joint_positions = list(robot.get_joint_positions())
        target_joint_positions[0] += 30  # Asynchronous +30 deg move on joint 0

        robot.set_joint_positions(
            joint_positions=target_joint_positions,
            speed=20,
            acceleration=40,
            asynchronous=True,
        )

        # ==================== Run Skill ============================================
        # Let the move run briefly, then pause it without discarding the remaining path
        time.sleep(0.5)
        robot.pause_motion()
        logger.info(f"Paused at joint positions: {robot.get_joint_positions()}")

        time.sleep(2.0)

        # Confirm the cell is safe before continuing on real hardware
        robot.continue_motion()
        logger.info("Continued motion.")

        # Give the remaining path time to finish
        time.sleep(3.0)
        logger.success(f"Final joint positions: {robot.get_joint_positions()}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pause and continue an asynchronous motion")
    parser.add_argument("--ip", type=str, default=None,
                         help="Epson controller IP address for real hardware, e.g. 192.168.0.1")
    parser.add_argument("--prim_path", type=str, default=None,
                         help='Isaac Sim articulation prim path, e.g. "/World/epson_cx4a601s"')
    args = parser.parse_args()

    main(ip=args.ip, prim_path=args.prim_path)
