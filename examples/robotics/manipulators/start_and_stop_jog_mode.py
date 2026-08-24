"""
Jogs the TCP +Z at 5 cm/s in the base frame for 5 seconds, then stops.

Supports Universal Robots (UR).

Usage:
    python start_and_stop_jog_mode.py [--ip <ROBOT_IP>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Jog the TCP +Z (upward) at 5 cm/s in the base frame for 5 seconds, then stop."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        # Cartesian twist [vx, vy, vz (m/s), ωx, ωy, ωz (deg/s)] in the base frame
        cartesian_velocity = [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]
        logger.info(f"Starting jog - cartesian_velocity [m/s, deg/s]: {cartesian_velocity}")
        robot.start_jog(
            cartesian_velocity=cartesian_velocity,
            feature=0,
            cartesian_acceleration=0.5,
        )

        # Let the jog run, then stop
        time.sleep(5.0)
        robot.stop_jog()
        logger.success("Jog mode stopped.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Cartesian jog mode, then stop it")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    args = parser.parse_args()

    main(ip=args.ip)
