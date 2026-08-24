"""
Jogs the TCP toward -Z and polls is_tool_in_contact each tick, stopping the instant contact is detected.

Supports Universal Robots (UR).

Usage:
    python is_tool_in_contact.py [--ip <ROBOT_IP>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Jog the TCP toward -Z and stop the instant contact is detected."""
    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    
    # Motion parameters
    cartesian_velocity = [0.0, 0.0, -0.05, 0.0, 0.0, 0.0]  # -Z at 5 cm/s in base
    direction = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]            # contact axis matches motion
    poll_dt = 0.005          # 200 Hz polling
    safety_timeout = 5.0     # stop after this long even if no contact

    

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        # Start the jog. is_tool_in_contact only returns True while moving.
        logger.info(f"Starting jog along -Z at {abs(cartesian_velocity[2])} m/s")
        robot.start_jog(
            cartesian_velocity=cartesian_velocity,
            feature=0,
            cartesian_acceleration=0.5,
        )

        # Poll for contact and stop as soon as it's detected.
        t0 = time.monotonic()
        contact = False
        while time.monotonic() - t0 < safety_timeout:
            if robot.is_tool_in_contact(direction=direction):
                contact = True
                break
            time.sleep(poll_dt)

        # Halt the motion regardless of how the loop exited.
        robot.stop_jog()
        if contact:
            logger.success(f"Contact detected after {time.monotonic() - t0:.3f} s — jog stopped.")
        else:
            logger.warning(f"No contact within {safety_timeout} s — jog stopped on timeout.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool contact polling Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
