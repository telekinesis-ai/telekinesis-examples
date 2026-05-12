"""
Tool contact polling example for the Synapse SDK.

``is_tool_in_contact`` is the low-level contact-detection primitive on UR
— it only returns ``True`` **while the robot is actively executing
motion**. On an idle robot it always returns ``False``, so the usual
pattern is to poll it from inside a streaming-motion loop.

This example starts a slow Cartesian jog toward -Z and polls
``is_tool_in_contact`` each tick. When contact is detected (or a safety
timeout elapses), it issues ``stop_jog`` to halt the motion.

For the one-shot equivalent (start a move, block until contact, stop),
use ``move_until_contact`` instead.

Currently supported only for Universal Robots (UR10e).

Usage:
    python is_tool_in_contact.py --ip <ROBOT_IP>
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Jog the TCP toward -Z and stop the instant contact is detected."""

    # Motion parameters
    cartesian_velocity = [0.0, 0.0, -0.05, 0.0, 0.0, 0.0]  # -Z at 5 cm/s in base
    direction = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]            # contact axis matches motion
    poll_dt = 0.005          # 200 Hz polling
    safety_timeout = 5.0     # stop after this long even if no contact

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    try:
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

    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool contact polling Synapse example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
