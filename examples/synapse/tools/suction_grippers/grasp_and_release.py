"""Piab PiCobot Electric grasp and release example for the Synapse SDK.

Supported only for real hardware. Only URCAP protocol is supported currently
via the Piab URCap XML-RPC service.

Usage:
    python grasp_and_release.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger
from telekinesis.synapse.tools.suction_grippers import piab


def main(ip: str, protocol: str) -> None:
    """
    This example demonstrates how to grasp and release an object using a Piab
    PiCobot Electric vacuum gripper via the Synapse SDK.
    """

    gripper = piab.PiabPiCobotElectric()

    try:
        gripper.connect(ip=ip, protocol=protocol)
        gripper.set_vacuum_level(vacuum_level=30, unit="percentage")
        gripper.grasp()
        input("Press Enter to release...")
        gripper.release()

    except Exception as e:
        logger.error(f"An error occurred while operating the Piab gripper: {e}")

    finally:
        try:
            gripper.disconnect()
        except Exception as e:
            logger.warning(f"Failed to disconnect cleanly from Piab: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Piab PiCobot Electric grasp/release example")
    p.add_argument("--protocol", choices=["URCAP"], default="URCAP")
    p.add_argument("--ip", default="192.168.2.2", help="IP for Robot Controller")
    args = p.parse_args()

    main(ip=args.ip, protocol=args.protocol)
