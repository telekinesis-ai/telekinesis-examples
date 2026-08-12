"""Piab PiCobot Electric connect and disconnect example for the Synapse SDK.

Supported only for real hardware. Only URCAP protocol is supported currently
via the Piab URCap XML-RPC service.

Usage:
    python connection_and_disconnection.py --ip <ROBOT_IP>
"""

import argparse
import time
from telekinesis.synapse.tools.suction_grippers import piab

def main(ip: str, protocol: str) -> None:
    """
    This example demonstrates how to connect and disconnect a Piab PiCobot Electric gripper
    using the Synapse SDK. 
    """

    gripper = piab.PiabPiCobotElectric()
    gripper.connect(ip=ip, protocol=protocol)
    time.sleep(2.0)
    gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Piab PiCobot Electric connect/disconnect example")
    p.add_argument("--protocol", choices=["URCAP"], default="URCAP")
    p.add_argument("--ip", default="192.168.2.2", help="IP for Robot Controller")
    args = p.parse_args()

    main(ip=args.ip, protocol=args.protocol)
