#!/usr/bin/env python3
"""Demo for the Schunk EGU50 gripper wrapper.

The Schunk hardware interface is not yet implemented, so every call below
currently raises :class:`NotImplementedError`. This file serves as a
template for the intended usage once the backend lands.

Demonstates:
- ``connect()``, ``open()``, ``close()``, and ``move()`` methods.

Usage:
    python schunk_egu50_control.py --ip <ROBOT_IP>
"""

import argparse

from telekinesis.synapse.tools.parallel_grippers.schunk import SchunkEGU50




def main(ip: str) -> None:
    """Run the demo for the Schunk EGU50 gripper wrapper."""

    gripper = SchunkEGU50()
    try:
        gripper.connect(ip)
        try:
            print(f"Current width: {gripper.get_current_position()} mm")
            print(f"open()  -> {gripper.open()}")
            print(f"close() -> {gripper.close()}")
            print(f"move(30 mm) -> {gripper.move(30.0)}")
        finally:
            gripper.disconnect()
    except NotImplementedError as exc:
        print(f"[stub] Schunk EGU hardware interface is pending: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for the Schunk EGU50 gripper wrapper")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="Schunk EGU50 IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
