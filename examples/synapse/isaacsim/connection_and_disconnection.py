"""
Connection and disconnection example for isaacsim from the Synapse SDK.

Connects to and disconnects from a UR10e running in NVIDIA Isaac Sim, addressed
by its simulation prim path (no hardware connection is made).

Currently not supported.

Usage:
    python connection_and_disconnection.py [--simulation-prim-path <PRIM_PATH>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(simulation_prim_path: str):
    """Connect to a simulated UR10e at `simulation_prim_path` and cleanly disconnect."""

    # Create the robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the simulated robot at the given prim path
    robot.connect(simulation_prim_path=simulation_prim_path)
    logger.success(f"Connected to UR10e at {simulation_prim_path}.")

    # Sleep for a bit
    time.sleep(2)

    # Disconnect from the robot
    robot.disconnect()
    logger.success("Disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connection Synapse example (Isaac Sim)")
    parser.add_argument("--simulation-prim-path", type=str, default="/World/UR10e",
                        help="Simulation prim path for the UR10e robot (default: /World/UR10e)")
    args = parser.parse_args()

    main(simulation_prim_path=args.simulation_prim_path)
