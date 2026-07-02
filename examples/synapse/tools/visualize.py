"""Schunk EGP gripper visualization example for the Synapse SDK.

Loads a Schunk EGP gripper and visualizes it in Rerun (no hardware required).
Supported for all grippers

Run:
    python examples/synapse/tools/visualize.py
"""

from telekinesis.synapse.tools.parallel_grippers import schunk

def main():
    """Visualize a Schunk EGP gripper in Rerun."""

    gripper = schunk.SchunkEGP()
    gripper.visualize_rerun()


if __name__ == "__main__":
    main()
