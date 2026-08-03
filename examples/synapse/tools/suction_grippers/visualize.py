"""
Example demonstrating how to use the Piab class to control a Piab vacuum gripper
via the Piab URCap XML-RPC service.

Run  python visualize.py
"""

from telekinesis.synapse.tools.suction_grippers import piab

def main():
    """Visualize a Piab PiCobot Electric gripper in Rerun."""

    gripper = piab.PiabPiCobotElectric()
    gripper.visualize_rerun()

if __name__ == "__main__":
    main()