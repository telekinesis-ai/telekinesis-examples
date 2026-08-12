"""
Example: connect to a URSim virtual controller and control a UR10e.

Prerequisites:
  - URSim running in Docker with ports 30001-30004 and 29999 exposed
  - Remote Control enabled in the URSim teach pendant
    (hamburger menu -> Settings -> System -> Remote Control -> Enable)

Run:
  python ur10e_virtual_controller.py
"""

from telekinesis.synapse.robots.manipulators import universal_robots

URSIM_IP = "127.0.0.1"


def main():
    """
    Connect to a URSim virtual controller and control a UR10e."""

    # Create a UR10e robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the URSim virtual controller
    robot.connect(URSIM_IP)

    # Get robot information
    print("Robot mode:   ", robot.get_robot_mode())
    print("Safety mode:  ", robot.get_safety_mode())
    print("Robot status: ", robot.get_robot_status())
    print("Joint positions (deg):", robot.get_joint_positions())
    print("TCP pose (m, deg):    ", robot.get_cartesian_pose())

    # Move command
    robot.set_joint_positions(robot.default_joint_configuration)

    # Disconnect from the URSim virtual controller
    robot.disconnect()


if __name__ == "__main__":
    main()
