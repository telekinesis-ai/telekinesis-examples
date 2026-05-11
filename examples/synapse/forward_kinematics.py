"""
Forward kinematics example for the Synapse SDK.

This example demonstrates how to compute the forward kinematics for a manipulator
using the Synapse SDK. The example uses the Universal Robots UR10e purely for
illustration, but the forward kinematics method is defined on the abstract
manipulator and supports all robot brands.

Usage:
    python forward_kinematics_example.py
"""

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """
    Demonstrates forward kinematics computation for the Universal Robot UR10e
    """

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    q = [0, -90, 90, 0, 90, 0]
    tcp_pose = robot.forward_kinematics(q=q)
    print("TCP pose: ", tcp_pose)


if __name__ == "__main__":
    main()
