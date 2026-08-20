"""
Individual test for the JointTrajectoryController C++ binding.

Verifies three behaviors without a control loop: holding the current pose when
no trajectory is set, linear interpolation between waypoints as the internal
clock advances by dt, and holding the final waypoint after the trajectory ends.

Run:
    python joint_trajectory_controller.py

Expected output:
    no trajectory -> holds state: [7. 7. 7. 7. 7. 7.]
    supported_command_types: ['position']
    t=0.5s (halfway): [5. 5. 5. 5. 5. 5.]
    t=1.0s (end):     [10. 10. 10. 10. 10. 10.]
    t=1.5s (after):   [10. 10. 10. 10. 10. 10.]
    OK
"""
import numpy as np

from telekinesis.synapse import robot_state
from telekinesis.synapse.controllers import joint_trajectory_controller


def main():
    #===================== Create Controller ======================================
    controller = joint_trajectory_controller.JointTrajectoryController(hz=500)
    print("supported_command_types:", controller.supported_command_types)

    state = robot_state.RobotState()
    state.joint_positions = np.full(6, 7.0)

    # ==================== Run Skill ============================================
    # With no trajectory set, the controller holds the current joint positions.
    held = controller.compute_control_command(state, 0.01)
    print("no trajectory -> holds state:", held)
    assert np.allclose(held, np.full(6, 7.0))

    # Linear ramp from all-zeros to all-tens over one second.
    trajectory = [
        (np.zeros(6), 0.0),
        (np.full(6, 10.0), 1.0),
    ]
    controller.set_trajectory(trajectory)

    halfway = controller.compute_control_command(state, 0.5)
    print("t=0.5s (halfway):", halfway)
    assert np.allclose(halfway, np.full(6, 5.0))

    end = controller.compute_control_command(state, 0.5)
    print("t=1.0s (end):    ", end)
    assert np.allclose(end, np.full(6, 10.0))

    after = controller.compute_control_command(state, 0.5)
    print("t=1.5s (after):  ", after)
    assert np.allclose(after, np.full(6, 10.0))

    print("OK")


if __name__ == "__main__":
    main()
