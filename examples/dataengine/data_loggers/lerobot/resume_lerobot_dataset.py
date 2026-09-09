"""Example script demonstrating how to resume logging a LeRobot dataset using the Telekinesis Data Engine."""

import time
from pathlib import Path

import numpy as np
from loguru import logger

from telekinesis.dataengine import datasets, data_loggers


def resume_lerobot_dataset_example():
    """Example function to resume logging an existing LeRobot dataset."""

    #============= Step 1: Define the repo ID and existing local path ================
    repo_id = "user/my_record_example"
    local_path = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "results"
        / repo_id
    )

    #============= Step 2: Define the dataset writer configuration ================
    config = datasets.LeRobotDatasetWriterConfig(
        tolerance_s=1e-4,
    )
    # Use the same FPS as the existing dataset being resumed.
    dataset_fps = 30

    #============= Step 3: Resume the LeRobot dataset logger with configuration ================
    # Resume mode loads the existing dataset schema and continues its episode indices.
    lerobot_logger = data_loggers.LeRobotDatasetLogger(
        repo_id=repo_id,
        local_path=local_path,
        mode="resume",
        config=config,
    )

    #============= Step 4: Define the number of additional episodes as required ================
    num_episodes = 5

    #============= Step 5: Loop over and log additional episodes ================
    try:
        for episode_index in range(num_episodes):

            # Start a new episode for a new task
            lerobot_logger.start_episode()

            frame_index = 0
            try:
                # Termination criterion or task completeness
                while not task_complete(frame_index):

                    frame_start = time.perf_counter()

                    # Get the task name
                    task = "Pick up the blue cube"

                    # Make the frame using the existing dataset schema
                    frame = make_frame(task=task)

                    # Log the frame
                    lerobot_logger.log(frame)
                    frame_index += 1

                    # Wait to log at the required FPS.
                    wait_for_next_frame(frame_start, dataset_fps)

            except (Exception, KeyboardInterrupt):
                # A failed or interrupted episode is incomplete, so discard it
                # before propagating the original exception.
                lerobot_logger.discard_episode()
                logger.exception("Episode recording failed. Current episode discarded.")
                # Stop the episode and raise the issue as needed
                raise

            else:
                # Save only successfully completed episodes.
                lerobot_logger.stop_episode()
                logger.info(f"Additional episode {episode_index} saved.")

    except KeyboardInterrupt:
        logger.info("Stopping data collection.")

    finally:
        # If saving failed, stop_episode() deliberately leaves the episode
        # active so the caller can decide whether to retry or discard it.
        logger.info("Cleaning up active episode if any.")
        if lerobot_logger.episode_active:
            try:
                lerobot_logger.discard_episode()
            except Exception:
                logger.exception("Failed to discard the active episode during cleanup.")

        # Always finalize writers and pending dataset state after cleanup.
        lerobot_logger.close()
        logger.info("Logging complete.")
        

def read_observation_camera() -> np.ndarray:
    """Return the latest RGB camera observation.

    Replace this with your camera or vision pipeline, for example:
    - a robot-mounted camera stream,
    - an external RGB camera,
    - or a simulator image observation.
    """
    return np.random.rand(3, 480, 640).astype("float32")


def read_observation_robot_state() -> np.ndarray:
    """Return the current robot state.

    Replace this with your robot state interface, for example:
    - joint positions / velocities,
    - end-effector pose,
    - gripper state,
    - or simulator state observations.
    """
    observation_state = np.random.rand(7).astype("float32")
    return observation_state


def get_robot_action() -> np.ndarray:
    """Return the current action applied to the robot.

    Replace this with your control source, for example:
    - teleoperation input,
    - a joystick or keyboard controller,
    - a policy output,
    - or commands from a motion planner.
    """
    # In this example, consider a manipulator with 6 DoF plus 1 gripper position.
    robot_action = np.random.rand(7).astype("float32")
    return robot_action


def make_frame(task: str) -> dict:
    """Assemble a single dataset frame from robot observations and action.

    The frame must match the schema of the existing dataset being resumed.
    Replace the helper functions with your own robot, sensor, and controller
    interfaces.
    """
    return {
        "observation.camera_rgb": read_observation_camera(),
        "observation.state": read_observation_robot_state(),
        "action": get_robot_action(),
        "task": task,
    }


def task_complete(frame_index: int) -> bool:
    """Return whether the current task is complete.

    In a real robot application, replace this with the actual task termination
    condition, for example:

    - the robot reaches a target pose,
    - a successful grasp is detected,
    - the task success signal becomes true,
    - an operator confirms completion,
    - or a maximum episode duration is reached.

    This example simply terminates after `max_frames` frames.
    """
    max_frames = 5
    return frame_index >= max_frames


def wait_for_next_frame(start_time: float, fps: int) -> None:
    """Wait until the next dataset frame should be recorded."""
    frame_period = 1.0 / fps
    elapsed = time.perf_counter() - start_time
    remaining = frame_period - elapsed

    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    resume_lerobot_dataset_example()
