"""Example script demonstrating how to log a LeRobot dataset using the Telekinesis Data Engine."""

import shutil
import time
from pathlib import Path

import numpy as np
from loguru import logger

from telekinesis.dataengine import datasets, data_loggers

def record_lerobot_dataset_example():
    """Example function to record a LeRobot dataset."""

    #======================= Step 1: Define Dataset =======================
    # 1. Define Repo ID and Local path
    repo_id = "user/my_record_example"
    local_path = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "results"
        / repo_id
    )
    if local_path.exists():
        shutil.rmtree(local_path)

    # 2. Define feature schema
    features = {
        "observation.camera_rgb": {
            "dtype": "video",
            "shape": [3, 480, 640],
            "names": [
                "channel",
                "height",
                "width",
            ],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow",
                "wrist_1",
                "wrist_2",
                "wrist_3",
                "gripper",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow",
                "wrist_1",
                "wrist_2",
                "wrist_3",
                "gripper",
            ],
        },
    }

    # 3. Define the dataset writer config
    config = datasets.LeRobotDatasetWriterConfig(
        tolerance_s=1e-4,
    )
    # Choose this based on your acquisition speed
    dataset_fps = 30

    # ===================== Step 2: Create the Logger ======================
    lerobot_logger = data_loggers.LeRobotDatasetLogger(
        repo_id=repo_id,
        local_path=local_path,
        mode="create",
        fps=dataset_fps,
        features=features,
        robot_type="my_dummy_ur",
        config=config,
    )

    #======================= Step 3: Record Episodes ========================
    num_episodes = 5
    try:
        for episode_index in range(num_episodes):

            # Start a new episode for a new task
            lerobot_logger.start_episode()

            frame_index = 0
            try:
                # Termination criterion or Task completeness
                while not task_complete(frame_index):

                    frame_start = time.perf_counter()

                    # Get the task name
                    task = "Pick up the blue cube"

                    # Placeholder to make the frame
                    frame = make_frame(task=task)

                    # Log the frame
                    lerobot_logger.log(frame)
                    frame_index += 1

                    # Wait to log at the required fps
                    wait_for_next_frame(frame_start, dataset_fps)

            except (Exception, KeyboardInterrupt):
                # A failed or interrupted episode is incomplete, so discard it
                # before propagating the original exception.
                lerobot_logger.discard_episode()
                logger.exception("Episode recording failed. Current episode discarded.")
                # Stop the episode and raise issue as needed
                raise

            else:
                # Save only successfully completed episodes.
                lerobot_logger.stop_episode()
                logger.info(f"Episode {episode_index} saved.")

    except KeyboardInterrupt:
        logger.info("Stopping data collection.")

    # ======================= Step 4: Finalize recording ===========================
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
    # In this examples, considering a manipulator with 6 dof plus 1 gripper position
    robot_action = np.random.rand(7).astype("float32")
    return robot_action


def make_frame(task: str) -> dict:
    """Assemble a single dataset frame from robot observations and action.

    This helper shows the structure expected by the dataset logger.
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
    max_frames= 5
    return frame_index >= max_frames

def wait_for_next_frame(start_time: float, fps: int) -> None:
    """Wait until the next dataset frame should be recorded."""
    frame_period = 1.0 / fps
    elapsed = time.perf_counter() - start_time
    remaining = frame_period - elapsed

    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    record_lerobot_dataset_example()
