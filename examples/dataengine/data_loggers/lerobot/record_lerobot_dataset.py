"""Example script demonstrating how to log a LeRobot dataset using the Telekinesis Data Engine."""

import numpy as np
from loguru import logger

from telekinesis.dataengine import datasets
from telekinesis.dataengine import data_loggers


def record_lerobot_dataset_example():
    """Example function to load a LeRobot dataset."""

    # 1. Define the path, configuration, and the dataset features.
    repo_id = "user/my_example_dataset"
    local_path = "data/logger_test_dataset"
    features = {
        "observation.camera_rgb": {
            "dtype": "video",
            "shape": [3, 64, 64],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [10],
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
        },
    }
    config = datasets.LeRobotDatasetWriterConfig(
        tolerance_s=1e-4,
    )

    # 2. Create the LeRobot dataset logger with the specified configuration.
    lerobot_logger = data_loggers.LeRobotDatasetLogger(
        repo_id=repo_id,
        local_path=local_path,
        mode="create",
        fps=30,
        features=features,
        robot_type="my_dummy_ur",
        config=config,
    )

    num_episodes = 5
    max_frames_per_episode = 5

    # 3. Loop over episodes and log frames for each episode.
    try:
        for episode_index in range(num_episodes):
            lerobot_logger.start_episode()

            frame_index = 0
            try:
                while not task_complete(frame_index, max_frames_per_episode):
                    frame = {
                        "observation.camera_rgb": np.random.rand(3, 64, 64).astype("float32"),
                        "observation.state": np.random.rand(10).astype("float32"),
                        "action": np.random.rand(7).astype("float32"),
                        "task": "Dummy pick-and-place task",
                    }
                    lerobot_logger.log(frame)
                    frame_index += 1

            except (Exception, KeyboardInterrupt):
                # A failed or interrupted episode is incomplete, so discard it
                # before propagating the original exception.
                lerobot_logger.discard_episode()
                logger.exception("Episode recording failed. Current episode discarded.")
                raise

            else:
                # Save only successfully completed episodes.
                lerobot_logger.stop_episode()
                logger.info(f"Episode {episode_index} saved.")

    except KeyboardInterrupt:
        logger.info("Stopping data collection.")

    finally:
        # If saving failed, stop_episode() deliberately leaves the episode
        # active so the caller can decide whether to retry or discard it.
        if lerobot_logger.episode_active:
            try:
                lerobot_logger.discard_episode()
            except Exception:
                logger.exception("Failed to discard the active episode during cleanup.")

        # Always finalize writers and pending dataset state after cleanup.
        lerobot_logger.close()


def task_complete(frame_index: int, max_frames: int) -> bool:
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
    return frame_index >= max_frames


if __name__ == "__main__":
    record_lerobot_dataset_example()
