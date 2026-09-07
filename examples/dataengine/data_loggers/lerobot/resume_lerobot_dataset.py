"""Example script demonstrating how to resume recording a LeRobot dataset."""

import numpy as np
from loguru import logger

from telekinesis.dataengine import data_loggers, datasets


def resume_lerobot_dataset_example():
    """Resume an existing LeRobot dataset and record additional episodes."""

    # 1. Define the existing dataset location and writer configuration.
    path_or_repo_id = "user/my_example_dataset"
    local_path = "data/logger_test_dataset"

    config = datasets.LeRobotDatasetWriterConfig(
        tolerance_s=1e-4,
    )

    # 2. Resume the existing dataset.
    lerobot_logger = data_loggers.LeRobotDatasetLogger(
        repo_id=path_or_repo_id,
        local_path=local_path,
        mode="resume",
        config=config,
    )

    num_episodes = 5
    max_frames_per_episode = 5

    # 3. Record additional episodes.
    try:
        for episode_index in range(num_episodes):
            lerobot_logger.start_episode()

            frame_index = 0
            try:
                while not task_complete(frame_index, max_frames_per_episode):
                    frame = {
                        "observation.camera_rgb": np.random.rand(
                            3, 64, 64
                        ).astype("float32"),
                        "observation.state": np.random.rand(10).astype("float32"),
                        "action": np.random.rand(7).astype("float32"),
                        "task": "Dummy pick-and-place task",
                    }

                    lerobot_logger.log(frame)
                    frame_index += 1

            except (Exception, KeyboardInterrupt):
                lerobot_logger.discard_episode()
                logger.exception(
                    "Episode recording failed. Current episode discarded."
                )
                raise

            else:
                lerobot_logger.stop_episode()
                logger.info(f"Additional episode {episode_index} saved.")

    except KeyboardInterrupt:
        logger.info("Stopping data collection.")

    finally:
        if lerobot_logger.episode_active:
            try:
                lerobot_logger.discard_episode()
            except Exception:
                logger.exception(
                    "Failed to discard the active episode during cleanup."
                )

        lerobot_logger.close()


def task_complete(frame_index: int, max_frames: int) -> bool:
    """Return whether the current task is complete.

    In a real robot application, replace this with the actual task termination
    condition, such as task success, operator confirmation, or a maximum
    episode duration.

    This example terminates after `max_frames` frames.
    """
    return frame_index >= max_frames


if __name__ == "__main__":
    resume_lerobot_dataset_example()