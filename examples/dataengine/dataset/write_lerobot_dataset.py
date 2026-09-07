"""Example script demonstrating how to write a LeRobot dataset using the Telekinesis Data Engine."""

from pathlib import Path
import shutil

import numpy as np
from loguru import logger

from telekinesis.dataengine import datasets


def write_lerobot_dataset_example():
    """Programmatically create and write episodes to a LeRobot dataset."""

    # 1. Define the dataset identity, local storage path, and features.
    repo_id = "user/my_example_dataset"
    local_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "results"
        / repo_id
    )
    # Remove the existing dataset directory to ensure rerun of example
    if local_path.exists():
        shutil.rmtree(local_path)

    features = {
        "observation.images.camera1": {
            "dtype": "video",
            "shape": [64, 64, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [6],
            "names": [
                "shoulder_pan_joint.pos",
                "shoulder_lift_joint.pos",
                "elbow_joint.pos",
                "wrist_1_joint.pos",
                "wrist_2_joint.pos",
                "wrist_3_joint.pos",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": [6],
            "names": [
                "shoulder_pan_joint.pos",
                "shoulder_lift_joint.pos",
                "elbow_joint.pos",
                "wrist_1_joint.pos",
                "wrist_2_joint.pos",
                "wrist_3_joint.pos",
            ],
        },
    }

    # 2. Create a writable LeRobot dataset.
    dataset = datasets.LeRobotDataset.create(
        repo_id=repo_id,
        local_path=local_path,
        fps=30,
        features=features,
        robot_type="ur10e",
        use_videos=True,
    )

    num_episodes = 3
    frames_per_episode = 5

    try:
        # 3. Programmatically add frames and save each episode.
        for episode_index in range(num_episodes):
            for _ in range(frames_per_episode):
                frame = {
                    "observation.images.camera1": np.random.randint(
                        0,
                        256,
                        size=(64, 64, 3),
                        dtype=np.uint8,
                    ),
                    "observation.state": np.random.rand(6).astype(np.float32),
                    "action": np.random.rand(6).astype(np.float32),
                    "task": "Dummy pick-and-place task",
                }

                dataset.add_frame(frame)

            dataset.save_episode()

            logger.info(f"Episode {episode_index+1} saved.")

    finally:
        # 4. Finalize all pending writers and metadata.
        dataset.finalize()

    logger.info("LeRobot dataset written successfully.")
    logger.info(dataset)
    logger.info(f"Local path: {local_path}")


if __name__ == "__main__":
    write_lerobot_dataset_example()