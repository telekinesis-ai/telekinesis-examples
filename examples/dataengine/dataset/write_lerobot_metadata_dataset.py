"""Create and write episode metadata for a small LeRobot dataset."""

from pathlib import Path

from loguru import logger
import numpy as np

from telekinesis.dataengine.datasets.lerobot import LeRobotDatasetMetadata
from telekinesis.dataengine.datasets.lerobot.compute_stats import compute_episode_stats


NUM_EPISODES = 5
FRAMES_PER_EPISODE = 10
TASK = "Move the robot to the target pose"

def main() -> None:
    """Create metadata, write five episode records, and verify the result."""
    root = Path(__file__).resolve().parent.parent / "datasets" / "eval_test"
    repo_id = "eval_ur10e_real"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"joint_{index}" for index in range(6)],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"joint_{index}" for index in range(6)],
        },
    }

    metadata = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=30,
        root=root,
        features=features,
        robot_type="ur10e",
        use_videos=False,
        metadata_buffer_size=NUM_EPISODES,
    )

    metadata.save_episode_tasks([TASK])
    random = np.random.default_rng(seed=0)

    for episode_index in range(NUM_EPISODES):
        episode_data = {
            "observation.state": random.standard_normal((FRAMES_PER_EPISODE, 6), dtype=np.float32),
            "action": random.standard_normal((FRAMES_PER_EPISODE, 6), dtype=np.float32),
        }
        episode_stats = compute_episode_stats(episode_data, metadata.features)

        metadata.save_episode(
            episode_index=episode_index,
            episode_length=FRAMES_PER_EPISODE,
            episode_tasks=[TASK],
            episode_stats=episode_stats,
            episode_metadata={},
        )
        logger.info("Saved metadata for episode {}.", episode_index)

    # Flush any buffered episode records and close the Parquet writer.
    metadata.finalize()

    saved_metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    logger.info("Metadata written to {}.", root)
    logger.info(
        "Saved {} episodes and {} frames.",
        saved_metadata.total_episodes,
        saved_metadata.total_frames,
    )
    logger.info("Tasks: {}", saved_metadata.tasks)
    logger.info("Statistics: {}", saved_metadata.stats)
    logger.info("Episodes: {}", saved_metadata.episodes)


if __name__ == "__main__":
    main()
