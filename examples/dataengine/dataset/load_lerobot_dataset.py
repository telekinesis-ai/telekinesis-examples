"""Example script demonstrating how to load a LeRobot dataset using the Telekinesis Data Engine."""

from pathlib import Path

from loguru import logger

from telekinesis.dataengine import datasets

def load_lerobot_dataset_example():
    """Load a LeRobot dataset from the Hugging Face Hub."""

    # 1. Define the dataset identity and local storage path.
    repo_id = "lerobot/pusht"
    local_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "results"
        / repo_id
    )

    # 2. Configure how the dataset should be loaded.
    reader_config = datasets.LeRobotDatasetReaderConfig(
        episode_indices=[0, 1, 2],
    )

    # 3. Load the LeRobot dataset.
    dataset = datasets.LeRobotDataset(
        repo_id=repo_id,
        local_path=local_path,
        config=reader_config,
    )

    logger.info("LeRobot dataset loaded successfully.")
    logger.info(dataset)
    logger.info("Selected episodes: {}", dataset.episode_indices)
    logger.info("Number of selected episodes: {}", dataset.num_episodes)
    logger.info("Number of selected frames: {}", dataset.num_frames)

    # 4. Access the first frame from the selected episodes.
    first_frame = dataset[0]
    logger.info("First frame: {}", first_frame)


if __name__ == "__main__":
    load_lerobot_dataset_example()
