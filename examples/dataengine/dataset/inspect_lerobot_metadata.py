"""Example script demonstrating how to inspect LeRobot dataset metadata."""

from pathlib import Path

from loguru import logger

from telekinesis.dataengine import datasets

def inspect_lerobot_metadata_example():
    """Inspect metadata of an existing LeRobot dataset."""

    # 1. Define the dataset identity and local path.
    repo_id = "lerobot/aloha_sim_insertion_scripted"
    local_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "results"
        / repo_id
    )

    # 2. Load the dataset metadata.
    metadata = datasets.LeRobotDatasetMetadata(
        repo_id=repo_id,
        local_path=local_path,
    )

    logger.info("Metadata loaded successfully.")
    logger.info(metadata)

    # 3. Inspect general dataset information.
    logger.info(f"Local path: {local_path}")
    logger.info(f"URL root: {metadata.url_root}")
    logger.info(f"Codebase version: {metadata.version}")
    logger.info(f"Robot type: {metadata.robot_type}")
    logger.info(f"FPS: {metadata.fps}")

    logger.info(f"Data path: {metadata.data_path}")
    logger.info(f"Video path: {metadata.video_path}")

    # 4. Inspect the dataset schema and features.
    logger.info(f"Features: {metadata.features}")
    logger.info(f"Names: {metadata.names}")
    logger.info(f"Shapes: {metadata.shapes}")

    logger.info(f"Image keys: {metadata.image_keys}")
    logger.info(f"Video keys: {metadata.video_keys}")
    logger.info(f"Depth keys: {metadata.depth_keys}")
    logger.info(f"Camera keys: {metadata.camera_keys}")
    logger.info(f"Has language columns: {metadata.has_language_columns}")

    # 5. Inspect dataset statistics and episode information.
    logger.info(f"Total episodes: {metadata.total_episodes}")
    logger.info(f"Total frames: {metadata.total_frames}")
    logger.info(f"Total tasks: {metadata.total_tasks}")

    logger.info(f"Tasks: {metadata.tasks}")
    logger.info(f"Episodes: {metadata.episodes}")
    logger.info(f"Statistics: {metadata.stats}")
    logger.info(f"Tools: {metadata.tools}")

    # 6. Inspect dataset storage and chunking configuration.
    logger.info(f"Max chunk size: {metadata.max_chunk_size}")
    logger.info(
        f"Max data file size (MB): {metadata.max_data_files_size_in_mb}",
    )
    logger.info(f"Max video file size (MB): {metadata.max_video_files_size_in_mb}")
    logger.info(f"Chunk settings: {metadata.get_chunk_settings()}")

    # 7. Resolve metadata for a specific episode.
    episode_index = 0
    # Use the key as per the lerobot features downloaded
    video_key = "observation.images.top"

    logger.info(
        f"Episode {episode_index} data path: {metadata.get_data_file_path(episode_index)}"
    )
    logger.info(
        f"Episode {episode_index} video path for '{video_key}': {metadata.get_video_file_path(episode_index, video_key)}"
    )


if __name__ == "__main__":
    inspect_lerobot_metadata_example()