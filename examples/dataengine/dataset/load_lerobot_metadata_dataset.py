from pathlib import Path

from loguru import logger
from telekinesis.dataengine.datasets.lerobot import LeRobotDatasetMetadata


def main():
    root = Path(__file__).resolve().parent.parent / "datasets" / "eval_ur10e_real"
    metadata = LeRobotDatasetMetadata(repo_id="eval_ur10e_real", root=root)

    logger.info("Metadata loaded successfully.")
    logger.info(metadata)
    logger.info("Root path: {}", root)

    # Log detailed metadata information
    logger.info("Info: {}", metadata.info)
    logger.info("Root: {}", metadata.url_root)
    logger.info("Codebase version: {}", metadata.version)
    logger.info("Video path: {}", metadata.video_path)
    logger.info("Data path: {}", metadata.data_path)
    logger.info("Robot type: {}", metadata.robot_type)

    # Features
    logger.info("Names: {}", list(metadata.names))
    logger.info("Features: {}", metadata.features)
    logger.info("Image keys: {}", metadata.image_keys)
    logger.info("Video keys: {}", metadata.video_keys)
    logger.info("Camera keys: {}", metadata.camera_keys)
    logger.info("Depth keys: {}", metadata.depth_keys)
    logger.info("FPS: {}", metadata.fps)
    logger.info("Has language columns: {}", metadata.has_language_columns)
    logger.info("Names: {}", metadata.names)
    logger.info("Shapes: {}", metadata.shapes)

    # Dataset statistics
    logger.info("Total episodes: {}", metadata.total_episodes)
    logger.info("Total frames: {}", metadata.total_frames)
    logger.info("Total tasks: {}", metadata.total_tasks)
    logger.info("Tasks: {}", metadata.tasks)
    logger.info("Stats: {}", metadata.stats.keys())
    logger.info("Episodes: {}", metadata.episodes)
    logger.info("Tools: {}", metadata.tools)

    logger.info("Max chunk size: {}", metadata.max_chunk_size)
    logger.info("Max data files size (MB): {}", metadata.max_data_files_size_in_mb)
    logger.info("Max video files size (MB): {}", metadata.max_video_files_size_in_mb)

    logger.info("Chunk settings: {}", metadata.get_chunk_settings())

    # Episode index
    ep_index = 0  # Example episode index
    task = "Grab the green cube and put it in the blue box"
    video_key = "observation.images.camera1"
    logger.info(f"Episode {ep_index} data path: {metadata.get_data_file_path(ep_index)}")
    logger.info(f"Episode {ep_index} task index: {metadata.get_task_index(task)}")
    logger.info(
        f"Episode {ep_index} video path: {metadata.get_video_file_path(ep_index, video_key)}"
    )


if __name__ == "__main__":
    main()
