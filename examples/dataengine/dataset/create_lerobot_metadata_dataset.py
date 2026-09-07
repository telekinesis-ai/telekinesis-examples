from pathlib import Path

from loguru import logger
from telekinesis.dataengine.datasets.lerobot import LeRobotDatasetMetadata


def main():

    repo_root = Path(__file__).resolve().parent.parent / "results"

    dataset_root = repo_root / "eval_ur10e_real"
    fps = 30  # Example FPS value
    repo_id = "eval_ur10e_real"
    features = {
        "observation.images.camera1": {"dtype": "video", "shape": [3, 480, 640]},
        "observation.depths.camera1": {"dtype": "depth", "shape": [480, 640]},
        "language_persistent": {"dtype": "text", "shape": []},
    }
    metadata = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=fps,
        root=dataset_root,
        features=features,
        robot_type="ur10e",
        use_videos=True,
        metadata_buffer_size=None,
        max_chunk_size=None,
        max_data_files_size_in_mb=None,
        max_video_files_size_in_mb=None,
    )

    logger.info("Metadata loaded successfully.")
    logger.info(metadata)


if __name__ == "__main__":
    main()
