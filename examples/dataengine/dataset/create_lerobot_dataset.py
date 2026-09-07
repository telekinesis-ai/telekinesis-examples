"""Example script demonstrating how to create a LeRobot dataset using the Telekinesis Data Engine."""

from pathlib import Path
import shutil

from loguru import logger

from telekinesis.dataengine import datasets


def create_lerobot_dataset_example():
    """Create a new writable LeRobot dataset."""

    # 1. Define the dataset identity, local storage path, and features.
    repo_id = "user/my_create_example"
    local_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "results"
        / repo_id
    )
    # Remove any previous example dataset so this script can be rerun.
    if local_path.exists():
        shutil.rmtree(local_path)

    features = {
        "observation.images.camera1": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.depths.camera1": {
            "dtype": "depth",
            "shape": [480, 640],
            "names": ["height", "width"],
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
        "language_persistent": {
            "dtype": "text",
            "shape": [],
            "names": None,
        },
    }

    # 2. Create the LeRobot dataset.
    dataset = datasets.LeRobotDataset.create(
        repo_id=repo_id,
        local_path=local_path,
        fps=30,
        features=features,
        robot_type="dummy_ur10e",
        use_videos=True,
    )

    logger.info("LeRobot dataset created successfully.")
    logger.info(dataset)
    logger.info(f"Local path: {local_path}")

    # 3. Finalize the dataset when no more data will be written.
    dataset.finalize()

    return dataset


if __name__ == "__main__":
    create_lerobot_dataset_example()