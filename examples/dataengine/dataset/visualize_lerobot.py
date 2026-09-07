"""Example script demonstrating how to visualize a LeRobot dataset using the Telekinesis Data Engine."""

from telekinesis.dataengine import datasets


def visualize_lerobot_dataset_example():
    """Load and visualize a LeRobot dataset."""

    # 1. Define the dataset identity and local dataset path.
    repo_id = "user/my_example_dataset"
    local_path = "C:\\Users\\amsve\\Documents\\Telekinesis\\Code\\data-engine\\data\\eval_ur10e_real"

    # 2. Load the LeRobot dataset.
    config = datasets.LeRobotDatasetConfig(
        root=local_path,
    )

    dataset = datasets.LeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

    # 3. Visualize the dataset using Rerun.
    dataset.visualize()


if __name__ == "__main__":
    visualize_lerobot_dataset_example()