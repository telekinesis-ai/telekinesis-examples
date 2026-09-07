"""Example script demonstrating how to visualize a LeRobot dataset using the Telekinesis Data Engine."""

from pathlib import Path

from telekinesis.dataengine import datasets

def visualize_lerobot_dataset_example():
    """Load and visualize a LeRobot dataset."""

    # 1. Define the dataset identity and local dataset path.
    repo_id = "lerobot/aloha_sim_insertion_scripted"
    local_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "results"
        / repo_id
    )
    # 2. Load the LeRobot dataset from the local path.
    dataset = datasets.LeRobotDataset(
        repo_id=repo_id,
        local_path=local_path,
    )

    # 3. Visualize the dataset using Rerun.
    dataset.visualize()


if __name__ == "__main__":
    visualize_lerobot_dataset_example()