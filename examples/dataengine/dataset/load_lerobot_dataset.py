"""Example script demonstrating how to load a LeRobot dataset using the Telekinesis Data Engine."""

from telekinesis.dataengine import datasets


def load_lerobot_dataset_example():
    """Example function to load a LeRobot dataset."""

    path_or_repo_id = "lerobot/aloha_sim_insertion_scripted"
    config = datasets.LeRobotDatasetConfig(
        root="data/testing",
        episode_indices=[0, 1, 2],
    )

    dataset = datasets.load_dataset(
        path_or_repo_id=path_or_repo_id,
        dataset_format="lerobot",
        config=config,
    )
    return dataset


if __name__ == "__main__":
    load_lerobot_dataset_example()
