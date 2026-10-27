"""Generate a "flying things"-type dataset."""

import argparse

import numpy as np

from telekinesis.illusion import config
from telekinesis.illusion.core.environment import Environment
from telekinesis.illusion.core.engine import BlenderEngine
from telekinesis.illusion.core.synthetic_data_generator import (
    SyntheticDataGenerator,
)
from telekinesis.illusion.randomizer.randomizer import Randomizer
from telekinesis.illusion.randomizer.randomizer_node import (
    BackgroundRandomizer,
    CameraPoseRandomizer,
    LightPoseRandomizer,
    LightRandomizer,
    MaterialRandomizer,
    ModelInstanceRandomizer,
    ModelPoseRandomizer,
)
from telekinesis.illusion.types.model import Model
from telekinesis.illusion.utils.assets import resolve_asset_dir
from telekinesis.illusion.writer.writer import CocoWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the generated dataset in the interactive viewer.",
    )
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--device", default="CUDA")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        import random

        random.seed(args.seed)
        np.random.seed(args.seed)

    # Create the environment
    environment = Environment()
    environment.add(
        config.CameraConfig(
            image_width=args.resolution, image_height=args.resolution
        )
    )

    assets_dir = resolve_asset_dir()

    environment.add(
        config.LightConfig(
            "key", "POINT", properties={"power": 40.0, "radius": 0.05}
        )
    )
    environment.add(config.LightConfig("sun", "SUN", properties={"power": 0.2}))

    # Add models to the environment
    model_1_path = str(
        assets_dir / "models" / "mechanical_parts" / "gearwheel_1.glb"
    )
    environment.add(
        config.ModelConfig.from_path(
            model_1_path,
            model_name="part_1",
            groups="mechanical_parts",
            uv_mapping="cube",
        )
    )

    model_2_path = str(
        assets_dir / "models" / "mechanical_parts" / "pipe_1.glb"
    )
    environment.add(
        config.ModelConfig.from_path(
            model_2_path,
            model_name="part_2",
            groups="mechanical_parts",
            uv_mapping="cube",
        )
    )

    # Add materials and HDRIs to the environment. The randomizers below only
    # select among the assets registered here.
    environment.add(config.MaterialConfig.discover(assets_dir / "materials"))
    environment.add(config.BackgroundConfig.discover(assets_dir / "hdris"))

    # Create the randomizer
    randomizer = Randomizer()

    # Add model instance randomizer
    model_instance_randomizer = ModelInstanceRandomizer(
        config.InstanceRandomizerConfig(
            counts={"part_1": (1, 3), "part_2": (1, 1)},
            min_num_total_models=2,
            max_num_total_models=4,
        )
    )

    randomizer.add_randomizer(
        randomizer_node=model_instance_randomizer,
        node_name="instance_randomizer_models",
    )

    # Add model pose randomizer
    def sample_pose(model: Model):
        """
        Randomly sample and apply a 6-DoF pose to a model.

        The model's location is sampled uniformly within an axis-aligned box
        centered around the origin.
        """
        model.set_location(
            np.random.uniform((-0.1, -0.1, 0.1), (0.1, 0.1, 0.1))
        )
        model.set_rotation(
            np.random.uniform((-180, -180, -180), (180, 180, 180))
        )

    model_pose_randomizer = ModelPoseRandomizer(
        config.ModelPoseRandomizerConfig(),
        pose_sampling_function=sample_pose,
        target_models=["part_1", "part_2"],
    )

    randomizer.add_randomizer(
        randomizer_node=model_pose_randomizer, node_name="pose_randomizer"
    )

    # Add matrial randomizer
    material_randomizer = MaterialRandomizer(
        config.MaterialRandomizerConfig(
            target_models=("part_1", "part_2"), groups=("metal",)
        )
    )

    randomizer.add_randomizer(
        randomizer_node=material_randomizer, node_name="material_randomizer"
    )

    # Add background randomizer
    background_randomizer = BackgroundRandomizer(
        config.BackgroundRandomizerConfig(
            groups=("indoor/industrial", "indoor/studio")
        )
    )

    randomizer.add_randomizer(
        randomizer_node=background_randomizer, node_name="background_randomizer"
    )

    # Randomize point-light power/radius and sun strength/angular diameter.
    randomizer.add_randomizer(
        LightRandomizer(
            config.LightRandomizerConfig(
                properties={
                    "color": {"min": (0.8, 0.85, 0.9), "max": (1.0, 1.0, 1.0)},
                    "power": {"min": 20.0, "max": 60.0},
                    "radius": {"min": 0.02, "max": 0.08},
                }
            ),
            ["key"],
        ),
        node_name="point_light_properties",
    )
    randomizer.add_randomizer(
        LightRandomizer(
            config.LightRandomizerConfig(
                properties={
                    "power": {"min": 0.1, "max": 0.4},
                    "angle": {"min": 0.01, "max": 0.1},
                }
            ),
            ["sun"],
        ),
        node_name="sun_properties",
    )
    # Sample overhead poses aimed at the parts, keeping the emitter clear.
    randomizer.add_randomizer(
        LightPoseRandomizer(
            config.LightPoseRandomizerConfig(
                sampler="shell_sampler",
                min_distance=0.3,
                params={
                    "radius_min": 1.0,
                    "radius_max": 1.5,
                    "elevation_min": 35.0,
                    "elevation_max": 75.0,
                },
            ),
            target_lights=["key", "sun"],
        ),
        node_name="light_poses",
    )

    # Add camera pose randomizer
    camera_pose_randomizer = CameraPoseRandomizer(
        config.CameraPoseRandomizerConfig(
            sampler="shell_sampler",
            number_of_views=2,
            params={"radius_min": 0.5, "radius_max": 0.7},
        )
    )

    randomizer.add_randomizer(
        randomizer_node=camera_pose_randomizer,
        node_name="camera_pose_randomizer",
    )

    # Create the writer
    writer = CocoWriter(
        output_dir=args.output_dir, include_camera_metadata=True
    )

    # Create the data generator with environment, randomizer and writer
    data_generator = SyntheticDataGenerator(
        environment=environment,
        randomizer=randomizer,
        writer=writer,
        engine=BlenderEngine(
            config.RenderConfig(device=args.device, samples=args.samples),
            config.PhysicsConfig(active=False),
        ),
    )

    # Generate data
    data_generator.generate(num_images=args.num_images)

    if args.preview:
        from telekinesis.illusion.viewer.shard_viewer import view_coco

        view_coco(writer.get_output_dir())


if __name__ == "__main__":
    main()
