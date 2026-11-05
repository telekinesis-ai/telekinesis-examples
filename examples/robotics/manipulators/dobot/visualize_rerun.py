"""Display all 17 Dobot models using the installed Synapse native runtime.

conda run -n synapse python examples/robotics/manipulators/dobot/prepare_dobot.py
conda run -n synapse python examples/robotics/manipulators/dobot/visualize_rerun.py
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


def main():
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description-dir", type=Path, default=root / ".cache/dobot")
    parser.add_argument("--save", type=Path, default=root / ".cache/dobot/all_dobot.rrd")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # Prefer the installed module; older wheels can use these bundled wrappers
    # with their matching installed native runtime.
    module_name = "telekinesis.synapse.robots.manipulators.dobot"
    if importlib.util.find_spec(module_name) is not None:
        dobot = importlib.import_module(module_name)
    else:
        spec = importlib.util.spec_from_file_location(
            "dobot_models", Path(__file__).with_name("_models.py")
        )
        dobot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dobot)
    manifest = json.loads((args.description_dir / "manifest.json").read_text())
    if set(manifest["models"]) != set(dobot.MODELS):
        raise ValueError("Prepared descriptions do not match the supported Dobot models.")
    recording = rr.RecordingStream("synapse_dobot_gallery")
    args.save.parent.mkdir(parents=True, exist_ok=True)
    recording.save(str(args.save))
    recording.send_blueprint(rrb.Spatial3DView(origin="/"))
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for index, (model, cls) in enumerate(dobot.MODELS.items()):
        robot = cls(description_dir=args.description_dir)
        if robot.ndof != 6:
            raise ValueError(f"{model}: expected six degrees of freedom")
        meshes = robot.get_visual_meshes_data()
        if sum(mesh["vertices"] is not None for mesh in meshes.values()) != 7:
            raise ValueError(f"{model}: missing visual meshes")
        pose = np.eye(4)
        pose[:2, 3] = [(index % 5) * 3.5, (index // 5) * 3.5]
        # Synapse's single-robot viewer uses /robot for every instance. Give
        # each gallery robot a separate path so models cannot overwrite one another.
        transforms = robot.get_visual_mesh_transforms()
        recording.log(f"robots/{model}", rr.Transform3D(
            translation=pose[:3, 3], mat3x3=pose[:3, :3]
        ), static=True)
        for link, mesh in meshes.items():
            if mesh["vertices"] is None:
                continue
            path = f"robots/{model}/{link}"
            transform = transforms[link]
            if not np.isfinite(transform).all():
                raise ValueError(f"{model}/{link}: non-finite forward kinematics")
            recording.log(path, rr.Mesh3D(
                vertex_positions=mesh["vertices"], triangle_indices=mesh["triangles"],
                vertex_normals=mesh["vertex_normals"], vertex_colors=mesh["vertex_colors"],
                albedo_factor=mesh["color"] if mesh["vertex_colors"] is None else None,
            ), rr.Transform3D(translation=transform[:3, 3], mat3x3=transform[:3, :3]),
                static=True)
        recording.log(f"labels/{model}", rr.Points3D(
            [[pose[0, 3], pose[1, 3], -0.15]], labels=[model.upper()], radii=0.01
        ), static=True)
        print(f"Loaded {model}: {robot.ndof} joints, seven visual meshes", flush=True)
    recording.flush()
    print(f"Saved all {len(dobot.MODELS)} models to {args.save}", flush=True)
    if not args.headless:
        import subprocess
        import sys
        subprocess.Popen([sys.executable, "-m", "rerun", str(args.save)])


if __name__ == "__main__":
    main()
