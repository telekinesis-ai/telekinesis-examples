"""Convert the official Dobot xacros without requiring a ROS installation.

Run in the synapse environment after installing xacro. Meshes stay in the
upstream checkout; rerun this script if that checkout moves.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import xacro


def prepare(repository: Path, output: Path) -> list[str]:
    repository = repository.resolve()
    output = output.resolve()
    sources = sorted((repository / "cra_description/urdf").glob("*_robot.xacro"))
    if not sources:
        raise FileNotFoundError(f"No Dobot descriptions in {repository}")
    output.mkdir(parents=True, exist_ok=True)
    models = []
    for source in sources:
        model = source.name.removesuffix("_robot.xacro")

        def resolve_package(match):
            package = repository / match.group(1)
            if not (package / "package.xml").is_file():
                raise FileNotFoundError(f"ROS package not found: {package}")
            return package.as_posix()

        # Resolve ROS package substitutions locally, then let xacro expand XML.
        text = re.sub(r"\$\(find\s+([^\)]+)\)", resolve_package, source.read_text(encoding="utf-8"))
        document = xacro.parse(text)
        xacro.process_doc(document)
        root = ET.fromstring(document.toxml())
        # Upstream Nova2S uses alpha=50, outside URDF's valid [0, 1] range.
        # Make that material opaque, preserving its RGB values.
        for color in root.iter("color"):
            rgba = [float(value) for value in color.attrib["rgba"].split()]
            if len(rgba) != 4 or any(not 0 <= value <= 1 for value in rgba[:3]):
                raise ValueError(f"{model}: invalid material color {rgba}")
            rgba[3] = min(1.0, max(0.0, rgba[3]))
            color.set("rgba", " ".join(str(value) for value in rgba))
        for tag in ("ros2_control", "gazebo", "transmission"):
            for element in root.findall(tag):
                root.remove(element)
        for mesh in root.iter("mesh"):
            filename = mesh.attrib["filename"].removeprefix("file://")
            if filename.startswith("package://"):
                filename = str(repository / filename.removeprefix("package://"))
            path = Path(filename)
            if not path.is_file():
                raise FileNotFoundError(path)
            mesh.set("filename", path.resolve().as_posix())
        joints = [j for j in root.findall("joint") if j.get("type") != "fixed"]
        if len(joints) != 6:
            raise ValueError(f"{model}: expected six joints, got {len(joints)}")
        ET.indent(root)
        ET.ElementTree(root).write(output / f"{model}.urdf", encoding="utf-8", xml_declaration=True)
        srdf = repository / f"{model}_moveit/config/{model}_robot.srdf"
        if srdf.is_file():
            shutil.copy2(srdf, output / f"{model}.srdf")
        models.append(model)
        print(f"Converted {model}: six joints, {len(list(root.iter('mesh')))} meshes")
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    (output / "manifest.json").write_text(json.dumps({
        "source": "https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4",
        "commit": commit, "repository": str(repository), "models": models,
    }, indent=2) + "\n")
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(".cache/DOBOT_6Axis_ROS2_V4"))
    parser.add_argument("--output", type=Path, default=Path(".cache/dobot"))
    args = parser.parse_args()
    prepare(args.repository, args.output)
