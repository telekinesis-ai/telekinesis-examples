"""Example: load every supported manipulator, across every brand, and
visualize each in Rerun.

Covers Universal Robots, ABB, Kuka, Motoman, Fanuc, Neura Robotics, Franka
Robotics, and Epson. Each robot is created offline (no hardware/simulator
needed), fetching its URDF via URDF.from_url() (see the corresponding
manufacturer module, e.g. universal_robots.py). It's then shown in its own
Rerun viewer window, one at a time, before being released and moving on to
the next.

If https://assets.telekinesis.ai is unreachable, serve a local
telekinesis-assets checkout instead:

    1. cd /path/to/telekinesis-assets
    2. python3 -m http.server 8931
    3. Each manufacturer module has its own _<BRAND>_ASSETS_BASE_URL
       constant near the top; they already default to "http://localhost:8931".

Run:
    python examples/visualize_all_robots.py
"""

import inspect
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import (
    abb,
    epson,
    fanuc,
    franka_robotics,
    kuka,
    motoman,
    neura_robotics,
    universal_robots,
)

# All manufacturer modules
_MANUFACTURER_MODULES = [
    (abb, abb.ABB),
    (kuka, kuka.Kuka),
    (motoman, motoman.Motoman),
    (fanuc, fanuc.Fanuc),
    (neura_robotics, neura_robotics.NeuraRobotics),
    (franka_robotics, franka_robotics.FrankaRobotics),
    (epson, epson.Epson),
    (universal_robots, universal_robots.UniversalRobots),
]

# ============================ Get All Robot Classes =====================================
def _all_robot_classes() -> list[type]:
    """Return every concrete manipulator class across all brands."""
    classes = []
    for module, base_cls in _MANUFACTURER_MODULES:
        classes.extend(
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj) and issubclass(obj, base_cls) and obj is not base_cls
        )
    return classes


def main():
    """
    Load every supported manipulator, across every brand, and visualize each in Rerun.
    """

    # ================================== Create Robot ================================================
    robot_classes = _all_robot_classes()
    logger.info(f"Found {len(robot_classes)} robots across {len(_MANUFACTURER_MODULES)} brands.")

    for robot_cls in robot_classes:
        logger.info(f"Loading {robot_cls.__name__}...")
        try:
            robot = robot_cls(name=robot_cls.__name__.lower())
        except Exception as e:
            cause = e.__cause__ if e.__cause__ is not None else e
            logger.error(f"Skipping {robot_cls.__name__}: failed to load ({e}) caused by: {cause!r}")
            continue

    
    # ================================== Visualize ================================================
        try:
            robot.visualize_rerun(live=False)
            time.sleep(3.0)

    # ================================== Shutdown ================================================
        finally:
            robot.shutdown()


if __name__ == "__main__":
    main()
