"""Example: load every supported parallel gripper, across every brand, and
visualize each in Rerun.

Covers Robotiq, OnRobot, and Schunk. Each gripper is created offline (no
hardware/simulator needed), fetching its URDF via URDF.from_url() (see
AbstractParallelGripper._build_model in abstract_gripper.py). It's then shown
in its own Rerun viewer window, one at a time, before moving on to the next.

If https://assets.telekinesis.ai is unreachable, serve a local
telekinesis-assets checkout instead:

    1. cd /path/to/telekinesis-assets
    2. python3 -m http.server 8931
    3. abstract_gripper.py has a _PARALLEL_GRIPPER_ASSETS_BASE_URL constant
       near the top; it already defaults to "http://localhost:8931".

Run:
    python examples/robotics/parallel_grippers/advanced/visualize_all_parallel_grippers.py
"""

import inspect
import time

import rerun as rr
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import (
    onrobot,
    robotiq,
    schunk,
)

# All parallel-gripper brand modules paired with their brand base class.
_BRAND_MODULES = [
    (robotiq, robotiq.Robotiq),
    (onrobot, onrobot.OnRobot),
    (schunk, schunk.SchunkEGU),
]


# ============================ Get All Gripper Classes ===================================
def _all_gripper_classes() -> list[type]:
    """Return every concrete parallel-gripper model class across all brands.

    The brand base classes (e.g. ``Robotiq``, ``OnRobot``) are excluded — only
    the concrete models that ship a URDF bundle (e.g. ``Robotiq2F85``,
    ``OnRobotRG6``) are returned.
    """
    classes = []
    for module, base_cls in _BRAND_MODULES:
        classes.extend(
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj) and issubclass(obj, base_cls) and obj is not base_cls
        )
    return classes


def main():
    """
    Load every supported parallel gripper, across every brand, and visualize
    each in Rerun.
    """

    # ================================== Create Gripper =============================================
    gripper_classes = _all_gripper_classes()
    logger.info(f"Found {len(gripper_classes)} grippers across {len(_BRAND_MODULES)} brands.")

    for gripper_cls in gripper_classes:
        logger.info(f"Loading {gripper_cls.__name__}...")
        try:
            gripper = gripper_cls()
        except Exception as e:
            cause = e.__cause__ if e.__cause__ is not None else e
            logger.error(
                f"Skipping {gripper_cls.__name__}: failed to load ({e}) caused by: {cause!r}"
            )
            continue

        # A gripper whose URDF bundle could not be fetched has no model, so
        # there is nothing to visualize — skip it rather than raise.
        if gripper.urdf_path is None:
            logger.error(f"Skipping {gripper_cls.__name__}: no URDF model available.")
            continue

    # ================================== Visualize ================================================
        # Each gripper gets its own spawned viewer window. The first
        # visualize_rerun call uploads the static meshes; there are no further
        # updates here since the gripper is not moving.
        rr.init(f"telekinesis_synapse_{gripper_cls.__name__}", spawn=True)
        gripper.visualize_rerun(recording_stream=rr.get_global_data_recording())
        time.sleep(3.0)


if __name__ == "__main__":
    main()
