"""Example: load every supported suction gripper, across every brand, and
visualize each in Rerun.

Run:
    python visualize_all_suction_grippers.py
"""

import inspect
import time

import rerun as rr
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import isaacsim, piab

# All suction-gripper brand modules paired with their brand base class.
_BRAND_MODULES = [
    (piab, piab.Piab)
]

_EXTRA_GRIPPER_CLASSES = []

# ============================ Get All Gripper Classes ===================================


def _all_gripper_classes() -> list[type]:
    """Return every concrete suction-gripper model class across all brands.

    The brand base classes (e.g. ``Piab``) are excluded — only the concrete
    models that ship a URDF bundle (e.g. ``PiabPiCobotElectric``) are
    returned, plus any classes listed in ``_EXTRA_GRIPPER_CLASSES``.
    """
    classes = []
    for module, base_cls in _BRAND_MODULES:
        classes.extend(
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj) and issubclass(obj, base_cls) and obj is not base_cls
        )

    classes.extend(_EXTRA_GRIPPER_CLASSES)
    return classes


def main():
    """
    Load every supported suction gripper, across every brand, and visualize
    each in Rerun.
    """

    # ================================== Create Gripper ======================
    gripper_classes = _all_gripper_classes()
    logger.info(
        f"Found {len(gripper_classes)} grippers across {len(_BRAND_MODULES) + len(_EXTRA_GRIPPER_CLASSES)} brands.")

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

    # ================================== Visualize ================================================
        rr.init(f"telekinesis_synapse_{gripper_cls.__name__}", spawn=True)
        gripper.visualize_rerun(recording_stream=rr.get_global_data_recording())
        time.sleep(3.0)


if __name__ == "__main__":
    main()
