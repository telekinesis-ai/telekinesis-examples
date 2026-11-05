"""Offline Dobot models from DOBOT_6Axis_ROS2_V4.

Run the adjacent prepare_dobot.py first. Pass description_dir or set
DOBOT_DESCRIPTION_DIR to the generated URDF directory when outside the workspace.
Hardware control is not implemented.
"""

import os
from pathlib import Path

from telekinesis.synapse.robots.manipulators.abstract_robot import AbstractManipulator


class Dobot(AbstractManipulator):
    """Shared six-axis Dobot kinematics and visualization support."""

    model: str = ""

    def __init__(self, name: str | None = None, description_dir: str | Path | None = None):
        directory = Path(description_dir or os.environ.get("DOBOT_DESCRIPTION_DIR", ".cache/dobot"))
        directory = directory.resolve()
        urdf = directory / f"{self.model}.urdf"
        if not urdf.is_file():
            raise FileNotFoundError(f"{urdf}: run prepare_dobot.py first.")
        super().__init__(name=name)
        self.default_tcp = "Link6"
        self.active_tcp = self.default_tcp
        self.simulation_mount_link = self.default_tcp
        self.set_default_joint_configuration([0] * 6)
        self.model_root_dir = directory
        srdf = directory / f"{self.model}.srdf"
        self._build_pinocchio_model(str(urdf), str(directory), str(srdf) if srdf.is_file() else None)
        self.setup_kinematics_solver("multi_start_clik")
        self._initialize_state_and_publishers()

    def _connect_real(self, ip: str) -> None:
        raise NotImplementedError("Dobot hardware control is not implemented.")

    def _disconnect_real(self) -> None:
        raise NotImplementedError("Dobot hardware control is not implemented.")

    def _push_active_tcp_to_controller_real(self) -> None:
        raise NotImplementedError("Dobot hardware control is not implemented.")

    def _set_joint_positions_real(self, joint_positions, speed, acceleration, asynchronous):
        raise NotImplementedError("Dobot hardware control is not implemented.")

    def _set_cartesian_pose_real(self, cartesian_pose, speed, acceleration, asynchronous):
        raise NotImplementedError("Dobot hardware control is not implemented.")


class DobotCR3(Dobot):
    """Dobot CR3."""
    model = "cr3"


class DobotCR3AF(Dobot):
    """Dobot CR3AF."""
    model = "cr3af"


class DobotCR5(Dobot):
    """Dobot CR5."""
    model = "cr5"


class DobotCR5AF(Dobot):
    """Dobot CR5AF."""
    model = "cr5af"


class DobotCR6AL(Dobot):
    """Dobot CR6AL."""
    model = "cr6al"


class DobotCR7(Dobot):
    """Dobot CR7."""
    model = "cr7"


class DobotCR10(Dobot):
    """Dobot CR10."""
    model = "cr10"


class DobotCR10AF(Dobot):
    """Dobot CR10AF."""
    model = "cr10af"


class DobotCR12(Dobot):
    """Dobot CR12."""
    model = "cr12"


class DobotCR16(Dobot):
    """Dobot CR16."""
    model = "cr16"


class DobotCR20(Dobot):
    """Dobot CR20."""
    model = "cr20"


class DobotCR20AF(Dobot):
    """Dobot CR20AF."""
    model = "cr20af"


class DobotCR30H(Dobot):
    """Dobot CR30H."""
    model = "cr30h"


class DobotME6(Dobot):
    """Dobot ME6."""
    model = "me6"


class DobotNova2(Dobot):
    """Dobot Nova 2."""
    model = "nova2"


class DobotNova2S(Dobot):
    """Dobot Nova 2S."""
    model = "nova2s"


class DobotNova5(Dobot):
    """Dobot Nova 5."""
    model = "nova5"


MODELS = {cls.model: cls for cls in (
    DobotCR3, DobotCR3AF, DobotCR5, DobotCR5AF, DobotCR6AL, DobotCR7,
    DobotCR10, DobotCR10AF, DobotCR12, DobotCR16, DobotCR20, DobotCR20AF,
    DobotCR30H, DobotME6, DobotNova2, DobotNova2S, DobotNova5,
)}
