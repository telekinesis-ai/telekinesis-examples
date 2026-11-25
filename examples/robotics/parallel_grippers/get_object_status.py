"""
Reads the object-detection status of a parallel gripper, and shows the gripper in Rerun.

Supports OnRobot and Robotiq grippers, and Isaac Sim.

Usage:
    python get_object_status.py --ip <GRIPPER_IP>
    python get_object_status.py --prim_path <PRIM_PATH>
    python get_object_status.py --prim_path <PRIM_PATH> --save-rrd get_object_status.rrd
"""

import argparse

import rerun as rr
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import onrobot


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def main(ip: str | None, prim_path: str | None, save_rrd: str | None) -> None:
    """Closes an OnRobot gripper and reads its object status."""

    #===================== Start Rerun ==========================================
    recording = init_rerun("telekinesis_synapse_get_object_status", save_rrd)

    #===================== Create Gripper ======================================
    gripper = onrobot.OnRobotRG6()

    try:
        #===================== Connect Gripper =================================
        if prim_path:
            gripper.connect(simulation_prim_path=prim_path)
        else:
            gripper.connect(ip=ip, protocol="MODBUS_TCP")

        # ==================== Visualization (Optional) =========================
        # Draw the gripper before it moves. Scrub the "step" timeline in the viewer
        # to compare the gripper before and after the close.
        rr.set_time("step", sequence=0)
        gripper.visualize_rerun(recording_stream=recording)

        # ==================== Run Skill ====================================
        gripper.close()
        status = gripper.get_object_status()
        logger.success(f"Object status: {status}")

        # ==================== Visualization (Optional) =========================
        rr.set_time("step", sequence=1)
        gripper.visualize_rerun(recording_stream=recording)
        rr.log("/gripper/object_status", rr.TextLog(f"Object status: {status}"))
        rr.log("/gripper/position", rr.Scalars(float(gripper.get_current_position())))
    except (ConnectionError, OSError, RuntimeError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        if gripper.is_connected:
            gripper.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gripper get object status Synapse example")
    parser.add_argument("--ip", type=str, default=None, help="IP for OnRobot Gripper")
    parser.add_argument("--prim_path", type=str, default=None,
                        help='Isaac Sim gripper prim path, e.g. "/World/onrobot_rg6"')
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()
    if not (args.ip or args.prim_path):
        parser.error("provide --ip for real hardware or --prim_path for Isaac Sim")

    main(ip=args.ip, prim_path=args.prim_path, save_rrd=args.save_rrd)
