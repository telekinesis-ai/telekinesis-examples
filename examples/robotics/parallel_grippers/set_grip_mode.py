"""
Grips the inside of a bore with a parallel gripper, and shows the gripper in Rerun.

Supports OnRobot and Robotiq grippers, and Isaac Sim.

Usage:
    python set_grip_mode.py --ip <GRIPPER_IP>
    python set_grip_mode.py --prim_path <PRIM_PATH>
    python set_grip_mode.py --prim_path <PRIM_PATH> --save-rrd set_grip_mode.rrd
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
    """Sets inner grip mode, then grips and releases."""

    #===================== Start Rerun ==========================================
    recording = init_rerun("telekinesis_synapse_set_grip_mode", save_rrd)

    #===================== Create Gripper ======================================
    gripper = onrobot.OnRobotRG6()

    # ==================== Run Skill ============================================
    # Call it right after creating the gripper: before any close() or open(),
    # and before set_max_opening_width_mm(), which inner mode ignores.
    gripper.set_grip_mode("inner")

    try:
        #===================== Connect Gripper =================================
        if prim_path:
            gripper.connect(simulation_prim_path=prim_path)
        else:
            gripper.connect(ip=ip, protocol="MODBUS_TCP")

        # ==================== Visualization (Optional) =========================
        # Draw the gripper at each stage. Scrub the "step" timeline in the viewer:
        # in inner mode the fingers spread on close and close again on open.
        rr.set_time("step", sequence=0)
        gripper.visualize_rerun(recording_stream=recording)
        rr.log("/gripper/stage", rr.TextLog("Connected, inner grip mode"))

        gripper.close()  # the fingers spread against the bore
        part_present = gripper.get_part_present()
        logger.success(f"Part held: {part_present}")
        rr.set_time("step", sequence=1)
        gripper.visualize_rerun(recording_stream=recording)
        rr.log("/gripper/stage", rr.TextLog(f"close(): fingers spread, part held: {part_present}"))

        gripper.open()   # the fingers close: the part is released
        rr.set_time("step", sequence=2)
        gripper.visualize_rerun(recording_stream=recording)
        rr.log("/gripper/stage", rr.TextLog("open(): fingers closed, part released"))
    except (ConnectionError, OSError, RuntimeError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        if gripper.is_connected:
            gripper.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gripper set grip mode Synapse example")
    parser.add_argument("--ip", type=str, default=None, help="IP for OnRobot Gripper")
    parser.add_argument("--prim_path", type=str, default=None,
                        help='Isaac Sim gripper prim path, e.g. "/World/onrobot_rg6"')
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()
    if not (args.ip or args.prim_path):
        parser.error("provide --ip for real hardware or --prim_path for Isaac Sim")

    main(ip=args.ip, prim_path=args.prim_path, save_rrd=args.save_rrd)
