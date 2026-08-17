"""
Full eye-in-hand calibration: data collection → calibration in one script.

Step 1 — Drives the robot through a set of perturbation poses and saves
         synchronized frames + TCP poses to DATA_DIR/cam_00/.

Step 2 — Loads the saved dataset and runs eye-in-hand calibration,
         printing and saving the tcp_T_camera result.

Pass --help to see all overridable options, or edit the defaults below to
match your hardware and board.
"""

import argparse
import pathlib

from telekinesis.medulla.cameras import ids
from telekinesis.synapse.robots.manipulators import universal_robots

from telekinesis import axon
from telekinesis.axon import targets

# Default data and output dirs
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "eye_in_hand"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "calibrations"

# Default home pose and perturbation parameters for the robot sampler
HOME_POSE = [0.5, -0.2, 0.4, 180.0, 0.0, 0.0]
MAX_ROTATION_DEG = 15.0
MAX_TRANSLATION_M = 0.05

# Default connection parameters for the robot and camera
ROBOT_IP = "192.168.1.100"
CAMERA_SERIAL = "YOUR_CAMERA_SERIAL_NUMBER"


def parse_args():
    """
    Parse command line arguments for the data collection script.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # ── Hardware ──────────────────────────────────────────────────────────
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--camera-serial", default=CAMERA_SERIAL)

    # ── Calibration board ────────────────────────────────────────────────
    parser.add_argument("--squares-x", type=int, default=6)
    parser.add_argument("--squares-y", type=int, default=9)
    parser.add_argument("--square-length", type=float, default=0.030, help="meters")
    parser.add_argument("--marker-length", type=float, default=0.022, help="meters")
    parser.add_argument("--aruco-dict-id", default="DICT_4X4_50")

    # ── Perturbation sampling ────────────────────────────────────────────
    parser.add_argument(
        "--home-pose-deg",
        type=float,
        nargs=6,
        default=HOME_POSE,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Robot home pose the sampler perturbs around",
    )
    parser.add_argument(
        "--rotation-deg", type=float, default=MAX_ROTATION_DEG, help="Max rotation per axis"
    )
    parser.add_argument(
        "--translation-m", type=float, default=MAX_TRANSLATION_M, help="Max translation per axis"
    )
    parser.add_argument("--rotation-steps", type=int, default=3)
    parser.add_argument("--translation-steps", type=int, default=2)

    # ── Calibration method / output ──────────────────────────────────────
    parser.add_argument(
        "--method",
        default="TSAI",
        choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"],
        help="Hand-eye solver method",
    )
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--wipe", action="store_true", help="Delete existing data-dir before collecting"
    )
    return parser.parse_args()


def main(args):
    """
    Full pipeline: collect calibration data and run eye-in-hand calibration.

    1. Define the ChArUco target and perturbation sampler.
    2. Connect to the camera and robot.
    3. Move the robot to a series of poses and capture, and save images using DataCollector.
    4. Disconnect from the camera and robot.
    5. Load the saved dataset from DATA_DIR/cam_00/ and run eye-in-hand calibration.
    6. Calibrate and save the result to OUTPUT_DIR.

    """
    ## Define the ChArUco target
    target = targets.CharucoTarget(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        aruco_dict_id=args.aruco_dict_id,
    )

    # ── Step 1: collect data ─────────────────────────────────────────────

    # Define camera and robot
    cam = ids.IDS(name="cam_00", serial_number=args.camera_serial)
    robot = universal_robots.UniversalRobotsUR10E()

    try:
        # connect to camera and robot
        cam.connect()
        robot.connect(ip=args.robot_ip)

        # Define the perturbation sampler to generate robot poses around the home pose
        sampler = axon.PerturbationSampler(
            args.home_pose_deg,
            rotation_deg=args.rotation_deg,
            translation_m=args.translation_m,
            rotation_steps=args.rotation_steps,
            translation_steps=args.translation_steps,
        )

        # Collect data by moving the robot to a series of poses and capturing images
        collector = axon.DataCollector(robot, [cam], target, sampler)
        result = collector.collect(args.data_dir, wipe=args.wipe)

        if not result["ok"]:
            print("Warning: fewer than 4 valid captures — calibration may fail.")

    finally:
        # Disconnect from camera and robot
        cam.disconnect()
        robot.disconnect()
        robot.shutdown()

    # ── Step 2: calibrate ────────────────────────────────────────────────

    # Load the saved dataset from DATA_DIR/cam_00/ and run eye-in-hand calibration
    calibrator = axon.EyeInHandCalibrator(target)
    calib_result = calibrator.calibrate(
        robot_T_tcp_list=result["robot_T_tcp_list"],
        image_list=result["images"][0],
        method=args.method,
        output_path=args.data_dir,
    )

    if not calib_result.ok:
        raise RuntimeError("Eye-in-hand calibration failed — not enough valid frames.")


if __name__ == "__main__":
    main(parse_args())
