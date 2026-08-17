"""
Minimal robot-driven data collection for calibration of an eye-in-hand camera.

The robot moves the camera around a ChArUco target and saves the images and corresponding
robot poses to disk. The resulting dataset can be used for eye-in-hand calibration with
calibrate_eye_in_hand.py.

Output:
    data/cam_00/image_NN.png
    data/cam_00/calibration_data.npz  ← robot_T_tcp_list
"""

import argparse
from pathlib import Path

from telekinesis.medulla.cameras import realsense
from telekinesis.synapse.robots.manipulators import universal_robots

from telekinesis import axon
from telekinesis.axon import targets

# Default data dir
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "eye_in_hand"

# Default home pose and perturbation parameters for the robot sampler
HOME_POSE = [0.35, 0.1, 0.5, 180.0, 0.0, -90.0]
MAX_ROTATION_DEG = 15.0
MAX_TRANSLATION_M = 0.05

# Default connection parameters for the robot and camera
ROBOT_IP = "192.168.1.100"
CAM_SERIAL_PORT = "YOUR_CAMERA_SERIAL_NUMBER"


def parse_args():
    """
    Parse command line arguments for the data collection script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--camera-serial", default=CAM_SERIAL_PORT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
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
    parser.add_argument("--squares-x", type=int, default=6)
    parser.add_argument("--squares-y", type=int, default=9)
    parser.add_argument("--square-length", type=float, default=0.012, help="meters")
    parser.add_argument("--marker-length", type=float, default=0.009, help="meters")
    parser.add_argument("--aruco-dict-id", default="DICT_4X4_50")
    parser.add_argument(
        "--wipe", action="store_true", help="Delete existing data-dir before collecting"
    )
    return parser.parse_args()


def main(args):
    """
    Collect calibration data.

    1. Define the ChArUco target and perturbation sampler.
    2. Connect to the camera and robot.
    3. Move the robot to a series of poses and capture, and save images using DataCollector.
    4. Disconnect from the camera and robot.
    """

    # Define the ChArUco target
    target = targets.CharucoTarget(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        aruco_dict_id=args.aruco_dict_id,
    )

    # Define the perturbation sampler to generate robot poses around the home pose
    sampler = axon.PerturbationSampler(
        args.home_pose_deg,
        rotation_deg=args.rotation_deg,
        translation_m=args.translation_m,
        rotation_steps=args.rotation_steps,
        translation_steps=args.translation_steps,
    )

    # Define the camera and robot
    cam = realsense.RealSense(name="cam_00", serial_number=args.camera_serial)
    robot = universal_robots.UniversalRobotsUR10E(name="robot")

    try:
        # Connect to the camera and robot
        cam.connect()
        robot.connect(ip=args.robot_ip)

        # Start collection
        collector = axon.DataCollector(robot, [cam], target, sampler)
        result = collector.collect(args.data_dir, wipe=args.wipe)

        print(f"{'Done' if result['ok'] else 'Warning: <4 frames'} — data in {args.data_dir}")

    except KeyboardInterrupt:
        # Frames already written to data-dir are kept -- collect() writes as it goes
        print(f"\nInterrupted — stopping collection. Partial data in {args.data_dir}")

    finally:
        # Disconnect
        cam.disconnect()
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    main(parse_args())
