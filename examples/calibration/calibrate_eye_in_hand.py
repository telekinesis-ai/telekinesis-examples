"""
Eye-in-hand calibration from a previously collected dataset.

Loads images + robot TCP poses saved by collect_calibration_dataset.py and
solves for the tcp_T_camera hand-eye transform.
"""

import argparse
import pathlib

from telekinesis import axon
from telekinesis.axon import io, targets

# Default data and output dirs
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "eye_in_hand"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "calibrations"


def parse_args():
    """
    Parse command line arguments for the data collection script.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=DATA_DIR,
        help="Directory containing cam_00/ from collect_calibration_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=OUTPUT_DIR,
        help="Directory to write the hand-eye calibration result to",
    )
    parser.add_argument(
        "--method",
        default="TSAI",
        choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"],
        help="Hand-eye solver method",
    )
    parser.add_argument("--squares-x", type=int, default=6)
    parser.add_argument("--squares-y", type=int, default=9)
    parser.add_argument("--square-length", type=float, default=0.012, help="meters")
    parser.add_argument("--marker-length", type=float, default=0.009, help="meters")
    parser.add_argument("--aruco-dict-id", default="DICT_4X4_50")
    return parser.parse_args()


def main(args):
    """
    Generate the eye_in-hand calibration result from a previously collected dataset.

    1. Create and load the same target used for data collection (ChArUco board).
    2. Load the saved dataset from DATA_DIR/cam_00/ and run eye-in-hand calibration.
    3. Calibrate and save the result

    """

    # Create the same target used for data collection
    target = targets.CharucoTarget(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        aruco_dict_id=args.aruco_dict_id,
    )

    # Load the saved dataset from DATA_DIR/cam_00/ and run eye-in-hand calibration
    images, robot_poses = io.load_hand_eye_dataset(args.data_dir / "cam_00")

    # Calibrate
    calibrator = axon.EyeInHandCalibrator(target)
    result = calibrator.calibrate(
        robot_T_tcp_list=robot_poses,
        image_list=images,
        method=args.method,
        output_path=args.output_dir,
    )

    if not result.ok:
        raise RuntimeError("Eye-in-hand calibration failed — not enough valid frames.")


if __name__ == "__main__":
    main(parse_args())
