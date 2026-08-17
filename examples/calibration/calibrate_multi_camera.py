"""
Multi-camera extrinsic calibration from a previously collected dataset.

Loads a synchronized multi-camera dataset and solves for each camera's pose
relative to the reference camera via pairwise stereo calibration.
"""

import argparse
import pathlib

from telekinesis import axon
from telekinesis.axon import io, targets

# Default data and output dirs
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "multi_camera"
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
        help="Directory containing cam_00/, cam_01/, ... from data collection",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=OUTPUT_DIR,
        help="Directory to write the multi-camera calibration results to",
    )
    parser.add_argument("--num-cameras", type=int, default=3)
    parser.add_argument(
        "--reference-index",
        type=int,
        default=1,
        help="Index of the camera treated as the reference frame",
    )
    parser.add_argument("--squares-x", type=int, default=6)
    parser.add_argument("--squares-y", type=int, default=9)
    parser.add_argument("--square-length", type=float, default=0.012, help="meters")
    parser.add_argument("--marker-length", type=float, default=0.009, help="meters")
    parser.add_argument("--aruco-dict-id", default="DICT_4X4_1000")
    return parser.parse_args()


def main(args):
    """
    Calibrate a multi-camera rig from a previously collected dataset.

    1. Create and load the same target used for data collection (ChArUco board).
    2. Load the saved dataset from DATA_DIR/cam_XX/ and run multi-camera calibration.
    3. Calibrate and save the results to OUTPUT_DIR.
    """

    # Create the same target used for data collection
    target = targets.CharucoTarget(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        aruco_dict_id=args.aruco_dict_id,
    )

    # Load the saved dataset from DATA_DIR/cam_XX/ and run multi-camera calibration
    image_lists = io.load_multi_camera_dataset(args.data_dir, num_cameras=args.num_cameras)

    # Calibrate and save the results
    calibrator = axon.MultiCameraCalibrator(
        target, num_cameras=args.num_cameras, reference_index=args.reference_index
    )
    result = calibrator.calibrate(image_lists, output_path=args.output_dir)

    if not result.ok:
        raise RuntimeError("Multi-camera calibration failed for at least one camera pair.")


if __name__ == "__main__":
    main(parse_args())
