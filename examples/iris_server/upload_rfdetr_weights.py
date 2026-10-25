"""Upload and register an RF-DETR model with Iris Server."""

import argparse
from pathlib import Path

from loguru import logger

from telekinesis import iris_backend


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the exported RF-DETR ONNX model.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Name under which to register the RF-DETR model.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        required=True,
        help="Object class names in model output order.",
    )
    return parser.parse_args()


def upload_rfdetr_model(args: argparse.Namespace) -> None:
    """Upload an RF-DETR model and register its object classes."""
    manifest = iris_backend.model_upload(
        model_path=args.model,
        model_name=args.model_name,
        class_names=args.class_names,
    )
    logger.success(f"Uploaded {args.model} as {args.model_name!r}")
    logger.info(f"Registration manifest: {manifest}")


if __name__ == "__main__":
    upload_rfdetr_model(parse_args())
