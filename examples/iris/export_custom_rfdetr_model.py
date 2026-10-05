"""Export an RF-DETR training checkpoint to ONNX."""

import argparse
from pathlib import Path

from loguru import logger
from telekinesis.iris import export


def export_custom_rfdetr_model_example(args: argparse.Namespace) -> None:
    """Export the best RF-DETR checkpoint as an ONNX model."""
    artifact_path = export.export_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model_name="seg-medium",
        artifact_name="model",
    )
    logger.info(f"Exported RF-DETR model in ONNX format: {artifact_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained RF-DETR checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to save the exported ONNX model.",
    )
    args = parser.parse_args()
    export_custom_rfdetr_model_example(args)
