"""Export a SAM3 LoRA training checkpoint to an Iris PyTorch bundle."""

import argparse
from pathlib import Path

from loguru import logger
from telekinesis.iris import export


def export_custom_sam3_lora_model_example(args: argparse.Namespace) -> None:
    """Export the best SAM3 LoRA checkpoint as a deployment bundle."""
    artifact_path = export.export_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model_name="sam3-lora",
        artifact_name="model",
    )
    logger.info(
        "Exported SAM3 LoRA model in Iris PyTorch bundle format: "
        f"{artifact_path.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained SAM3 LoRA checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to save the exported model bundle.",
    )
    args = parser.parse_args()
    export_custom_sam3_lora_model_example(args)
