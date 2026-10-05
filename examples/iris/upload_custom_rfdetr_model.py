"""Upload and register a custom RF-DETR model with Iris."""

import argparse
from pathlib import Path

from loguru import logger

from telekinesis import iris_backend


def deploy_custom_rfdetr_model_example(args: argparse.Namespace) -> None:
    """Upload an RF-DETR model and register its object classes."""
    # ===================== Load Model =========================================

    model_path = args.model
    model_name = "rfdetr_segmedium_epson_real_original"
    class_names = [
        "black_cable_gland",
        "multi_pin_circular_connector",
        "panel_mount_pilot_light",
        "50_pin_male_Centronics_connector",
    ]

    # ===================== Run Skill ==========================================
    manifest = iris_backend.model_upload(
        model_path=model_path,
        model_name=model_name,
        class_names=class_names,
    )

    # ===================== Log ================================================
    logger.success(f"Uploaded {model_path} as {model_name!r}")
    logger.info(f"Registration manifest: {manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the exported RF-DETR ONNX model.",
    )
    args = parser.parse_args()
    deploy_custom_rfdetr_model_example(args)
