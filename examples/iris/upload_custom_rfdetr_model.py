"""Upload and register a custom RF-DETR model with Iris."""

from pathlib import Path

from loguru import logger

from telekinesis import iris_backend


def upload_custom_rfdetr_model_example():
    """Upload an RF-DETR model and register its object classes."""
    # ===================== Load Model =========================================
    model_path = Path(
        r"C:\Users\AmruthaVenkatesan\Documents\Telekinesis\Code"
        r"\telekinesis-iris\results\yu_model\model.onnx"
    )
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
    upload_custom_rfdetr_model_example()
