"""Run RF-DETR inference on one image."""

import argparse
from pathlib import Path

import rerun as rr
from loguru import logger

from telekinesis.iris import deploy, visualization


def infer_custom_rfdetr_model_local_example(args: argparse.Namespace) -> None:
    """Load one exported RF-DETR model and predict one image."""

    # ===================== Load Model =====================================
    model = deploy.Model(
        args.model,
        class_names={1: "green_circle", 2: "red_triangle", 3: "blue_square"},
        postprocess=deploy.RFDETRPostprocessConfig(
            confidence_threshold=0.5,
            max_detections=100,
        ),
    )

    # ===================== Run Skill ======================================
    prediction = model.predict(image=args.image)
    logger.info(f"Found {len(prediction)} detections in {args.image.name}")
    for box, score, class_id, class_name in zip(
        prediction.boxes,
        prediction.scores,
        prediction.class_ids,
        prediction.class_names or map(str, prediction.class_ids),
        strict=True,
    ):
        logger.info(
            f"class={class_name} id={class_id} score={score:.3f} "
            f"box={box.round(1).tolist()}"
        )

    # ===================== Visualization =================================
    rr.init("rfdetr_inference", spawn=True)
    visualization.visualize_prediction(args.image, prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the exported RF-DETR ONNX model.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image on which to run inference.",
    )
    args = parser.parse_args()
    infer_custom_rfdetr_model_local_example(args)
