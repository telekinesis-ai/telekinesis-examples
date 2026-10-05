"""Run local predictions with an exported Iris model."""

import argparse
from pathlib import Path

import rerun as rr
from loguru import logger

from telekinesis.iris import deploy, visualization


def predict_model_example(args: argparse.Namespace) -> None:
    """Load an exported model and predict one image."""

    # ===================== Load Model =====================================
    model = deploy.Model(
        args.model,
        model_name=args.model_name,
        class_names=args.class_names,
        num_select=args.max_detections,
        device=args.device,
    )

    # ===================== Run Skill ======================================
    prediction = model.predict(
        image=args.image,
        confidence_threshold=args.confidence_threshold,
        prompts=args.prompts,
    )
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
    rr.init("iris_prediction", spawn=True)
    visualization.visualize_prediction(args.image, prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to an exported ONNX model or Iris bundle.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image on which to run prediction.",
    )
    parser.add_argument(
        "--model-name",
        help="Optional model name; the artifact extension selects the family when omitted.",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        help="Optional RFDETR class names or SAM3-LoRA prompt names.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        help="Optional prompts for a SAM3-LoRA model.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Optional per-prediction confidence threshold.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        help="Maximum detections to return.",
    )
    parser.add_argument(
        "--device",
        help="PyTorch device for SAM3-LoRA, such as 'cuda' or 'cpu'.",
    )
    predict_model_example(parser.parse_args())
