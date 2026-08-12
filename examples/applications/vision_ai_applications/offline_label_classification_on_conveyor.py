
# Pipeline: Classify styrofoam boxes as empty/non-empty using multi-stage vision
# Stages: ROI crop -> BiRefNet foreground mask -> contour detection -> HSV-based classification
# Visualizations: 4 rerun windows (input, mask, contour bbox, final)

import cv2
import numpy as np
import rerun as rr
from loguru import logger

from datatypes import datatypes
from telekinesis import cornea, pupil, retina

# =====================================================================
# Hardcoded constants / tunables
# =====================================================================
VIDEO_PATH = "db_boxes.mp4"
OUTPUT_VIDEO_PATH = "db_boxes_annotated.mp4"

ROI = [455, 207, 511, 356]  # x, y, width, height
FRAME_SKIP = 1

# Stage 2: contour filtering
MIN_CONTOUR_AREA = 500

# Stage 3: HSV classification thresholds
S_COLOR_THRESHOLD = 60       # S > 60 -> colorful pixel
V_DARK_THRESHOLD = 70        # V < 70 -> dark pixel
CLASSIFICATION_RATIO = 0.08  # ratio of (colorful + dark) pixels => non_empty

# Classes
CLASS_NAMES = {0: "empty", 1: "non_empty"}
CLASS_COLORS = {
    0: (255, 0, 0),   # red for empty (RGB)
    1: (0, 255, 0),   # green for non_empty (RGB)
}

# BiRefNet mask binarization threshold (pixel value)
BIREFNET_MASK_THRESHOLD = 0

# Rerun entity paths
RR_APP_ID = "styrofoam_box_classifier"
RR_INPUT = "input_video"
RR_MASK = "birefnet_mask"
RR_CONTOUR = "contour_bbox_on_roi"
RR_FINAL = "final_bbox"


def classify_hsv_box(box_bgr: np.ndarray) -> tuple[int, float]:
    """HSV-based classification of styrofoam box crop.

    Returns (class_id, ratio).
    """
    if box_bgr is None or box_bgr.size == 0:
        return 0, 0.0
    hsv = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    colorful = S > S_COLOR_THRESHOLD
    dark = V < V_DARK_THRESHOLD
    combined = colorful | dark
    ratio = float(np.count_nonzero(combined)) / float(combined.size)
    cls = 1 if ratio > CLASSIFICATION_RATIO else 0
    return cls, ratio


def main() -> None:
    logger.info("Initializing rerun viewer (app_id={})", RR_APP_ID)
    rr.init(RR_APP_ID, spawn=True)

    logger.info("Opening input video: {}", VIDEO_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error("Failed to open video file: {}", VIDEO_PATH)
        return

    # Setup VideoWriter for annotated full-frame output (window 4 style)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = max(1.0, fps / FRAME_SKIP)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, out_fps, (width, height))
    if not writer.isOpened():
        logger.error("Failed to open VideoWriter for output: {}", OUTPUT_VIDEO_PATH)
        cap.release()
        return

    x_roi, y_roi, w_roi, h_roi = ROI

    frame_idx = -1
    processed = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                logger.info("End of video reached at frame_idx={}", frame_idx)
                break
            frame_idx += 1

            if frame_idx % FRAME_SKIP != 0:
                continue

            logger.info("Processing frame {}", frame_idx)

            # Convert BGR -> RGB for rerun / SDK functions that expect RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Window 1: input video (full frame)
            rr.log(RR_INPUT, rr.Image(frame_rgb))

            # =============================================================
            # Stage 1: ROI crop + BiRefNet foreground mask
            # =============================================================
            H, W = frame_rgb.shape[:2]
            x1 = max(0, x_roi)
            y1 = max(0, y_roi)
            x2 = min(W, x_roi + w_roi)
            y2 = min(H, y_roi + h_roi)
            if x2 <= x1 or y2 <= y1:
                logger.warning("Frame {}: invalid ROI after clamping; skipping", frame_idx)
                writer.write(frame_bgr)
                continue

            roi_rgb = frame_rgb[y1:y2, x1:x2].copy()
            logger.info("Stage 1: running BiRefNet on ROI of shape {}", roi_rgb.shape)

            mask_img_np = None
            try:
                pano = cornea.segment_image_using_foreground_birefnet(
                    image=roi_rgb, mask_threshold=BIREFNET_MASK_THRESHOLD
                )
                # PanopticSegmentationAnnotation has labeled_mask (Image or ndarray)
                lm = getattr(pano, "labeled_mask", None)
                if isinstance(lm, datatypes.Image):
                    mask_img_np = lm.to_numpy()
                elif isinstance(lm, np.ndarray):
                    mask_img_np = lm
                else:
                    logger.warning("Frame {}: BiRefNet returned unexpected mask type", frame_idx)
            except Exception as e:
                logger.exception("Frame {}: BiRefNet failed: {}", frame_idx, e)
                mask_img_np = None

            if mask_img_np is None or mask_img_np.size == 0:
                logger.warning(
                    "Frame {}: empty/invalid BiRefNet mask; skipping classification",
                    frame_idx)
                writer.write(frame_bgr)
                continue

            # Binarize the labeled mask: any non-zero label = foreground
            if mask_img_np.ndim == 3:
                mask_gray = cv2.cvtColor(mask_img_np, cv2.COLOR_RGB2GRAY)
            else:
                mask_gray = mask_img_np
            binary_mask = (mask_gray > 0).astype(np.uint8) * 255

            # Window 2: BiRefNet foreground mask
            rr.log(RR_MASK, rr.Image(binary_mask))

            # =============================================================
            # Stage 2: contour detection on the mask -> largest bbox
            # =============================================================
            logger.info("Stage 2: contour detection on BiRefNet mask")
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
            if not valid:
                logger.warning(
                    "Frame {}: no contours above min_area={}; skipping",
                    frame_idx, MIN_CONTOUR_AREA,
                )
                writer.write(frame_bgr)
                continue

            largest = max(valid, key=cv2.contourArea)
            cx, cy, cw, ch = cv2.boundingRect(largest)
            logger.info("Stage 2: largest contour bbox (ROI coords) = {}", (cx, cy, cw, ch))

            # Draw contour bbox on a copy of the ROI for window 3
            roi_vis_rgb = roi_rgb.copy()
            cv2.rectangle(
                roi_vis_rgb, (cx, cy), (cx + cw, cy + ch), (255, 255, 0), 2
            )
            rr.log(RR_CONTOUR, rr.Image(roi_vis_rgb))

            # Crop styrofoam box region from ROI (use BGR for HSV conversion)
            roi_bgr = frame_bgr[y1:y2, x1:x2]
            box_bgr = roi_bgr[cy: cy + ch, cx: cx + cw]
            if box_bgr.size == 0:
                logger.warning("Frame {}: empty box crop after contour bbox; skipping", frame_idx)
                writer.write(frame_bgr)
                continue

            # =============================================================
            # Stage 3: HSV-based classification
            # =============================================================
            logger.info("Stage 3: HSV classification on box crop of shape {}", box_bgr.shape)
            cls_id, ratio = classify_hsv_box(box_bgr)
            cls_name = CLASS_NAMES[cls_id]
            color_rgb = CLASS_COLORS[cls_id]
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            logger.info(
                "Frame {}: classified as '{}' (id={}) with ratio={:.4f}",
                frame_idx, cls_name, cls_id, ratio,
            )

            # Map contour bbox from ROI coords back to full-frame coords
            fx1 = x1 + cx
            fy1 = y1 + cy
            fx2 = fx1 + cw
            fy2 = fy1 + ch

            # Window 4: final bounding box on full frame with class color
            annotated_bgr = frame_bgr.copy()
            cv2.rectangle(annotated_bgr, (fx1, fy1), (fx2, fy2), color_bgr, 2)
            cv2.putText(
                annotated_bgr,
                cls_name,
                (fx1, max(0, fy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_bgr,
                2,
            )

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            rr.log(RR_FINAL, rr.Image(annotated_rgb))

            # Save annotated frame to output video
            writer.write(annotated_bgr)
            processed += 1

    except Exception as e:
        logger.exception("Unexpected error in pipeline: {}", e)
    finally:
        logger.info("Releasing resources. Processed {} frames.", processed)
        cap.release()
        writer.release()
        logger.info("Annotated video saved to: {}", OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    main()
