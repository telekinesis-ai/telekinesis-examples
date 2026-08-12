# Pipeline for package label classification
# 3-stage: BiRefNet -> Contour detection -> Grounding DINO
# Visualizes 4 rerun windows and saves annotated video

import cv2
import numpy as np
import rerun as rr
from loguru import logger

from datatypes import datatypes
from telekinesis import cornea, retina, pupil

# ============================================================
# Tunable constants
# ============================================================
INPUT_VIDEO_PATH = "output/package_label.mp4"
OUTPUT_VIDEO_PATH = "output.mp4"

# Image ROI [x, y, width, height]
ROI_X, ROI_Y, ROI_W, ROI_H = 455, 207, 511, 356

# Process every Nth frame
FRAME_STRIDE = 20

# BiRefNet
BIREFNET_MASK_THRESHOLD = 0

# Contour detection
CONTOUR_MIN_AREA = 10000
CONTOUR_MAX_AREA = 10_000_000
CONTOUR_RETRIEVAL_MODE = "external"
CONTOUR_APPROX_METHOD = "simple"

# Grounding DINO
GDINO_PROMPT = "shipping label . barcode . address label . sticker ."
GDINO_BOX_THRESHOLD = 0.25
GDINO_TEXT_THRESHOLD = 0.25

# Class IDs and colors (RGB)
CLASS_NO_LABEL = 0
CLASS_LABEL_PRESENT = 1
COLOR_NO_LABEL = (255, 0, 0)       # red
COLOR_LABEL_PRESENT = (0, 255, 0)  # green

# Rerun entity paths
RR_APP_ID = "package_label_classification"
RR_PATH_INPUT = "input/frame"
RR_PATH_MASK = "stage1_birefnet/mask"
RR_PATH_CONTOUR = "stage2_contour/image"
RR_PATH_CONTOUR_BOX = "stage2_contour/image/bbox"
RR_PATH_FINAL = "stage3_final/image"
RR_PATH_FINAL_BOX = "stage3_final/image/bbox"


def get_largest_valid_contour_bbox(mask_np: np.ndarray):
    # Returns (x, y, w, h) in ROI-local coords or None
    try:
        annotations = retina.detect_contours(
            image=mask_np,
            retrieval_mode=CONTOUR_RETRIEVAL_MODE,
            approx_method=CONTOUR_APPROX_METHOD,
            min_area=CONTOUR_MIN_AREA,
            max_area=CONTOUR_MAX_AREA,
        )
    except Exception as e:
        logger.warning(f"Contour detection failed: {e}")
        return None

    if annotations is None:
        return None

    try:
        ann_list = annotations.to_list()
    except Exception as e:
        logger.warning(f"Failed to read contour annotations: {e}")
        return None

    if not ann_list:
        return None

    # Pick contour with largest bbox area
    best = None
    best_area = -1.0
    for ann in ann_list:
        bbox = ann.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        area = float(w) * float(h)
        if area > best_area:
            best_area = area
            best = (float(x), float(y), float(w), float(h))
    return best


def detect_labels_in_crop(crop_bgr: np.ndarray) -> bool:
    # Grounding DINO expects RGB
    try:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        annotations, _categories = retina.detect_objects_using_grounding_dino(
            image=crop_rgb,
            prompt=GDINO_PROMPT,
            box_threshold=GDINO_BOX_THRESHOLD,
            text_threshold=GDINO_TEXT_THRESHOLD,
        )
    except Exception as e:
        logger.warning(f"Grounding DINO failed: {e}")
        return False

    if annotations is None:
        return False

    try:
        ann_list = annotations.to_list()
    except Exception as e:
        logger.warning(f"Failed to read GDINO annotations: {e}")
        return False

    return len(ann_list) > 0


def main():
    logger.info("Initializing rerun")
    rr.init(RR_APP_ID, spawn=True)

    logger.info(f"Opening input video: {INPUT_VIDEO_PATH}")
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        logger.error(f"Could not open input video: {INPUT_VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video: {frame_w}x{frame_h} @ {fps:.2f} FPS")

    # Output video writer (write at reduced FPS to match stride)
    out_fps = max(1.0, fps / FRAME_STRIDE)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, out_fps, (frame_w, frame_h))
    if not writer.isOpened():
        logger.error(f"Could not open output video for writing: {OUTPUT_VIDEO_PATH}")
        cap.release()
        return

    frame_idx = 0
    processed = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                logger.info("End of video reached")
                break

            if frame_idx % FRAME_STRIDE != 0:
                frame_idx += 1
                continue

            logger.info(f"Processing frame {frame_idx}")

            # Convert to RGB for visualization and processing
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ---- Window 1: input frame ----
            rr.log(RR_PATH_INPUT, rr.Image(frame_rgb))

            # ---- Crop ROI ----
            x0 = max(0, ROI_X)
            y0 = max(0, ROI_Y)
            x1 = min(frame_w, ROI_X + ROI_W)
            y1 = min(frame_h, ROI_Y + ROI_H)
            roi_rgb = frame_rgb[y0:y1, x0:x1].copy()
            if roi_rgb.size == 0:
                logger.warning(f"Frame {frame_idx}: empty ROI, skipping")
                frame_idx += 1
                continue

            # ---- Stage 1: BiRefNet ----
            logger.info(f"Frame {frame_idx}: Stage 1 - BiRefNet foreground segmentation")
            try:
                pano_ann = cornea.segment_image_using_foreground_birefnet(
                    image=roi_rgb,
                    mask_threshold=BIREFNET_MASK_THRESHOLD,
                )
            except Exception as e:
                logger.warning(f"Frame {frame_idx}: BiRefNet failed: {e}")
                frame_idx += 1
                continue

            if pano_ann is None:
                logger.warning(f"Frame {frame_idx}: BiRefNet returned no annotation, skipping")
                frame_idx += 1
                continue

            try:
                labeled_mask = pano_ann.labeled_mask
                if isinstance(labeled_mask, datatypes.Image):
                    mask_np = labeled_mask.to_numpy()
                else:
                    mask_np = np.asarray(labeled_mask)
            except Exception as e:
                logger.warning(f"Frame {frame_idx}: failed to extract BiRefNet mask: {e}")
                frame_idx += 1
                continue

            # Make a binary uint8 mask for downstream + visualization
            if mask_np.ndim == 3:
                mask_gray = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
            else:
                mask_gray = mask_np
            mask_bin = (mask_gray > 0).astype(np.uint8) * 255

            # ---- Window 2: BiRefNet mask ----
            rr.log(RR_PATH_MASK, rr.Image(mask_bin))

            # ---- Stage 2: contour detection ----
            logger.info(f"Frame {frame_idx}: Stage 2 - Contour detection")
            bbox_local = get_largest_valid_contour_bbox(mask_bin)

            # Visualize stage 2 (ROI image + bbox if any)
            rr.log(RR_PATH_CONTOUR, rr.Image(roi_rgb))

            if bbox_local is None:
                logger.warning(f"Frame {frame_idx}: no valid contour bbox, skipping to next frame")
                # Clear any previous bbox on this entity
                rr.log(RR_PATH_CONTOUR_BOX, rr.Clear(recursive=False))
                rr.log(RR_PATH_FINAL, rr.Image(frame_rgb))
                rr.log(RR_PATH_FINAL_BOX, rr.Clear(recursive=False))
                # Still write the unannotated frame to output
                writer.write(frame_bgr)
                frame_idx += 1
                processed += 1
                continue

            bx, by, bw, bh = bbox_local
            # Clip bbox to ROI
            bx_i = int(max(0, bx))
            by_i = int(max(0, by))
            bw_i = int(max(1, min(roi_rgb.shape[1] - bx_i, bw)))
            bh_i = int(max(1, min(roi_rgb.shape[0] - by_i, bh)))

            # Log stage 2 bbox (centers + half_sizes, in ROI image coords)
            cx_local = bx_i + bw_i / 2.0
            cy_local = by_i + bh_i / 2.0
            rr.log(
                RR_PATH_CONTOUR_BOX,
                rr.Boxes2D(
                    centers=[[cx_local, cy_local]],
                    half_sizes=[[bw_i / 2.0, bh_i / 2.0]],
                    colors=[[0, 255, 255]],
                    labels=["package"],
                ),
            )

            # ---- Stage 3: Grounding DINO on bbox crop ----
            logger.info(f"Frame {frame_idx}: Stage 3 - Grounding DINO label detection")
            crop_rgb = roi_rgb[by_i:by_i + bh_i, bx_i:bx_i + bw_i].copy()
            if crop_rgb.size == 0:
                logger.warning(f"Frame {frame_idx}: empty bbox crop, treating as no_label")
                has_label = False
            else:
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                has_label = detect_labels_in_crop(crop_bgr)

            if has_label:
                class_id = CLASS_LABEL_PRESENT
                color = COLOR_LABEL_PRESENT
                label_text = "label_present"
            else:
                class_id = CLASS_NO_LABEL
                color = COLOR_NO_LABEL
                label_text = "no_label"

            logger.info(f"Frame {frame_idx}: classification = {label_text} (class {class_id})")

            # Convert bbox to full-frame coords
            fx = x0 + bx_i
            fy = y0 + by_i
            fw = bw_i
            fh = bh_i

            # ---- Window 4: final annotated frame ----
            annotated_bgr = frame_bgr.copy()
            cv2.rectangle(
                annotated_bgr,
                (fx, fy),
                (fx + fw, fy + fh),
                (color[2], color[1], color[0]),  # BGR
                3,
            )
            cv2.putText(
                annotated_bgr,
                label_text,
                (fx, max(0, fy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (color[2], color[1], color[0]),
                2,
                cv2.LINE_AA,
            )

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            rr.log(RR_PATH_FINAL, rr.Image(annotated_rgb))
            rr.log(
                RR_PATH_FINAL_BOX,
                rr.Boxes2D(
                    centers=[[fx + fw / 2.0, fy + fh / 2.0]],
                    half_sizes=[[fw / 2.0, fh / 2.0]],
                    colors=[[color[0], color[1], color[2]]],
                    labels=[label_text],
                    class_ids=[class_id],
                ),
            )

            # Write annotated frame to output video
            writer.write(annotated_bgr)
            processed += 1
            frame_idx += 1

    finally:
        logger.info(f"Processed {processed} frames; releasing resources")
        cap.release()
        writer.release()
        logger.info(f"Output video saved to {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
