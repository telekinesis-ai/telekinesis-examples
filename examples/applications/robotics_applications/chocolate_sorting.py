# Pipeline: Sort colored chocolates (yellow/orange/brown) into separate bins
# Hardware: UR10E + OnRobot RG2 + Intel RealSense
# Vision: Grounding DINO (detection) -> SAM (segmentation) -> classical
# HSV color classification -> PCA for grasp orientation"

"""
Prompt:
In the bin there are 3 different types of chocolate, yellow, orange and brown.
Can you sort them into 3 different bins. Do qwen/dino to get all chocolates,
 and then use again qwen/dino to classify which color it is. Do PCA on the mask
 for grasping orientation. Add detailed logging and Rerun visualization for
 every stage and step so we can see what is going on. The goal is robust
 debugging and traceability so failures can be diagnosed easily.


Claude followup:
- yellow classified as brown (fixed hsv ranges)
- orange classified as brown(fixed hsv ranges)
- pca wrong matched to correct version (was not sampling and then projecting but using raw)
...
"""


import signal
import sys
import time
import numpy as np
import cv2
import rerun as rr
from loguru import logger

from telekinesis.synapse import utils as tfutils
from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E
from telekinesis.synapse.tools.parallel_grippers.onrobot import OnRobotRG2
from telekinesis.medulla.cameras.realsense import RealSense

from telekinesis import retina
from telekinesis import cornea
from telekinesis import pupil
from datatypes import datatypes

# ============================================================
# HARDCODED / TUNABLE CONSTANTS
# ============================================================
ROBOT_IP = "192.168.2.2"
GRIPPER_IP = "192.168.1.1"
CAMERA_NAME = "realsense_top"

# TCP offset (23cm from flange along z)
TCP_OFFSET = [0.0, 0.0, 0.23, 0.0, 0.0, 0.0]

# Hand-eye calibration
CAMERA_IN_TCP = [0.07520960896570618, -0.0352478269641629, -0.2162654145229983,
                 -0.07505179364087063, 0.8826477579985493, 90.3598403373567]
TCP_T_CAMERA = tfutils.pose_to_transformation_matrix(CAMERA_IN_TCP, rot_type='deg')

# Poses
SCAN_POSE = [-0.25462, 0.59302, 0.3141, 180.0, 0.0, 90.0]
HOME_JOINT_POSITIONS = [120, -90, -90, -90, 90, -90]
INTERMEDIATE_JOINT_POSITIONS = [120, -70, -120, -80, 90, -90]

YELLOW_BIN_POSE = [0.3041, 0.85220, 0.086, 180.0, 0.0, 180.0]
ORANGE_BIN_POSE = [0.3041, 0.73786, 0.086, 180.0, 0.0, 180.0]
BROWN_BIN_POSE = [0.3041, 0.62573, 0.086, 180.0, 0.0, 180.0]

BIN_POSE_MAP = {
    "yellow": YELLOW_BIN_POSE,
    "orange": ORANGE_BIN_POSE,
    "brown": BROWN_BIN_POSE,
}

# Grasp parameters
GRASP_Z = 0.008
PRE_GRASP_OFFSET_Z = 0.10  # above grasp height
PLACE_DROP_OFFSET_Z = 0.10  # above drop height

# Motion parameters
MOVE_SPEED = 1.0
MOVE_ACC = 1.2
JOINT_SPEED = 70.0
JOINT_ACC = 80.0

# Gripper parameters
GRIPPER_FORCE_N = 16.0       # RG2 max ~40N
GRIPPER_OPEN_WIDTH_MM = 60  # near-fully open
GRIPPER_CLOSE_FOR_GRASP = True  # use gripper.close

# Vision
DETECTION_PROMPT = "wrapped chocolate bars ."
QWEN_PROMPT = "Can you find all chocolate bars in this image and return every of them ."
DETECTION_BOX_THRESHOLD = 0.30
DETECTION_TEXT_THRESHOLD = 0.25
SAM_THRESHOLD = 0.5
PCA_AXIS_SAMPLE_PIXELS = 30  # pixel offset along principal axis for world-frame yaw

# Bin ROI: first locate the blue bin, then ignore anything outside its bbox.
BIN_DETECTION_PROMPT = "a blue bin ."
BIN_BOX_THRESHOLD = 0.30
BIN_TEXT_THRESHOLD = 0.25
BIN_ROI_SHRINK_PX = 0  # erode bin bbox slightly to avoid grabbing the rim itself
BIN_BBOX_OVERLAP_MIN = 0.95  # min fraction of chocolate bbox area that must lie inside bin ROI

# Color classification HSV ranges (OpenCV HSV: H in 0-179)
# Yellow ~ H 20-35, Orange ~ H 5-20, Brown ~ low S or low V or H 5-20 with low V/S
COLOR_HSV_RANGES = {
    "yellow": {"h": (20, 35), "s_min": 60, "v_min": 70},
    "orange": {"h": (8, 20), "s_min": 140, "v_min": 120},  # stricter — vivid only
    "brown": {"h": (0, 20), "s_min": 20, "v_min": 20, "v_max": 119},  # dark only
}

COLOR_PRIORITY = ["yellow", "orange", "brown"]

# Median-HSV prototypes — set from observed `HSV stats:` logs of each color.
# Brown calibrated from observed median (H=13, S=188, V=189) — wrapper photographs
# as a bright warm color, NOT the "dim" brown the per-pixel V_max range assumed.
# Yellow + orange are still placeholders — replace with your observed medians.
COLOR_PROTOTYPES_HSV = {
    # Calibrated from observed `HSV stats:` logs. Yellow's S is noisy across samples
    # (saw 224 and 159 → midpoint ~192), so its prototype is the midpoint, with H bumped
    # to 27 (between the two yellow H values of 25 and 30). Brown is uniquely identified
    # by low V (~189 vs ~253). Yellow vs orange is essentially a hue decision now.
    "yellow": (27, 192, 253),
    "orange": (22, 186, 255),
    "brown":  (13, 188, 189),
}
HUE_WEIGHT = 3.0          # hue dominates — it's the most stable channel across lighting
SV_WEIGHT = 1.0           # S+V mostly used to peel off brown via low V
COLOR_MASK_ERODE_ITER = 3 # erodes mask before sampling stats to drop rim/glare


# Loop control
MAX_SORT_ITERATIONS = 30
SETTLE_AFTER_MOVE_S = 0.3

# Camera intrinsics will be fetched at runtime from RealSense
# (fx, fy, cx, cy, distortion_coeffs)


# ============================================================
# Global state for SIGINT
# ============================================================
_interrupted = False


def _sigint_handler(signum, frame):
    global _interrupted
    logger.warning("SIGINT received - requesting graceful shutdown")
    _interrupted = True


# ============================================================
# Helpers
# ============================================================

def _cyclic_hue_diff(a: float, b: float) -> float:
    """Smallest angular distance between two OpenCV hues (0-179, wraps)."""
    d = abs(float(a) - float(b)) % 180.0
    return d if d <= 90.0 else 180.0 - d


def classify_color_from_mask(rgb_image: np.ndarray, mask: np.ndarray) -> str:
    if mask.dtype != bool:
        mask_bool = mask > 0
    else:
        mask_bool = mask
    if mask_bool.sum() < 50:
        logger.warning("Mask too small for reliable color classification")
        return "brown"

    # Eroded interior: drops rim pixels and glare hotspots that distort stats.
    mask_u8 = mask_bool.astype(np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=COLOR_MASK_ERODE_ITER)
    eroded_bool = eroded > 0
    if eroded_bool.sum() < 30:
        # Mask was too thin to erode; fall back to the raw mask interior.
        eroded_bool = mask_bool

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    pixels_inner = hsv[eroded_bool]
    n_full = int(mask_bool.sum())
    n_inner = int(eroded_bool.sum())

    # Diagnostics — observed medians let you calibrate COLOR_PROTOTYPES_HSV from data.
    med_h = float(np.median(pixels_inner[:, 0]))
    med_s = float(np.median(pixels_inner[:, 1]))
    med_v = float(np.median(pixels_inner[:, 2]))
    p25_s, p75_s = (float(x) for x in np.percentile(pixels_inner[:, 1], [25, 75]))
    p25_v, p75_v = (float(x) for x in np.percentile(pixels_inner[:, 2], [25, 75]))
    logger.info(
        f"HSV stats: n_full={n_full} n_inner={n_inner} "
        f"median=(H={med_h:.1f},S={med_s:.1f},V={med_v:.1f}) "
        f"S_p25/p75={p25_s:.0f}/{p75_s:.0f} V_p25/p75={p25_v:.0f}/{p75_v:.0f}"
    )

    # Classify by median HSV vs per-color prototype. Per-pixel range counting was
    # dropped because brown and orange overlap on S/V too much in this lighting —
    # only the median + per-color prototypes separate them reliably.
    dists = {}
    for color, (ph, ps, pv) in COLOR_PROTOTYPES_HSV.items():
        d_h = _cyclic_hue_diff(med_h, ph) * HUE_WEIGHT
        d_sv = (abs(med_s - ps) + abs(med_v - pv)) * SV_WEIGHT
        dists[color] = d_h + d_sv
    best = min(dists, key=dists.get)
    logger.info(
        f"Color decision: {best} (prototype dists={ {k: round(v, 1) for k, v in dists.items()} })"
    )
    return best


def get_camera_frame_context(camera, robot):
    """Fetch intrinsics + world_T_camera once per frame, reusable for many deprojections."""
    fx, fy, h, w, ppx, ppy, model, coeffs = camera.get_intrinsics("color")
    cam_intrinsics = np.array([[fx, 0.0, ppx],
                               [0.0, fy, ppy],
                               [0.0, 0.0, 1.0]], dtype=np.float32)
    dist = np.array(coeffs, dtype=np.float64)

    tcp_pose = robot.get_cartesian_pose()  # [x,y,z,rx,ry,rz] deg
    base_T_tcp = tfutils.pose_to_transformation_matrix(tcp_pose, rot_type='deg')
    base_T_camera = base_T_tcp @ TCP_T_CAMERA
    return cam_intrinsics, dist, base_T_camera


def deproject_pixel(pixel_xy, depth_image_m, cam_intrinsics, dist, world_T_camera):
    """Deproject a single pixel to a world_T_point 4x4 matrix. Returns (world_T_point, depth_val) or (None, None) on failure."""
    px, py = int(round(pixel_xy[0])), int(round(pixel_xy[1]))
    py = np.clip(py, 0, depth_image_m.shape[0] - 1)
    px = np.clip(px, 0, depth_image_m.shape[1] - 1)
    depth_val = float(depth_image_m[py, px])
    if depth_val <= 0.0 or not np.isfinite(depth_val):
        y0, y1 = max(0, py - 3), min(depth_image_m.shape[0], py + 4)
        x0, x1 = max(0, px - 3), min(depth_image_m.shape[1], px + 4)
        patch = depth_image_m[y0:y1, x0:x1]
        valid = patch[(patch > 0) & np.isfinite(patch)]
        if valid.size == 0:
            return None, None
        depth_val = float(np.median(valid))

    world_T_point_mat = pupil.project_pixel_to_world_point(
        camera_intrinsics=cam_intrinsics,
        distortion_coefficients=dist,
        pixel=[float(px), float(py)],
        depth=depth_val,
        world_T_camera=world_T_camera,
    )
    world_T_point = world_T_point_mat.to_numpy() if hasattr(
        world_T_point_mat, "to_numpy") else np.asarray(world_T_point_mat)
    return world_T_point, depth_val


def compute_yaw_world_from_axis(centroid_uv, principal_axis_uv, depth_image_m,
                                cam_intrinsics, dist, world_T_camera, k_pixels):
    """Sample two pixels along the pixel-space principal axis, deproject both to world,
    and return yaw (degrees) computed in the world XY plane. Returns None on failure."""
    vx = float(principal_axis_uv[0])
    vy = float(principal_axis_uv[1])
    p_plus = (centroid_uv[0] + k_pixels * vx, centroid_uv[1] + k_pixels * vy)
    p_minus = (centroid_uv[0] - k_pixels * vx, centroid_uv[1] - k_pixels * vy)
    w_plus, _ = deproject_pixel(p_plus, depth_image_m, cam_intrinsics, dist, world_T_camera)
    w_minus, _ = deproject_pixel(p_minus, depth_image_m, cam_intrinsics, dist, world_T_camera)
    if w_plus is None or w_minus is None:
        return None
    dx = float(w_plus[0, 3] - w_minus[0, 3])
    dy = float(w_plus[1, 3] - w_minus[1, 3])
    return float(np.degrees(np.arctan2(dy, dx)))


def visualize_detections(rgb, bboxes_xywh, labels, color_labels=None):
    # rerun bounding boxes; do NOT draw contours
    if len(bboxes_xywh) == 0:
        return
    centers = []
    half_sizes = []
    for b in bboxes_xywh:
        x, y, w, h = b
        centers.append([x + w / 2.0, y + h / 2.0])
        half_sizes.append([w / 2.0, h / 2.0])
    rr.log("camera/rgb/detections",
           rr.Boxes2D(centers=np.array(centers, dtype=np.float32),
                      half_sizes=np.array(half_sizes, dtype=np.float32),
                      labels=labels))


# ============================================================
# Main
# ============================================================
def main():
    global _interrupted

    logger.add(sys.stderr, level="INFO")
    logger.info("Initializing Rerun")
    rr.init("chocolate_sort", spawn=True)

    signal.signal(signal.SIGINT, _sigint_handler)

    robot = UniversalRobotsUR10E()
    gripper = OnRobotRG2()
    camera = RealSense(name=CAMERA_NAME)

    robot_connected = False
    gripper_connected = False
    camera_connected = False

    try:
        # --- Connect hardware ---
        logger.info(f"Connecting to UR10E at {ROBOT_IP}")
        robot.connect(ip=ROBOT_IP)
        robot_connected = True
        robot.set_tcp(TCP_OFFSET)
        logger.info("Robot connected and TCP set")

        logger.info(f"Connecting to OnRobot RG2 at {GRIPPER_IP}")
        gripper.connect(ip=GRIPPER_IP)
        gripper_connected = True
        gripper.set_force(GRIPPER_FORCE_N)
        logger.info("Gripper connected")

        logger.info("Connecting to RealSense camera")
        camera.connect(warmup_frames=30)
        camera_connected = True
        logger.info("Camera connected")

        # --- Go HOME ---
        rr.log("robot/state", rr.TextLog("home", level=rr.TextLogLevel.INFO))
        logger.info("Moving to HOME joint positions")
        # robot.set_joint_positions(HOME_JOINT_POSITIONS, speed=JOINT_SPEED, acceleration=JOINT_ACC)
        gripper.open(force=GRIPPER_FORCE_N)
        time.sleep(SETTLE_AFTER_MOVE_S)

        iteration = 0
        while iteration < MAX_SORT_ITERATIONS and not _interrupted:
            iteration += 1
            logger.info(f"========== Sort iteration {iteration} ==========")

            rr.log("robot/state", rr.TextLog("scanning", level=rr.TextLogLevel.INFO))

            # --- Move to scan pose ---
            logger.info(f"Moving to SCAN_POSE: {SCAN_POSE}")
            robot.set_cartesian_pose(SCAN_POSE, speed=MOVE_SPEED, acceleration=MOVE_ACC)
            time.sleep(SETTLE_AFTER_MOVE_S)

            # --- Capture RGB + Depth ---
            rgb = camera.capture_color_image()
            depth_m = camera.capture_depth_image()
            if rgb is None or depth_m is None:
                logger.error("Failed to capture RGB or depth frame")
                break
            logger.info(f"Captured RGB {rgb.shape} and depth {depth_m.shape}")
            rr.log("camera/rgb", rr.Image(rgb))
            rr.log("camera/depth", rr.DepthImage(depth_m, meter=1.0))

            # --- Locate the blue bin first: any detection outside this ROI is ignored ---
            rr.log("robot/state", rr.TextLog("locating bin", level=rr.TextLogLevel.INFO))
            logger.info(
                f"Running Grounding DINO with prompt='{BIN_DETECTION_PROMPT}' to find blue bin")
            try:

                bin_annotations, _ = retina.detect_objects_using_grounding_dino(
                    image=rgb,
                    prompt=BIN_DETECTION_PROMPT,
                    box_threshold=BIN_BOX_THRESHOLD,
                    text_threshold=BIN_TEXT_THRESHOLD,
                )
            except Exception as e:
                logger.exception(f"Bin detection failed: {e}")
                break

            bin_ann_list = bin_annotations.to_list() if hasattr(
                bin_annotations, "to_list") else list(bin_annotations)
            if len(bin_ann_list) == 0:
                logger.warning(
                    "Blue bin not detected; skipping this iteration (refuse to grasp outside bin)")
                break

            # Pick the highest-score bin detection
            best_bin = max(bin_ann_list, key=lambda a: float(a.get("score", 0.0)))
            bx, by, bw, bh = (float(best_bin["bbox"][0]), float(best_bin["bbox"][1]),
                              float(best_bin["bbox"][2]), float(best_bin["bbox"][3]))
            H_img, W_img = rgb.shape[:2]
            # Shrink slightly so we don't pick up the rim and clamp to image bounds.
            bin_x0 = int(max(0, round(bx + BIN_ROI_SHRINK_PX)))
            bin_y0 = int(max(0, round(by + BIN_ROI_SHRINK_PX)))
            bin_x1 = int(min(W_img, round(bx + bw - BIN_ROI_SHRINK_PX)))
            bin_y1 = int(min(H_img, round(by + bh - BIN_ROI_SHRINK_PX)))
            if bin_x1 <= bin_x0 or bin_y1 <= bin_y0:
                logger.warning(
                    f"Bin ROI degenerate after shrink: ({bin_x0},{bin_y0})-({bin_x1},{bin_y1}); skipping")
                break
            logger.info(f"Bin ROI (XYXY, shrunk by {BIN_ROI_SHRINK_PX}px) = "
                        f"({bin_x0},{bin_y0})-({bin_x1},{bin_y1}) "
                        f"score={float(best_bin.get('score', 0.0)):.2f}")
            rr.log("camera/rgb/bin_roi",
                   rr.Boxes2D(centers=[[(bin_x0 + bin_x1) / 2.0, (bin_y0 + bin_y1) / 2.0]],
                              half_sizes=[[(bin_x1 - bin_x0) / 2.0, (bin_y1 - bin_y0) / 2.0]],
                              labels=["blue_bin"]))

            # --- Detection: QWEN first, fall back to Grounding DINO (cup-sorting pattern) ---
            rr.log("robot/state", rr.TextLog("detecting", level=rr.TextLogLevel.INFO))
            ann_list_all = []
            detector_used = None

            # 1) QWEN
            # try:
            #     logger.info(f"Running QWEN with prompt='{QWEN_PROMPT}'")
            #     qwen_ann = retina.detect_objects_using_qwen(image=rgb, prompt=QWEN_PROMPT)
            #     qwen_list = qwen_ann.to_list() if hasattr(qwen_ann, "to_list") else list(qwen_ann)
            #     # Keep only items with a valid bbox
            #     qwen_list = [a for a in qwen_list if a.get("bbox") is not None]
            #     if len(qwen_list) > 0:
            #         ann_list_all = qwen_list
            #         detector_used = "qwen"
            #         logger.info(f"QWEN returned {len(ann_list_all)} chocolate detections")
            #     else:
            #         logger.warning("QWEN returned no detections; falling back to Grounding DINO")
            # except Exception as e:
            #     logger.warning(f"QWEN detection failed ({e}); falling back to Grounding DINO")

            # 2) Grounding DINO fallback
            if detector_used is None:
                logger.info(f"Running Grounding DINO with prompt='{DETECTION_PROMPT}'")
                try:
                    det_annotations, det_categories = retina.detect_objects_using_grounding_dino(
                        image=rgb,
                        prompt=DETECTION_PROMPT,
                        box_threshold=DETECTION_BOX_THRESHOLD,
                        text_threshold=DETECTION_TEXT_THRESHOLD,
                    )
                except Exception as e:
                    logger.exception(f"Grounding DINO detection failed: {e}")
                    break
                ann_list_all = det_annotations.to_list() if hasattr(
                    det_annotations, "to_list") else list(det_annotations)
                detector_used = "grounding_dino"
                logger.info(f"Grounding DINO returned {len(ann_list_all)} chocolate detections")

            logger.info(
                f"Detector used: {detector_used}, total raw detections: {len(ann_list_all)}")

            # Keep only detections whose centroid AND bbox lie inside the bin ROI.
            # Centroid-inside is the primary test; bbox-fully-inside is a secondary guard
            # so a chocolate poking over the bin rim cannot be picked.
            ann_list = []
            for a in ann_list_all:
                bb = a.get("bbox", [0, 0, 0, 0])
                ax, ay, aw, ah = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                cxp, cyp = ax + aw / 2.0, ay + ah / 2.0
                centroid_in = (bin_x0 <= cxp <= bin_x1) and (bin_y0 <= cyp <= bin_y1)
                # Fractional overlap = intersection / chocolate-bbox area. The strict
                # fully-inside test was failing when the detection bbox poked a few pixels
                # over the bin rim (very common); BIN_BBOX_OVERLAP_MIN (default 0.95) lets
                # those through while still rejecting things that are mostly outside.
                bbox_area = max(aw * ah, 1e-6)
                ix0 = max(ax, bin_x0)
                iy0 = max(ay, bin_y0)
                ix1 = min(ax + aw, bin_x1)
                iy1 = min(ay + ah, bin_y1)
                iw = max(0.0, ix1 - ix0)
                ih = max(0.0, iy1 - iy0)
                overlap_frac = (iw * ih) / bbox_area
                overlap_ok = overlap_frac >= BIN_BBOX_OVERLAP_MIN
                if centroid_in and overlap_ok:
                    ann_list.append(a)
                else:
                    logger.info(
                        f"  filtered detection outside bin ROI: bbox=({ax:.0f},{ay:.0f},{aw:.0f},{ah:.0f})"
                        f" centroid_in={centroid_in} overlap_frac={overlap_frac:.2f}"
                        f" (min={BIN_BBOX_OVERLAP_MIN:.2f})")
            logger.info(f"After bin-ROI filter: {len(ann_list)} chocolates inside bin "
                        f"(dropped {len(ann_list_all) - len(ann_list)})")

            if len(ann_list) == 0:
                logger.info("No chocolates inside blue bin. Stopping sort loop.")
                break

            # Extract bboxes XYWH for visualization and SAM (XYXY)
            bboxes_xywh = []
            bboxes_xyxy = []
            scores = []
            for a in ann_list:
                bbox = a.get("bbox", [0, 0, 0, 0])  # XYWH per ObjectDetectionAnnotation
                x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                bboxes_xywh.append([x, y, w, h])
                bboxes_xyxy.append([int(round(x)), int(round(y)),
                                    int(round(x + w)), int(round(y + h))])
                scores.append(float(a.get("score", 0.0)))

            visualize_detections(
                rgb, bboxes_xywh, labels=[
                    f"choc {i} s={scores[i]:.2f}" for i in range(
                        len(bboxes_xywh))])

            # --- Segmentation: SAM on the detected bboxes ---
            logger.info("Running SAM segmentation on detected bboxes")
            try:
                sam_annotations = cornea.segment_image_using_sam(
                    image=rgb,
                    bboxes=bboxes_xyxy,
                    mask_threshold=SAM_THRESHOLD,
                    image_id=0,
                )
            except Exception as e:
                logger.exception(f"SAM segmentation failed: {e}")
                break

            sam_list = sam_annotations.to_list() if hasattr(
                sam_annotations, "to_list") else list(sam_annotations)
            logger.info(f"SAM returned {len(sam_list)} masks")
            if len(sam_list) == 0:
                logger.warning("SAM produced no masks; aborting iteration")
                break

            # Build per-detection processing: choose the first/best object to pick this iteration.
            # We will rank by detection score and pick top.
            order = sorted(range(len(sam_list)),
                           key=lambda i: -scores[i] if i < len(scores) else 0.0)

            picked_this_iteration = False
            for obj_idx in order:
                if _interrupted:
                    break
                ann = sam_list[obj_idx]
                seg = ann.get("segmentation", None)
                bbox_xywh = bboxes_xywh[obj_idx] if obj_idx < len(bboxes_xywh) else None

                # Build binary mask image (H,W) uint8
                # SAM annotation segmentation may be polygon list or RLE; we expect a numpy-friendly mask.
                # Fallback: if seg is a dict with "size"/"counts" (RLE) we cannot decode here.
                if isinstance(seg, np.ndarray):
                    mask = (seg > 0).astype(np.uint8) * 255
                elif isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (list, np.ndarray)):
                    # polygon list -> rasterize
                    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
                    for poly in seg:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                else:
                    # MISSING CAPABILITY: cannot decode RLE here without pycocotools.
                    # Fall back to bbox-shaped rectangle mask so the pipeline can proceed.
                    logger.warning(
                        f"Object {obj_idx}: unknown segmentation format; falling back to bbox mask")
                    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
                    if bbox_xywh is not None:
                        x, y, w, h = [int(round(v)) for v in bbox_xywh]
                        mask[max(0, y):y + h, max(0, x):x + w] = 255

                # Clip mask to bin ROI — SAM occasionally bleeds outside the prompted bbox,
                # and we want a hard guarantee that nothing downstream (centroid, PCA, deproject)
                # references a pixel outside the bin.
                bin_roi_mask = np.zeros_like(mask)
                bin_roi_mask[bin_y0:bin_y1, bin_x0:bin_x1] = 255
                pre_clip_px = int((mask > 0).sum())
                mask = cv2.bitwise_and(mask, bin_roi_mask)
                post_clip_px = int((mask > 0).sum())
                if post_clip_px < pre_clip_px:
                    logger.info(
                        f"Object {obj_idx}: mask clipped to bin ROI ({pre_clip_px} -> {post_clip_px} px)")

                if mask.sum() == 0:
                    logger.warning(f"Object {obj_idx}: empty mask after bin-ROI clip, skipping")
                    continue

                # Visualize this mask
                rr.log(f"camera/rgb/mask_{obj_idx}",
                       rr.SegmentationImage(mask // 255))

                # --- Color classification (deterministic) ---
                color_label = classify_color_from_mask(rgb, mask)
                logger.info(f"Object {obj_idx}: classified color = {color_label}")
                if color_label not in BIN_POSE_MAP:
                    logger.warning(f"Object {obj_idx}: unrecognized color, skipping")
                    continue

                # Annotate bbox with color label
                if bbox_xywh is not None:
                    rr.log(f"camera/rgb/labeled_{obj_idx}",
                           rr.Boxes2D(centers=[[bbox_xywh[0] + bbox_xywh[2] / 2.0,
                                                bbox_xywh[1] + bbox_xywh[3] / 2.0]],
                                      half_sizes=[[bbox_xywh[2] / 2.0, bbox_xywh[3] / 2.0]],
                                      labels=[color_label]))

                # --- PCA on the mask for grasp orientation + centroid ---
                try:
                    centroid_dt, eigenvectors_dt, eigenvalues_dt, principal_angle_dt = pupil.calculate_image_pca(
                        image=mask)
                except Exception as e:
                    logger.exception(f"Object {obj_idx}: PCA failed: {e}")
                    continue

                # Extract numpy values (pixel-space centroid + principal eigenvector).
                # `pupil.calculate_image_pca` returns eigenvectors as a 2x2 matrix where each
                # eigenvector is a COLUMN (np.linalg.eig convention), sorted so the principal
                # axis is column 0. Earlier we flattened with reshape(-1) and took ev[:2],
                # which mixes principal_x with secondary_x — that's why the overlay was off.
                centroid_np = centroid_dt.to_numpy() if hasattr(centroid_dt, "to_numpy") else np.asarray(centroid_dt)
                eigvec_np = eigenvectors_dt.to_numpy() if hasattr(
                    eigenvectors_dt, "to_numpy") else np.asarray(eigenvectors_dt)
                cx_px, cy_px = float(centroid_np[0]), float(centroid_np[1])
                eigvec_mat = np.asarray(eigvec_np).reshape(2, 2)
                principal = eigvec_mat[:, 0]
                norm = float(np.linalg.norm(principal))
                if norm < 1e-9:
                    logger.warning(f"Object {obj_idx}: degenerate PCA eigenvector, skipping")
                    continue
                principal_axis_uv = (float(principal[0] / norm), float(principal[1] / norm))

                # Visualize centroid + principal axis as line overlay (pixel space)
                axis_len = 40.0
                p0 = [
                    cx_px -
                    principal_axis_uv[0] *
                    axis_len,
                    cy_px -
                    principal_axis_uv[1] *
                    axis_len]
                p1 = [
                    cx_px +
                    principal_axis_uv[0] *
                    axis_len,
                    cy_px +
                    principal_axis_uv[1] *
                    axis_len]
                rr.log(f"camera/rgb/pca_axis_{obj_idx}",
                       rr.LineStrips2D([[p0, p1]]))
                rr.log(f"camera/rgb/centroid_{obj_idx}",
                       rr.Points2D([[cx_px, cy_px]], radii=[4.0]))

                # Build per-frame camera context, then deproject centroid to world.
                try:
                    cam_intrinsics, dist, world_T_camera = get_camera_frame_context(camera, robot)
                    world_T_point, depth_used = deproject_pixel(
                        (cx_px, cy_px), depth_m, cam_intrinsics, dist, world_T_camera
                    )
                    if world_T_point is None:
                        raise RuntimeError(f"No valid depth at centroid ({cx_px:.1f},{cy_px:.1f})")
                except Exception as e:
                    logger.exception(f"Object {obj_idx}: pixel->world failed: {e}")
                    continue

                # Previous bug: yaw was computed in pixel space (atan2 over the eigenvector
                # components in image coords) and used directly as a world-frame Rz. That
                # silently assumes image axes == base axes, which is false here (CAMERA_IN_TCP
                # rz≈90.36°, plus TCP rz, plus image-v points down) — eigenvector overlay
                # could look right while world yaw was wrong. Sample two pixels along the
                # principal axis, deproject both to world, take atan2 there instead.
                grasp_yaw_deg = compute_yaw_world_from_axis(
                    (cx_px, cy_px), principal_axis_uv, depth_m,
                    cam_intrinsics, dist, world_T_camera, PCA_AXIS_SAMPLE_PIXELS,
                )
                if grasp_yaw_deg is None:
                    logger.warning(
                        f"Object {obj_idx}: could not deproject axis samples; skipping to avoid wrong-yaw pick")
                    continue

                obj_xyz = world_T_point[:3, 3]
                logger.info(
                    f"Object {obj_idx}: world XYZ = {obj_xyz.tolist()} (depth={depth_used:.3f}m) "
                    f"world_yaw_deg={grasp_yaw_deg:.2f}")

                # Top-down (rx=180, ry=0). On this RG2 + TCP convention, the gripper closes
                # along the TCP y-axis, so setting rz == long-axis yaw makes the jaws straddle
                # the long axis (perpendicular closure direction, which is what we want).
                # Adding +90 would instead align jaws *with* the long axis (wrong).
                # Wrap to [0, 180) since the parallel gripper is symmetric.
                base_rx, base_ry = 180.0, 0.0
                grasp_rz = grasp_yaw_deg % 180.0
                pre_grasp_pose = [float(obj_xyz[0]), float(obj_xyz[1]),
                                  GRASP_Z + PRE_GRASP_OFFSET_Z,
                                  base_rx, base_ry, grasp_rz]
                grasp_pose = [float(obj_xyz[0]), float(obj_xyz[1]),
                              GRASP_Z,
                              base_rx, base_ry, grasp_rz]

                logger.info(f"Object {obj_idx}: pre_grasp={pre_grasp_pose} grasp={grasp_pose}")

                # Visualize the 3D grasp pose as a transform in Rerun
                T_world = tfutils.pose_to_transformation_matrix(grasp_pose, rot_type='deg')
                rr.log("world/grasp_frame",
                       rr.Transform3D(translation=T_world[:3, 3],
                                      mat3x3=T_world[:3, :3]))

                # --- Pick (no move_until_contact for picking) ---
                rr.log("robot/state", rr.TextLog(f"grasping object {obj_idx} ({color_label})",
                                                 level=rr.TextLogLevel.INFO))
                logger.info(f"Opening gripper before pick")
                gripper.open(force=GRIPPER_FORCE_N)

                logger.info(f"Moving to pre-grasp")
                robot.set_cartesian_pose(pre_grasp_pose, speed=MOVE_SPEED, acceleration=MOVE_ACC)
                time.sleep(SETTLE_AFTER_MOVE_S)

                logger.info(f"Descending to grasp height z={GRASP_Z}")
                robot.set_cartesian_pose(
                    grasp_pose,
                    speed=MOVE_SPEED * 0.5,
                    acceleration=MOVE_ACC * 0.5)
                time.sleep(SETTLE_AFTER_MOVE_S)

                logger.info("Closing gripper to grasp")
                grip_status = gripper.close(force=GRIPPER_FORCE_N)
                logger.info(f"Gripper close status: {grip_status}")

                logger.info("Lifting back to pre-grasp")
                robot.set_cartesian_pose(pre_grasp_pose, speed=MOVE_SPEED, acceleration=MOVE_ACC)
                time.sleep(SETTLE_AFTER_MOVE_S)

                # --- Move to intermediate then to drop bin ---
                rr.log("robot/state", rr.TextLog(f"placing in {color_label} bin",
                                                 level=rr.TextLogLevel.INFO))
                logger.info("Moving to INTERMEDIATE joint positions before place")
                robot.set_joint_positions(
                    INTERMEDIATE_JOINT_POSITIONS,
                    speed=JOINT_SPEED,
                    acceleration=JOINT_ACC)
                time.sleep(SETTLE_AFTER_MOVE_S)

                bin_pose = list(BIN_POSE_MAP[color_label])
                pre_drop_pose = list(bin_pose)
                pre_drop_pose[2] = bin_pose[2] + PLACE_DROP_OFFSET_Z
                logger.info(f"Moving above {color_label} bin: {pre_drop_pose}")
                robot.set_cartesian_pose(pre_drop_pose, speed=MOVE_SPEED, acceleration=MOVE_ACC)
                time.sleep(SETTLE_AFTER_MOVE_S)

                # Use move_until_contact when placing (allowed per preferences) to find
                # bin surface gently
                try:
                    logger.info("move_until_contact descending into bin (placing)")
                    robot.move_until_contact(
                        cartesian_speed=[0.0, 0.0, -0.05, 0.0, 0.0, 0.0],
                        direction=[0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                        acceleration=0.5,
                    )
                except Exception as e:
                    logger.warning(f"move_until_contact failed, fallback to direct pose: {e}")
                    robot.set_cartesian_pose(
                        bin_pose, speed=MOVE_SPEED * 0.5, acceleration=MOVE_ACC * 0.5)
                time.sleep(SETTLE_AFTER_MOVE_S)

                logger.info("Opening gripper to release")
                gripper.move(GRIPPER_OPEN_WIDTH_MM, force=GRIPPER_FORCE_N)
                time.sleep(SETTLE_AFTER_MOVE_S)

                logger.info("Retreating above bin")
                robot.set_cartesian_pose(pre_drop_pose, speed=MOVE_SPEED, acceleration=MOVE_ACC)
                time.sleep(SETTLE_AFTER_MOVE_S)

                # Back to intermediate then re-scan
                robot.set_joint_positions(
                    INTERMEDIATE_JOINT_POSITIONS,
                    speed=JOINT_SPEED,
                    acceleration=JOINT_ACC)
                time.sleep(SETTLE_AFTER_MOVE_S)

                picked_this_iteration = True
                break  # re-scan after each pick (scene changes)

            if not picked_this_iteration:
                logger.info("No object was successfully picked this iteration; stopping loop")
                break

        logger.info("Sort loop complete. Returning HOME.")
        rr.log("robot/state", rr.TextLog("home", level=rr.TextLogLevel.INFO))
        # robot.set_joint_positions(HOME_JOINT_POSITIONS, speed=JOINT_SPEED, acceleration=JOINT_ACC)

    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
    finally:
        # Reset SIGINT immediately so a second Ctrl+C terminates cleanly
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # Disconnect everything; catch BaseException so KeyboardInterrupt during cleanup
        # does not skip remaining teardown steps.
        if gripper_connected:
            try:
                logger.info("Disconnecting gripper")
                gripper.disconnect()
            except BaseException as e:
                logger.warning(f"Gripper disconnect error: {e!r}")

        if robot_connected:
            try:
                logger.info("Disconnecting robot")
                robot.disconnect()
            except BaseException as e:
                logger.warning(f"Robot disconnect error: {e!r}")

        if camera_connected:
            try:
                logger.info("Disconnecting camera")
                camera.disconnect()
            except BaseException as e:
                logger.warning(f"Camera disconnect error: {e!r}")

        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
