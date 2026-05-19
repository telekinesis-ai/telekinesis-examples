# Pipeline: Sort 3 types of screws from a blue bin into 3 separate boxes
# Hardware: UR10E + OnRobot RG2 + Intel RealSense
# Vision: Detect bin -> detect screws inside bin -> segment -> classify
# (color + size) -> PCA -> grasp -> place

import os
import signal
import sys
import time
import numpy as np
import cv2
from loguru import logger

try:
    import rerun as rr
    _HAS_RERUN = True
except ImportError:
    _HAS_RERUN = False
    logger.warning("rerun not available, visualizations disabled")

# Telekinesis SDK
from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E
from telekinesis.synapse.tools.parallel_grippers.onrobot import OnRobotRG2
from telekinesis.synapse.tools.parallel_grippers import abstract_parallel_gripper
from telekinesis.medulla.cameras.realsense import RealSense
from telekinesis.synapse import utils as tfutils

# Skill libraries
from telekinesis import retina, cornea, pupil
from datatypes import datatypes

# ============================================================================
# Signal handling — reliable Ctrl+C even when robot SDK blocks in C extensions
# ============================================================================
_stop_requested = False
_sigint_count = 0


def _sigint_handler(sig, frame):
    global _stop_requested, _sigint_count
    _sigint_count += 1
    _stop_requested = True
    if _sigint_count >= 2:
        logger.warning("Second Ctrl+C: force-exiting")
        os._exit(1)
    logger.warning(
        "Ctrl+C received — stopping after current robot operation. Press again to force quit.")


# ============================================================================
# HARDCODED / TUNABLE CONSTANTS (top of file per user preference)
# ============================================================================
ROBOT_IP = "192.168.2.2"
GRIPPER_IP = "192.168.1.1"

TCP_OFFSET_M = 0.30  # 23 cm from flange to TCP tip

# Hand-eye calibration: camera pose in TCP frame (deg)
CAMERA_IN_TCP = [
    0.07520960896570618,
    -0.0352478269641629,
    -0.2162654145229983,
    -0.07505179364087063,
    0.8826477579985493,
    90.3598403373567,
]
TCP_T_CAMERA = tfutils.pose_to_transformation_matrix(CAMERA_IN_TCP, rot_type="deg")

# Robot poses (Cartesian xyzrxryrz [m,deg])
SCAN_POSE = [-0.25462, 0.59302, 0.1041, 180.0, 0.0, 90.0]

# Joint poses [deg]
HOME_JOINT_POSITIONS = [120, -90, -90, -90, 90, -90]
INTERMEDIATE_JOINT_POSITIONS = [120, -70, -120, -80, 90, -90]

# Box drop poses for the 3 screw classes
BOX_POSES = [
    [0.3041, 0.85220, 0.086, 180.0, 0.0, 180.0],  # class 0
    [0.3041, 0.73786, 0.086, 180.0, 0.0, 180.0],  # class 1
    [0.3041, 0.62573, 0.086, 180.0, 0.0, 180.0],  # class 2
]

# Motion offsets / heights (m)
APPROACH_HEIGHT_M = 0.10          # height above grasp z for pre-pick approach
# small Z offset above mask surface for grasping (gripper tip just above object)
PICK_Z_OFFSET_M = 0.005
GRASP_Z_OFFSET_M = 0.011
LIFT_AFTER_PICK_M = 0.15          # how much to lift after grasp
PLACE_APPROACH_HEIGHT_M = 0.10    # height above box before lowering to drop
PLACE_LOWER_M = 0.05              # how far to lower into the box for drop

# Motion params
JOINT_SPEED_DEG_S = 150.0
JOINT_ACC_DEG_S2 = 120
CART_SPEED_M_S = 1
CART_ACC_M_S2 = 1

# Gripper params
GRIPPER_FORCE_N = 15.0  # RG2 max grip force is 40 N

# Vision params
SCREW_DETECTION_SCORE_THRESHOLD = 0.25
SCREW_DETECTION_TEXT_THRESHOLD = 0.25
BIN_DETECTION_SCORE_THRESHOLD = 0.30
SCREW_BBOX_MAX_AREA_PX = 20000  # drop detections larger than this (w*h); catches bin/tray false positives

# Classification thresholds
# Black vs gray on mask luminance (HSV Value channel mean, 0-255). Low V => black.
BLACK_V_THRESHOLD = 100
# Minimum PCA major axis length (pixels) for a gray screw to be considered "big".
# Tune this after observing axis_len= values in the logs for both screw types.
GRAY_BIG_AXIS_LEN_PX_MIN = 120
# Class indexing for box assignment:
CLASS_BLACK = 0
CLASS_GRAY_BIG = 1
CLASS_GRAY_SMALL = 2

# Safety
MAX_PICK_ITERATIONS = 100   # avoid infinite loops
SCAN_SETTLE_S = 0.4
GRIPPER_OPEN_MM = 70
# Camera intrinsics will be populated at runtime via camera.get_intrinsics("color")
# placeholders here; actual values filled at runtime.
CAMERA_INTRINSICS_MAT = None  # 3x3
CAMERA_DISTORTION = None      # list of coeffs


# ============================================================================
# Helpers
# ============================================================================

def _log_rr_image(entity: str, image_np):
    if not _HAS_RERUN:
        return
    try:
        rr.log(entity, rr.Image(image_np))
    except BaseException as e:
        logger.warning(f"rerun log image failed for {entity}: {e}")


def _log_rr_bbox(entity: str, bbox_xywh, label: str = ""):
    """Log a 2D bbox in XYWH to rerun."""
    if not _HAS_RERUN:
        return
    try:
        x, y, w, h = bbox_xywh
        rr.log(
            entity,
            rr.Boxes2D(array=[[x, y, w, h]], array_format=rr.Box2DFormat.XYWH, labels=[label]),
        )
    except BaseException as e:
        logger.warning(f"rerun log bbox failed for {entity}: {e}")


def _log_rr_mask(entity: str, mask_np):
    if not _HAS_RERUN:
        return
    try:
        # convert binary/labeled mask to a viewable image
        vis = (mask_np > 0).astype(np.uint8) * 255
        rr.log(entity, rr.SegmentationImage(vis))
    except BaseException as e:
        logger.warning(f"rerun log mask failed for {entity}: {e}")


def _log_rr_points2d(entity: str, points_xy, labels=None):
    if not _HAS_RERUN:
        return
    try:
        rr.log(entity, rr.Points2D(points_xy, labels=labels))
    except BaseException as e:
        logger.warning(f"rerun log points2d failed for {entity}: {e}")


def _log_rr_text(entity: str, text: str):
    if not _HAS_RERUN:
        return
    try:
        rr.log(entity, rr.TextLog(text))
    except BaseException as e:
        logger.warning(f"rerun log text failed for {entity}: {e}")


def detect_bin(rgb_image):
    """Detect the blue bin. Try Grounding DINO first, fallback to Qwen, then to HSV blue filter.
    Returns bbox in XYXY pixel coords (x0,y0,x1,y1) or None.
    """
    logger.info("Detecting blue bin...")
    # Try Grounding DINO
    try:
        anns, _cats = retina.detect_objects_using_grounding_dino(
            image=rgb_image,
            prompt="a blue bin .",
            box_threshold=BIN_DETECTION_SCORE_THRESHOLD,
            text_threshold=BIN_DETECTION_SCORE_THRESHOLD,
        )
        ann_list = anns.to_list()
        if len(ann_list) > 0:
            # take highest-score (first) result
            best = max(ann_list, key=lambda a: a.get("score")
                       if a.get("score") is not None else 0.0)
            x, y, w, h = best["bbox"]
            logger.success(f"Bin detected via Grounding DINO: bbox=({x},{y},{w},{h})")
            return [int(x), int(y), int(x + w), int(y + h)]
    except BaseException as e:
        logger.warning(f"Grounding DINO bin detection failed: {e}")

    # Fallback: Qwen
    try:
        anns = retina.detect_objects_using_qwen(
            image=rgb_image,
            prompt="blue bin",
        )
        ann_list = anns.to_list()
        if len(ann_list) > 0:
            best = ann_list[0]
            x, y, w, h = best["bbox"]
            logger.success(f"Bin detected via Qwen: bbox=({x},{y},{w},{h})")
            return [int(x), int(y), int(x + w), int(y + h)]
    except BaseException as e:
        logger.warning(f"Qwen bin detection failed: {e}")

    # Fallback: HSV blue color filter — find largest blue contour
    logger.warning("Falling back to HSV blue color filter for bin")
    try:
        ann = cornea.segment_image_using_hsv(
            image=rgb_image,
            lower_bound=(100, 80, 40),
            upper_bound=(135, 255, 255),
        )
        ann_dict = ann.to_dict()
        labeled_mask = ann_dict["labeled_mask"]
        if hasattr(labeled_mask, "to_numpy"):
            mask = labeled_mask.to_numpy()
        else:
            mask = np.array(labeled_mask)
        mask_bin = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            logger.error("No blue regions found in HSV fallback")
            return None
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        logger.success(f"Bin detected via HSV fallback: bbox=({x},{y},{w},{h})")
        return [int(x), int(y), int(x + w), int(y + h)]
    except BaseException as e:
        logger.error(f"HSV fallback also failed: {e}")
        return None


def detect_screws_in_bin(rgb_image, bin_bbox_xyxy):
    """Detect screws strictly inside the bin patch. Try Grounding DINO then Qwen.
    Returns list of bboxes in image-frame XYXY (translated back).
    """
    x0, y0, x1, y1 = bin_bbox_xyxy
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(rgb_image.shape[1], x1)
    y1 = min(rgb_image.shape[0], y1)
    patch = rgb_image[y0:y1, x0:x1].copy()
    logger.info(f"Detecting screws inside bin patch shape={patch.shape}")
    _log_rr_image("bin_patch/rgb", patch)

    detections_xyxy_full = []

    # Try Grounding DINO first
    try:
        anns, _cats = retina.detect_objects_using_grounding_dino(
            image=patch,
            prompt="screws .",
            box_threshold=SCREW_DETECTION_SCORE_THRESHOLD,
            text_threshold=SCREW_DETECTION_TEXT_THRESHOLD,
        )
        ann_list = anns.to_list()
        logger.info(f"Grounding DINO found {len(ann_list)} screw candidates")
        for a in ann_list:
            bx, by, bw, bh = a["bbox"]
            full = [int(bx + x0), int(by + y0), int(bx + bw + x0), int(by + bh + y0)]
            detections_xyxy_full.append(full)
    except BaseException as e:
        logger.warning(f"Grounding DINO screw detection failed: {e}")

    # If nothing found, fallback to Qwen
    if len(detections_xyxy_full) == 0:
        try:
            anns = retina.detect_objects_using_qwen(
                image=patch,
                prompt="Find all screws on gray tray .",
            )
            ann_list = anns.to_list()
            logger.info(f"Qwen found {len(ann_list)} screw candidates")
            for a in ann_list:
                bx, by, bw, bh = a["bbox"]
                full = [int(bx + x0), int(by + y0), int(bx + bw + x0), int(by + bh + y0)]
                detections_xyxy_full.append(full)
        except BaseException as e:
            logger.warning(f"Qwen screw detection failed: {e}")

    # Filter detections strictly inside bin (already in image coords; clip)
    filtered = []
    for d in detections_xyxy_full:
        dx0, dy0, dx1, dy1 = d
        # require center inside bin
        cx = 0.5 * (dx0 + dx1)
        cy = 0.5 * (dy0 + dy1)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            filtered.append(d)
    logger.info(f"Filtered to {len(filtered)} screws strictly inside bin patch")
    return filtered


def segment_screw(rgb_image, bbox_xyxy):
    """Run SAM to get a mask for a screw bbox. Returns binary mask (H,W) uint8."""
    try:
        # SAM expects XYXY boxes
        anns = cornea.segment_image_using_sam(
            image=rgb_image,
            bboxes=[bbox_xyxy],
            mask_threshold=0.5,
        )
        ann_list = anns.to_list() if hasattr(anns, "to_list") else list(anns.to_list())
        if len(ann_list) == 0:
            logger.warning("SAM returned no annotations, falling back to BiRefNet on cropped patch")
            return _segment_with_birefnet_crop(rgb_image, bbox_xyxy)
        # take first
        ann0 = ann_list[0]
        seg = ann0.get("segmentation", None)
        if seg is None:
            return _segment_with_birefnet_crop(rgb_image, bbox_xyxy)
        # SAM segmentation from this SDK is RLE-like or polygon. We'll just rasterize from polygon if list, otherwise reconstruct.
        # The simplest robust path: use the bbox to crop and rebuild a binary mask
        # the same size as the image using polygons or counts.
        H, W = rgb_image.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (list, np.ndarray)):
            # polygon list
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
        elif isinstance(seg, dict) and "counts" in seg and "size" in seg:
            # RLE - not bothering to decode here; fallback
            return _segment_with_birefnet_crop(rgb_image, bbox_xyxy)
        else:
            return _segment_with_birefnet_crop(rgb_image, bbox_xyxy)
        return mask
    except BaseException as e:
        logger.warning(f"SAM segmentation failed: {e}, falling back to BiRefNet")
        return _segment_with_birefnet_crop(rgb_image, bbox_xyxy)


def _segment_with_birefnet_crop(rgb_image, bbox_xyxy):
    """Fallback: BiRefNet foreground segmentation on the bbox crop, placed back into full image mask."""
    H, W = rgb_image.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)
    patch = rgb_image[y0:y1, x0:x1].copy()
    full_mask = np.zeros((H, W), dtype=np.uint8)
    try:
        ann = cornea.segment_image_using_foreground_birefnet(image=patch, mask_threshold=0)
        ann_dict = ann.to_dict()
        lm = ann_dict["labeled_mask"]
        local = lm.to_numpy() if hasattr(lm, "to_numpy") else np.array(lm)
        local_bin = (local > 0).astype(np.uint8) * 255
        full_mask[y0:y1, x0:x1] = local_bin
    except BaseException as e:
        logger.error(f"BiRefNet fallback failed: {e}")
    return full_mask


def classify_color_on_mask(rgb_image, mask):
    """Return 'black' or 'gray' based on mean HSV V value on the mask pixels."""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    mask_bool = mask > 0
    if mask_bool.sum() == 0:
        return "gray", 0.0
    mean_v = float(v[mask_bool].mean())
    label = "black" if mean_v < BLACK_V_THRESHOLD else "gray"
    return label, mean_v


def compute_pca_on_mask(mask):
    """Use pupil.calculate_image_pca. Returns (centroid_xy (np.array shape (2,)), principal_axis_xy (np.array shape (2,)), eigenvalues (np.array)).

    Note: pupil.calculate_image_pca returns eigenvectors as a 2x2 matrix with eigenvectors as COLUMNS
    (np.linalg.eig convention). The principal axis is eigvec_mat[:, 0].
    Centroid is already (x, y).
    """
    # mask should be binary uint8
    centroid_dt, eigvec_dt, eigval_dt, principal_angle_dt = pupil.calculate_image_pca(mask)
    # centroid is Position2D
    centroid = centroid_dt.to_numpy().reshape(-1)[:2]
    # eigvec is Mat2X2
    eigvec_mat = eigvec_dt.to_numpy()
    # principal axis = first column (per np.linalg.eig column convention)
    principal_axis = eigvec_mat[:, 0]
    # normalize for safety
    n = np.linalg.norm(principal_axis)
    if n > 1e-9:
        principal_axis = principal_axis / n
    eigvals = np.asarray(eigval_dt.to_numpy()).reshape(-1)
    return centroid, principal_axis, eigvals


def pca_major_axis_length_pixels(mask):
    """Approximate physical length of the mask along its principal axis in pixels.
    We use the bounding rectangle of the rotated min-area-rect (cv2.minAreaRect) longest side."""
    contours, _ = cv2.findContours((mask > 0).astype(
        np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0
    biggest = max(contours, key=cv2.contourArea)
    if len(biggest) < 5:
        # too small to fit
        x, y, w, h = cv2.boundingRect(biggest)
        return float(max(w, h))
    rect = cv2.minAreaRect(biggest)
    (cx, cy), (w, h), angle = rect
    return float(max(w, h))


def deproject_pixel_to_world(pixel_xy, depth_value_m, world_T_camera_mat):
    """Use pupil.project_pixel_to_world_point. Returns 4x4 world_T_point matrix as np.array."""
    global CAMERA_INTRINSICS_MAT, CAMERA_DISTORTION
    res = pupil.project_pixel_to_world_point(
        camera_intrinsics=CAMERA_INTRINSICS_MAT,
        distortion_coefficients=CAMERA_DISTORTION,
        pixel=list(pixel_xy),
        depth=float(depth_value_m),
        world_T_camera=world_T_camera_mat,
    )
    return res.to_numpy()


def get_world_T_camera(robot):
    """Compose world_T_camera = world_T_tcp * tcp_T_camera."""
    tcp_pose = robot.get_cartesian_pose()
    world_T_tcp = tfutils.pose_to_transformation_matrix(tcp_pose, rot_type="deg")
    return world_T_tcp @ TCP_T_CAMERA


def sample_depth_around(depth_image_m, px, py, win=3):
    """Sample a robust depth value around pixel (px, py)."""
    H, W = depth_image_m.shape[:2]
    px = int(np.clip(px, 0, W - 1))
    py = int(np.clip(py, 0, H - 1))
    x0 = max(0, px - win)
    x1 = min(W, px + win + 1)
    y0 = max(0, py - win)
    y1 = min(H, py + win + 1)
    patch = depth_image_m[y0:y1, x0:x1]
    valid = patch[(patch > 0.05) & (patch < 3.0) & np.isfinite(patch)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def compute_world_grasp_pose_for_screw(rgb_image, depth_image_m, mask, world_T_camera_mat):
    """Given screw mask and depth, compute grasp pose [x,y,z,rx,ry,rz] (m, deg)
    by:
      - PCA on mask -> centroid (px,py), principal axis dx,dy in image space
      - sample two pixels along the principal axis, deproject both to world
      - world_long_axis_yaw = atan2(world_dy, world_dx)
      - grasp_rz = world_long_axis_yaw (NO +90° offset; RG2 jaws close along TCP-y so jaws straddle long axis)
      - approach orientation kept top-down (rx=180, ry=0) — only Rz is computed from PCA
    Returns (grasp_pose_world_xyzrxryrz, pca_info_dict) or None on failure.
    """
    H, W = rgb_image.shape[:2]
    centroid_xy, principal_axis_xy, eigvals = compute_pca_on_mask(mask)
    px, py = float(centroid_xy[0]), float(centroid_xy[1])
    dx_img, dy_img = float(principal_axis_xy[0]), float(principal_axis_xy[1])

    # sample two pixels along the principal axis, symmetric around centroid
    # use length scaled to mask size
    L = pca_major_axis_length_pixels(mask)
    step = max(8.0, 0.3 * L)
    p1 = (px - step * dx_img, py - step * dy_img)
    p2 = (px + step * dx_img, py + step * dy_img)

    z_c = sample_depth_around(depth_image_m, px, py)
    z_1 = sample_depth_around(depth_image_m, p1[0], p1[1])
    z_2 = sample_depth_around(depth_image_m, p2[0], p2[1])
    if z_c is None:
        logger.warning("No valid depth at centroid; skipping")
        return None
    # fallback to centroid depth if endpoints invalid (long axis still meaningful for yaw)
    if z_1 is None:
        z_1 = z_c
    if z_2 is None:
        z_2 = z_c

    world_T_centroid = deproject_pixel_to_world((px, py), z_c, world_T_camera_mat)
    world_T_p1 = deproject_pixel_to_world(p1, z_1, world_T_camera_mat)
    world_T_p2 = deproject_pixel_to_world(p2, z_2, world_T_camera_mat)

    cw = world_T_centroid[:3, 3]
    p1w = world_T_p1[:3, 3]
    p2w = world_T_p2[:3, 3]

    world_dx = p2w[0] - p1w[0]
    world_dy = p2w[1] - p1w[1]
    world_long_axis_yaw_rad = np.arctan2(world_dy, world_dx)
    world_long_axis_yaw_deg = float(np.degrees(world_long_axis_yaw_rad))

    # Per user preference: gripper jaws close along TCP-y => grasp_rz = world_long_axis_yaw (NO +90)
    grasp_rz_deg = world_long_axis_yaw_deg

    # Top-down approach orientation (rx=180, ry=0)
    grasp_pose_world = [
        float(cw[0]),
        float(cw[1]),
        GRASP_Z_OFFSET_M,  # add small offset to avoid collisions with mask surface
        180.0,
        0.0,
        grasp_rz_deg,
    ]

    pca_info = {
        "centroid_px": (px, py),
        "principal_axis_img": (dx_img, dy_img),
        "axis_pt1_px": p1,
        "axis_pt2_px": p2,
        "world_centroid": cw.tolist(),
        "world_pt1": p1w.tolist(),
        "world_pt2": p2w.tolist(),
        "world_long_axis_yaw_deg": world_long_axis_yaw_deg,
        "eigvals": eigvals.tolist(),
        "mask_axis_length_px": L,
    }
    return grasp_pose_world, pca_info


def make_pca_overlay(rgb_image, centroid_px, p1_px, p2_px):
    """Return RGB image with PCA centroid (circle) and major axis (line) drawn for visualization."""
    vis = rgb_image.copy()
    c = (int(centroid_px[0]), int(centroid_px[1]))
    p1 = (int(p1_px[0]), int(p1_px[1]))
    p2 = (int(p2_px[0]), int(p2_px[1]))
    cv2.line(vis, p1, p2, (255, 0, 0), 2)
    cv2.circle(vis, c, 5, (0, 255, 0), -1)
    return vis


def pick_screw(robot, gripper, grasp_pose_world):
    """Pick sequence: approach above -> open -> descend -> close -> lift.
    No move_until_contact for picking (per user preference)."""
    above = list(grasp_pose_world)
    above[2] = grasp_pose_world[2] + APPROACH_HEIGHT_M
    descend = list(grasp_pose_world)
    descend[2] = grasp_pose_world[2]
    lift = list(grasp_pose_world)
    lift[2] = grasp_pose_world[2] + LIFT_AFTER_PICK_M

    logger.info(f"Pick approach above: {above}")
    robot.set_cartesian_pose(above, speed=CART_SPEED_M_S, acceleration=CART_ACC_M_S2)
    logger.info("Opening gripper before pick")
    gripper.move(45, force=GRIPPER_FORCE_N, asynchronous=False)
    logger.info(f"Descending to grasp: {descend}")
    robot.set_cartesian_pose(descend, speed=CART_SPEED_M_S, acceleration=CART_ACC_M_S2)
    logger.info("Closing gripper to grasp")
    gripper.move(28, force=GRIPPER_FORCE_N, asynchronous=False)
    logger.info(f"Lifting after pick to: {lift}")
    robot.set_cartesian_pose(lift, speed=CART_SPEED_M_S, acceleration=CART_ACC_M_S2)


def place_at_box(robot, gripper, box_pose):
    """Place sequence: through INTERMEDIATE -> above box -> descend (move_until_contact ok for placing) -> open -> lift -> back through INTERMEDIATE."""
    logger.info(f"Moving to INTERMEDIATE before place: {INTERMEDIATE_JOINT_POSITIONS}")
    robot.set_joint_positions(
        INTERMEDIATE_JOINT_POSITIONS,
        speed=JOINT_SPEED_DEG_S,
        acceleration=JOINT_ACC_DEG_S2,
    )

    above = list(box_pose)
    above[2] = box_pose[2] + PLACE_APPROACH_HEIGHT_M
    logger.info(f"Place approach above box: {above}")
    robot.set_cartesian_pose(above, speed=CART_SPEED_M_S, acceleration=CART_ACC_M_S2)

    # # Use move_until_contact for placing (per user preference)
    # try:
    #     logger.info("Place: moving until contact (downward)")
    #     cartesian_speed = [0.0, 0.0, -0.05, 0.0, 0.0, 0.0]   # 5 cm/s downward
    #     direction = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
    #     robot.move_until_contact(
    #         cartesian_speed=cartesian_speed,
    #         direction=direction,
    #         acceleration=CART_ACC_M_S2,
    #     )
    # except BaseException as e:
    #     logger.warning(
    #         f"move_until_contact failed during place ({e}); falling back to lowering by PLACE_LOWER_M")
    #     lower = list(box_pose)
    #     lower[2] = box_pose[2] + PLACE_LOWER_M
    #     robot.set_cartesian_pose(lower, speed=CART_SPEED_M_S, acceleration=CART_ACC_M_S2)

    logger.info("Opening gripper to release")
    gripper.move(GRIPPER_OPEN_MM, force=GRIPPER_FORCE_N, asynchronous=False)

    logger.info(f"Lifting back to above box: {above}")
    robot.set_cartesian_pose(above, speed=CART_SPEED_M_S, acceleration=CART_ACC_M_S2)

    logger.info(f"Returning through INTERMEDIATE: {INTERMEDIATE_JOINT_POSITIONS}")
    robot.set_joint_positions(
        INTERMEDIATE_JOINT_POSITIONS,
        speed=JOINT_SPEED_DEG_S,
        acceleration=JOINT_ACC_DEG_S2,
    )


# ============================================================================
# Pipeline
# ============================================================================

def classify_screws(rgb_image, screw_records):
    """Given list of dicts with 'mask' and 'mask_area' and 'axis_len_px', assign class label per record.

    Strategy:
      - For each record, run color classification (black vs gray) on mask.
      - Among 'gray' records in this scan, split by median PCA major axis length (pixels): bigger half => gray_big, smaller half => gray_small.
      - If only one gray screw is present we cannot split by median; default it to gray_big.
    """
    # Color first
    for rec in screw_records:
        color_label, mean_v = classify_color_on_mask(rgb_image, rec["mask"])
        rec["color_label"] = color_label
        rec["mean_v"] = mean_v

    grays = [r for r in screw_records if r["color_label"] == "gray"]
    for r in grays:
        if r["axis_len_px"] >= GRAY_BIG_AXIS_LEN_PX_MIN:
            r["class_idx"] = CLASS_GRAY_BIG
            r["class_name"] = "gray_big"
        else:
            r["class_idx"] = CLASS_GRAY_SMALL
            r["class_name"] = "gray_small"

    for r in screw_records:
        if r["color_label"] == "black":
            r["class_idx"] = CLASS_BLACK
            r["class_name"] = "black"

    return screw_records


def scan_and_choose_screw(robot, camera, iteration_idx, bin_bbox):
    """Capture RGB+depth at current pose, detect bin, detect screws inside, segment each, classify, choose one.

    Returns (chosen_record, world_T_camera_mat, rgb, depth, bin_bbox_xyxy) or None if no screws.
    """
    logger.info(f"[iter {iteration_idx}] Capturing RGB and depth at scan pose")
    time.sleep(SCAN_SETTLE_S)
    rgb = camera.capture_color_image()
    depth = camera.capture_depth_image()
    if rgb is None or depth is None:
        logger.error("Failed to capture RGB or depth")
        return None
    logger.info(f"RGB shape={rgb.shape}, depth shape={depth.shape}")

    _log_rr_image(f"scan/iter_{iteration_idx}/rgb", rgb)
    _log_rr_image(f"scan/iter_{iteration_idx}/depth", depth)

    # Detect bin

    if bin_bbox is None:
        logger.error("Bin not detected; aborting this iteration")
        return None
    bx0, by0, bx1, by1 = bin_bbox
    _log_rr_bbox(
        f"scan/iter_{iteration_idx}/rgb/bin_bbox",
        [bx0, by0, bx1 - bx0, by1 - by0],
        label="blue_bin",
    )

    # Detect screws inside bin
    screw_bboxes_xyxy = detect_screws_in_bin(rgb, bin_bbox)
    if len(screw_bboxes_xyxy) == 0:
        logger.info("No screws detected inside bin")
        return None

    # Drop oversized bboxes (e.g. the bin/tray detected as a screw)
    before = len(screw_bboxes_xyxy)
    screw_bboxes_xyxy = [
        d for d in screw_bboxes_xyxy
        if (d[2] - d[0]) * (d[3] - d[1]) <= SCREW_BBOX_MAX_AREA_PX
    ]
    if len(screw_bboxes_xyxy) < before:
        logger.info(f"Dropped {before - len(screw_bboxes_xyxy)} oversized bbox(es) (>{SCREW_BBOX_MAX_AREA_PX} px²)")
    if len(screw_bboxes_xyxy) == 0:
        logger.info("No screws remain after size filter")
        return None

    # Visualize raw detection bboxes
    if _HAS_RERUN:
        try:
            arr = np.array(
                [[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in screw_bboxes_xyxy],
                dtype=np.float32,
            )
            rr.log(
                f"scan/iter_{iteration_idx}/rgb/screw_bboxes_raw",
                rr.Boxes2D(array=arr, array_format=rr.Box2DFormat.XYWH),
            )
        except BaseException as e:
            logger.warning(f"rerun bbox log failed: {e}")

    # Segment one screw at a time; attempt grasp immediately after each — return on first success.
    # classify_screws is called on records accumulated so far so gray median split uses whatever
    # gray screws have been seen up to this point.
    world_T_camera_mat = get_world_T_camera(robot)
    records = []
    for i, bbox in enumerate(screw_bboxes_xyxy):
        logger.info(f"Segmenting screw {i} bbox={bbox}")
        mask = segment_screw(rgb, bbox)
        if mask is None or mask.sum() == 0:
            logger.warning(f"Screw {i}: empty mask, skipping")
            continue
        _log_rr_mask(f"scan/iter_{iteration_idx}/screw_{i}/mask", mask)

        area = int((mask > 0).sum())
        axis_len = pca_major_axis_length_pixels(mask)
        rec = {
            "idx": i,
            "bbox_xyxy": bbox,
            "mask": mask,
            "mask_area_px": area,
            "axis_len_px": axis_len,
        }
        records.append(rec)

        out = compute_world_grasp_pose_for_screw(rgb, depth, mask, world_T_camera_mat)
        if out is None:
            continue

        grasp_pose, pca_info = out
        rec["grasp_pose_world"] = grasp_pose
        rec["pca_info"] = pca_info

        # Classify using only records seen so far (includes this one)
        classify_screws(rgb, records)

        # Visualize classified bboxes for records seen so far
        if _HAS_RERUN:
            try:
                arr = np.array(
                    [[r["bbox_xyxy"][0], r["bbox_xyxy"][1],
                      r["bbox_xyxy"][2] - r["bbox_xyxy"][0],
                      r["bbox_xyxy"][3] - r["bbox_xyxy"][1]] for r in records],
                    dtype=np.float32,
                )
                labels = [r.get("class_name", "?") for r in records]
                rr.log(
                    f"scan/iter_{iteration_idx}/rgb/screw_bboxes_classified",
                    rr.Boxes2D(array=arr, array_format=rr.Box2DFormat.XYWH, labels=labels),
                )
            except BaseException as e:
                logger.warning(f"rerun classified bbox log failed: {e}")

        overlay = make_pca_overlay(
            rgb,
            pca_info["centroid_px"],
            pca_info["axis_pt1_px"],
            pca_info["axis_pt2_px"],
        )
        _log_rr_image(f"scan/iter_{iteration_idx}/screw_{rec['idx']}/pca_overlay", overlay)
        _log_rr_text(
            f"scan/iter_{iteration_idx}/screw_{rec['idx']}/info",
            (
                f"class={rec.get('class_name','?')} "
                f"area={rec['mask_area_px']} "
                f"axis_len_px={rec['axis_len_px']:.1f} "
                f"mean_V={rec['mean_v']:.1f} "
                f"world_yaw_deg={pca_info['world_long_axis_yaw_deg']:.2f}"
            ),
        )
        logger.info(
            f"Chosen screw idx={rec['idx']} class={rec.get('class_name','?')} "
            f"area={rec['mask_area_px']} axis_len={rec['axis_len_px']:.1f} "
            f"grasp_pose={grasp_pose}"
        )
        return rec, world_T_camera_mat, rgb, depth, bin_bbox

    logger.warning("No screw produced a valid grasp pose this iteration")
    return None


def main():
    if _HAS_RERUN:
        rr.init("screw_sorting", spawn=True)
        logger.info("Rerun initialized")

    signal.signal(signal.SIGINT, _sigint_handler)

    robot = None
    gripper = None
    camera = None

    try:
        # Connect robot
        logger.info(f"Connecting to UR10E at {ROBOT_IP}")
        robot = UniversalRobotsUR10E()
        robot.connect(ROBOT_IP)
        # Set TCP offset (rx=ry=rz=0, just a Z offset from flange)
        robot.set_tcp([0.0, 0.0, TCP_OFFSET_M, 0.0, 0.0, 0.0])
        logger.success("Robot connected and TCP offset set")

        # Connect gripper
        logger.info(f"Connecting to OnRobot RG2 at {GRIPPER_IP}")
        gripper = OnRobotRG2()
        gripper.connect(GRIPPER_IP, protocol="MODBUS_TCP", verbose=True)
        logger.success("Gripper connected")

        # Connect camera
        logger.info("Connecting to Intel RealSense")
        camera = RealSense(name="head")
        camera.connect()
        logger.success("RealSense connected")

        # Read intrinsics
        global CAMERA_INTRINSICS_MAT, CAMERA_DISTORTION
        fx, fy, h_im, w_im, ppx, ppy, model, coeffs = camera.get_intrinsics("color")
        CAMERA_INTRINSICS_MAT = np.array(
            [[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        CAMERA_DISTORTION = list(coeffs)
        logger.info(
            f"Camera intrinsics: fx={fx} fy={fy} ppx={ppx} ppy={ppy} "
            f"model={model} h={h_im} w={w_im} dist={coeffs}"
        )

        # Move to HOME
        logger.info(f"Moving to HOME joints: {HOME_JOINT_POSITIONS}")
        # robot.set_joint_positions(
        #     HOME_JOINT_POSITIONS,
        #     speed=JOINT_SPEED_DEG_S,
        #     acceleration=JOINT_ACC_DEG_S2,
        # )

        # Pre-open gripper
        logger.info("Opening gripper at start")
        gripper.open(force=GRIPPER_FORCE_N, asynchronous=False)

        # Move to SCAN pose
        logger.info(f"Moving to SCAN pose: {SCAN_POSE}")
        robot.set_cartesian_pose(
            SCAN_POSE,
            speed=CART_SPEED_M_S,
            acceleration=CART_ACC_M_S2,
        )

        robot.set_cartesian_pose(
            SCAN_POSE,
            speed=CART_SPEED_M_S,
            acceleration=CART_ACC_M_S2,
        )
        rgb = camera.capture_color_image()
        bin_bbox = detect_bin(rgb)
        # Main loop: scan, pick, place, repeat
        for it in range(MAX_PICK_ITERATIONS):
            if _stop_requested:
                logger.warning("Stop requested; exiting main loop")
                break
            logger.info(f"========== Iteration {it} ==========")

            # Always rescan from SCAN_POSE (bin contents changed after each pick)
            robot.set_cartesian_pose(
                SCAN_POSE,
                speed=CART_SPEED_M_S,
                acceleration=CART_ACC_M_S2,
            )

            scan_out = scan_and_choose_screw(robot, camera, it, bin_bbox)
            if scan_out is None:
                logger.success("No more screws to pick. Finishing.")
                break

            rec, world_T_camera_mat, rgb, depth, bin_bbox = scan_out
            class_idx = rec.get("class_idx", CLASS_BLACK)
            grasp_pose = rec["grasp_pose_world"]

            # Pick
            try:
                pick_screw(robot, gripper, grasp_pose)
            except BaseException as e:
                logger.error(f"Pick failed: {e}; aborting iteration")
                # Return to scan pose and continue
                try:
                    robot.set_cartesian_pose(
                        SCAN_POSE,
                        speed=CART_SPEED_M_S,
                        acceleration=CART_ACC_M_S2,
                    )
                except BaseException:
                    pass
                continue

            # Place
            box_pose = BOX_POSES[class_idx]
            logger.info(f"Placing screw at class {class_idx} box pose: {box_pose}")
            try:
                place_at_box(robot, gripper, box_pose)
            except BaseException as e:
                logger.error(f"Place failed: {e}; opening gripper to release and continuing")
                try:
                    gripper.open(force=GRIPPER_FORCE_N, asynchronous=False)
                except BaseException:
                    pass

        # Return to HOME at end
        # logger.info("Returning to HOME at end")
        # try:
        #     robot.set_joint_positions(
        #         HOME_JOINT_POSITIONS,
        #         speed=JOINT_SPEED_DEG_S,
        #         acceleration=JOINT_ACC_DEG_S2,
        #     )
        # except BaseException as e:
        #     logger.warning(f"Final HOME move failed: {e}")

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received; proceeding to cleanup")
    except BaseException as e:
        logger.exception(f"Unhandled exception in pipeline: {e}")
    finally:
        # Reset SIGINT to default FIRST so a second Ctrl+C is a hard kill but doesn't skip cleanup.
        try:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        except BaseException:
            pass

        # Disconnect gripper
        try:
            if gripper is not None:
                logger.info("Disconnecting gripper")
                gripper.disconnect()
        except BaseException as e:
            logger.warning(f"Gripper disconnect raised: {e}")

        # Disconnect robot
        try:
            if robot is not None:
                logger.info("Disconnecting robot")
                robot.disconnect()
        except BaseException as e:
            logger.warning(f"Robot disconnect raised: {e}")

        # Disconnect camera
        try:
            if camera is not None:
                logger.info("Disconnecting camera")
                camera.disconnect()
        except BaseException as e:
            logger.warning(f"Camera disconnect raised: {e}")

        logger.info("Cleanup complete.")
        # Do NOT re-raise KeyboardInterrupt — would interrupt cleanup of background threads.


if __name__ == "__main__":
    main()
