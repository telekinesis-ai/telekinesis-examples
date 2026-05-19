# Pipeline: Sort screws by size using UR10E + OnRobot RG2 + RealSense
# Strategy: scan -> QWEN (fallback GDINO) -> SAM masks -> classify by area -> PCA grasp -> pick/place

import signal
import sys
import time
import numpy as np
import cv2
import rerun as rr
from loguru import logger

from datatypes import datatypes
from telekinesis import retina, cornea, pupil
from telekinesis.medulla.cameras import RealSense
from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E
from telekinesis.synapse.tools.parallel_grippers.onrobot import OnRobotRG2
from telekinesis.synapse import utils as tfutils

# =========================================================================
# HARDCODED / TUNABLE CONSTANTS
# =========================================================================
ROBOT_IP = "192.168.2.2"
GRIPPER_IP = "192.168.1.1"

# TCP offset (23 cm along flange Z)
TCP_OFFSET = [0.0, 0.0, 0.23, 0.0, 0.0, 0.0]

# Hand-eye calibration
CAMERA_IN_TCP = [0.07520960896570618, -0.0352478269641629, -0.2162654145229983,
                 -0.07505179364087063, 0.8826477579985493, 90.3598403373567]

# Poses (XYZ in m, RPY in deg)
SCAN_POSE = [-0.25462, 0.59302, 0.24541, 180.0, 0.0, 90.0]
ABOVE_BLUE_BIN = [0.27387, 0.58234, 0.2, -180.0, 0.0, 90.0]   # small screws
ABOVE_RED_BIN = [0.27387, 0.75084, 0.2, -180.0, 0.0, 90.0]    # big screws

HOME_JOINT_POSITIONS = [120, -90, -90, -90, 90, -90]
INTERMEDIATE_JOINT_POSITIONS = [120, -70, -120, -80, 90, -90]

# Motion params
J_SPEED = 150.0       # deg/s
J_ACCEL = 80.0       # deg/s^2
L_SPEED = 0.7       # m/s
L_ACCEL = 0.5        # m/s^2

# Picking parameters
PICK_APPROACH_HEIGHT = 0.10   # m above grasp point
PICK_DEPTH_OFFSET = -0.007     # m above the detected surface (avoid crashing)
PLACE_DROP_HEIGHT = 0.05      # additional clearance above bin pose

# Gripper
GRIPPER_FORCE_N = 40.0  # RG2 max
GRIPPER_OPEN_WIDTH_MM = 100.0  # for RG2 max ~110

# Vision
QWEN_PROMPT = "Different screws on a gray tray . "
GDINO_PROMPT = "screws ."
SCORE_THRESHOLD = 0.25
SAM_MASK_THRESHOLD = 0.5

# Size classification: relative to median mask area
SIZE_THRESHOLD_RATIO = 0.80  # area > median*ratio => big

# Camera intrinsics fallback (will fetch real ones from camera)
DEFAULT_DISTORTION = [0.0, 0.0, 0.0, 0.0, 0.0]

# =========================================================================
# Helpers
# =========================================================================

def pca_angle_from_mask(mask: np.ndarray) -> tuple[float, tuple[int, int], tuple[float, float]]:
    """Compute PCA principal angle (rad), centroid (cx, cy), and unit eigenvector (vx, vy) from binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return 0.0, (int(np.mean(xs)) if len(xs) else 0, int(np.mean(ys)) if len(ys) else 0), (1.0, 0.0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean, eigvecs = cv2.PCACompute(pts, mean=None)
    cx, cy = float(mean[0, 0]), float(mean[0, 1])
    vx, vy = float(eigvecs[0, 0]), float(eigvecs[0, 1])
    angle = np.arctan2(vy, vx)  # principal axis in image plane
    return angle, (int(cx), int(cy)), (vx, vy)


def compute_yaw_world_from_axis(centroid_uv, principal_axis_uv, depth, K, dist, world_T_tcp, tcp_T_camera, k_pixels=40):
    """Deproject two points along the principal axis into world space, return screw axis yaw (deg) in world XY."""
    cx, cy = centroid_uv
    vx, vy = principal_axis_uv
    u_plus  = int(round(cx + k_pixels * vx))
    v_plus  = int(round(cy + k_pixels * vy))
    u_minus = int(round(cx - k_pixels * vx))
    v_minus = int(round(cy - k_pixels * vy))
    w_plus  = pixel_to_world(u_plus,  v_plus,  depth, K, dist, world_T_tcp, tcp_T_camera)
    w_minus = pixel_to_world(u_minus, v_minus, depth, K, dist, world_T_tcp, tcp_T_camera)
    dx = float(w_plus[0] - w_minus[0])
    dy = float(w_plus[1] - w_minus[1])
    logger.info(f"World axis: dx={dx:.4f} dy={dy:.4f}")
    return float(np.degrees(np.arctan2(dy, dx)))


def pixel_to_world(u: int, v: int, depth: float, K: np.ndarray, dist: list,
                   world_T_tcp: np.ndarray, tcp_T_camera: np.ndarray) -> np.ndarray:
    """Project pixel + depth -> world frame point."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_c = (u - cx) * depth / fx
    y_c = (v - cy) * depth / fy
    z_c = depth
    p_cam = np.array([x_c, y_c, z_c, 1.0])
    p_tcp = tcp_T_camera @ p_cam
    p_world = world_T_tcp @ p_tcp
    return p_world[:3]


def get_depth_at(depth_img: np.ndarray, u: int, v: int, win: int = 5) -> float:
    """Sample median depth in a window to be robust."""
    h, w = depth_img.shape
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth_img[v0:v1, u0:u1]
    valid = patch[(patch > 0.05) & (patch < 2.0)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def detect_screws(color_img: np.ndarray):
    """Try QWEN first, then GDINO. Returns ObjectDetectionAnnotations."""
    # try:
    #     logger.info("Running QWEN detection...")
    #     anns = retina.detect_objects_using_qwen(color_img, QWEN_PROMPT)
    #     if anns is not None and len(anns.to_list()) > 0:
    #         logger.info(f"QWEN found {len(anns.to_list())} detections")
    #         return anns
    #     logger.warning("QWEN returned no detections, falling back to GDINO")
    # except BaseException as e:
    #     logger.warning(f"QWEN failed: {e}, falling back to GDINO")

    try:
        logger.info("Running Grounding DINO detection...")
        anns, _cats = retina.detect_objects_using_grounding_dino(
            color_img, GDINO_PROMPT, box_threshold=SCORE_THRESHOLD, text_threshold=SCORE_THRESHOLD
        )
        if anns is not None and len(anns.to_list()) > 0:
            logger.info(f"GDINO found {len(anns.to_list())} detections")
            return anns
        logger.warning("GDINO also returned no detections")
    except BaseException as e:
        logger.error(f"GDINO failed: {e}")
    return None


def anns_to_xyxy_boxes(anns) -> list[list[int]]:
    """Convert ObjectDetectionAnnotations (COCO bbox xywh) to XYXY list."""
    out = []
    for a in anns.to_list():
        x, y, w, h = a["bbox"]
        out.append([int(x), int(y), int(x + w), int(y + h)])
    return out


def run_sam(color_img: np.ndarray, xyxy_boxes: list[list[int]]):
    """Run SAM on detection boxes, return mask list (H,W) bool arrays."""
    if not xyxy_boxes:
        return []
    logger.info(f"Running SAM on {len(xyxy_boxes)} boxes")
    sam_anns = cornea.segment_image_using_sam(color_img, xyxy_boxes, mask_threshold=SAM_MASK_THRESHOLD)
    masks = []
    h, w = color_img.shape[:2]
    for a in sam_anns.to_list():
        seg = a.get("segmentation", None)
        if seg is None:
            masks.append(np.zeros((h, w), dtype=bool))
            continue
        # Try to convert polygon-or-rle into binary mask via decoding logic.
        # Easiest: rasterize from bbox-cropped polygon if list, else fallback.
        m = np.zeros((h, w), dtype=np.uint8)
        if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (list, np.ndarray)):
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(m, [pts], 1)
        masks.append(m.astype(bool))
    return masks


def classify_sizes(masks: list[np.ndarray]) -> list[str]:
    """Return 'big' or 'small' per mask using median area as threshold."""
    areas = np.array([int(m.sum()) for m in masks], dtype=np.float64)
    if len(areas) == 0:
        return []
    median = float(np.median(areas))
    threshold = median * SIZE_THRESHOLD_RATIO
    labels = ["big" if a > threshold else "small" for a in areas]
    logger.info(f"Mask areas: {areas.tolist()}, median={median:.1f}, labels={labels}")
    return labels


# =========================================================================
# Main
# =========================================================================

def main():
    rr.init("screw_sort", spawn=True)

    robot = UniversalRobotsUR10E()
    gripper = OnRobotRG2()
    camera = RealSense(name="cam0")

    interrupted = {"flag": False}

    def _sigint(_sig, _frm):
        interrupted["flag"] = True
        logger.warning("SIGINT received - will stop after current step")

    signal.signal(signal.SIGINT, _sigint)

    try:
        # ----- Connect -----
        logger.info("Connecting hardware...")
        robot.connect(ROBOT_IP)
        robot.set_tcp(TCP_OFFSET)
        gripper.connect(GRIPPER_IP)
        camera.connect(warmup_frames=30)

        # Camera intrinsics
        fx, fy, h_img, w_img, ppx, ppy, _model, dist = camera.get_intrinsics("color")
        K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]], dtype=np.float64)
        logger.info(f"Camera intrinsics: fx={fx}, fy={fy}, ppx={ppx}, ppy={ppy}")

        TCP_T_CAMERA = tfutils.pose_to_transformation_matrix(CAMERA_IN_TCP, rot_type='deg')

        # ----- Home -----
        logger.info("Moving to HOME")
        robot.set_joint_positions(HOME_JOINT_POSITIONS, speed=J_SPEED, acceleration=J_ACCEL)
        gripper.open(force=GRIPPER_FORCE_N)

        # ----- Single scan -----
        logger.info("Moving to SCAN_POSE")
        robot.set_cartesian_pose(SCAN_POSE, speed=L_SPEED, acceleration=L_ACCEL)
        time.sleep(0.5)

        # Capture
        color = camera.capture_color_image()
        depth = camera.capture_depth_image()
        if color is None or depth is None:
            logger.error("Capture failed, aborting")
            return

        rr.log("scan/color", rr.Image(color))
        rr.log("scan/depth", rr.DepthImage(depth, meter=1.0))

        # Detect
        anns = detect_screws(color)
        if anns is None or len(anns.to_list()) == 0:
            logger.warning("No screws detected, aborting")
            return

        xyxy = anns_to_xyxy_boxes(anns)
        # Visualize bboxes (xywh format expected by rerun Boxes2D? use min/half)
        mins = np.array([[b[0], b[1]] for b in xyxy])
        sizes = np.array([[b[2] - b[0], b[3] - b[1]] for b in xyxy])
        rr.log("scan/detections", rr.Boxes2D(mins=mins, sizes=sizes))

        # SAM
        masks = run_sam(color, xyxy)
        if not masks:
            logger.warning("SAM produced no masks, aborting")
            return

        # Visualize masks combined as label image
        label_img = np.zeros(color.shape[:2], dtype=np.uint16)
        for i, m in enumerate(masks, start=1):
            label_img[m] = i
        rr.log("scan/masks", rr.SegmentationImage(label_img))

        # Classify sizes
        labels = classify_sizes(masks)

        # Compute all grasp candidates once (using TCP pose at scan)
        tcp_pose = robot.get_cartesian_pose()
        world_T_tcp = tfutils.pose_to_transformation_matrix(tcp_pose, rot_type='deg')

        candidates = []
        order = np.argsort([-int(m.sum()) for m in masks])
        for idx in order:
            m = masks[idx]
            size_label = labels[idx]
            if int(m.sum()) < 50:
                continue

            angle_rad, (cx, cy), (vx_img, vy_img) = pca_angle_from_mask(m)
            logger.info(f"Screw idx={idx} size={size_label} centroid=({cx},{cy}) pca_angle_deg={np.degrees(angle_rad):.1f}")

            rr.log(f"scan/grasp_centroid_{idx}", rr.Points2D([[cx, cy]], radii=8.0))

            _L = 60
            p1 = [cx - vx_img * _L, cy - vy_img * _L]
            p2 = [cx + vx_img * _L, cy + vy_img * _L]
            rr.log(f"scan/grasp_axis_{idx}", rr.LineStrips2D([[p1, p2]], radii=2.0))

            d = get_depth_at(depth, cx, cy)
            if d <= 0.05:
                logger.warning(f"Invalid depth at ({cx},{cy}) - skipping")
                continue

            p_world = pixel_to_world(cx, cy, d, K, list(dist), world_T_tcp, TCP_T_CAMERA)
            logger.info(f"3D grasp point (world): {p_world}")

            rr.log(f"scan/grasp_world_{idx}", rr.Points3D([p_world.tolist()], radii=0.005))

            # Screw axis angle in world XY; gripper grasps perpendicular to it
            axis_yaw_deg = compute_yaw_world_from_axis(
                (cx, cy), (vx_img, vy_img), d, K, list(dist), world_T_tcp, TCP_T_CAMERA
            )
            yaw_deg = axis_yaw_deg
            # Wrap into [-180, 180]
            yaw_deg = ((yaw_deg + 180.0) % 360.0) - 180.0

            grasp_xyz = [float(p_world[0]), float(p_world[1]), float(p_world[2]) + PICK_DEPTH_OFFSET]
            approach_pose = [grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + PICK_APPROACH_HEIGHT,
                             180.0, 0.0, yaw_deg]
            grasp_pose = [grasp_xyz[0], grasp_xyz[1], grasp_xyz[2],
                          180.0, 0.0, yaw_deg]

            candidates.append({
                "idx": int(idx),
                "size_label": size_label,
                "approach_pose": approach_pose,
                "grasp_pose": grasp_pose,
            })

        if not candidates:
            logger.warning("No valid grasp candidates, aborting")
            return

        logger.info(f"Computed {len(candidates)} grasp candidates - starting pick loop")

        # ----- Iterate over precomputed detections -----
        for i, cand in enumerate(candidates):
            if interrupted["flag"]:
                break
            logger.info(f"=== Pick {i + 1}/{len(candidates)} (idx={cand['idx']}, size={cand['size_label']}) ===")

            approach_pose = cand["approach_pose"]
            grasp_pose = cand["grasp_pose"]
            size_label = cand["size_label"]

            logger.info(f"Approach pose: {approach_pose}")
            logger.info(f"Grasp pose:    {grasp_pose}")

            # Open, approach, descend, close
            gripper.open(force=GRIPPER_FORCE_N)
            robot.set_cartesian_pose(approach_pose, speed=L_SPEED, acceleration=L_ACCEL)
            robot.set_cartesian_pose(grasp_pose, speed=L_SPEED * 0.5, acceleration=L_ACCEL)
            logger.info("Closing gripper to grasp")
            gripper.close(force=GRIPPER_FORCE_N)
            # Lift
            robot.set_cartesian_pose(approach_pose, speed=L_SPEED, acceleration=L_ACCEL)

            # Place
            logger.info("Moving to INTERMEDIATE_JOINT_POSITIONS before drop")
            robot.set_joint_positions(INTERMEDIATE_JOINT_POSITIONS, speed=J_SPEED, acceleration=J_ACCEL)

            if size_label == "big":
                target_bin_pose = ABOVE_RED_BIN
                logger.info("Placing BIG screw in RED bin")
            else:
                target_bin_pose = ABOVE_BLUE_BIN
                logger.info("Placing SMALL screw in BLUE bin")

            robot.set_cartesian_pose(target_bin_pose, speed=L_SPEED, acceleration=L_ACCEL)
            logger.info("Opening gripper to release")
            gripper.open(force=GRIPPER_FORCE_N)

            # Back to intermediate to keep clearance
            robot.set_joint_positions(INTERMEDIATE_JOINT_POSITIONS, speed=J_SPEED, acceleration=J_ACCEL)

        # Final home
        logger.info("Returning to HOME")
        robot.set_joint_positions(HOME_JOINT_POSITIONS, speed=J_SPEED, acceleration=J_ACCEL)

    finally:
        # Reset SIGINT immediately so a second Ctrl+C does not abort cleanup
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        try:
            gripper.disconnect()
        except BaseException as e:
            logger.error(f"Gripper disconnect error: {e}")

        try:
            robot.disconnect()
        except BaseException as e:
            logger.error(f"Robot disconnect error: {e}")

        try:
            camera.disconnect()
        except BaseException as e:
            logger.error(f"Camera disconnect error: {e}")

        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()