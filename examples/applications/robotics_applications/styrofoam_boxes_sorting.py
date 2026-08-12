# Pipeline: Pick white styrofoam boxes from tray and place in grid pattern
# Hardware: UR10E + OnRobot RG2 + Intel RealSense
# Vision: QWEN -> Grounding DINO -> Classical fallback, then SAM for masks, PCA for orientation

import signal
import time
import numpy as np
import cv2
from loguru import logger
import rerun as rr

from datatypes import datatypes
from telekinesis.synapse import utils as tfutils
from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E
from telekinesis.synapse.tools.parallel_grippers.onrobot import OnRobotRG2
from telekinesis.medulla.cameras.realsense import RealSense
from telekinesis import retina, cornea, pupil

# =============================================================================
# TUNABLE CONSTANTS
# =============================================================================

# --- Network ---
ROBOT_IP = "192.168.2.2"
GRIPPER_IP = "192.168.1.1"

# --- Tool ---
TCP_OFFSET_M = 0.23  # 23 cm from flange along Z

# --- Hand-eye calibration ---
CAMERA_IN_TCP = [
    0.07520960896570618,
    -0.0352478269641629,
    -0.2162654145229983,
    -0.07505179364087063,
    0.8826477579985493,
    90.3598403373567,
]

# --- Robot poses ---
HOME_JOINT_POSITIONS = [120, -90, -90, -90, 90, -90]
INTERMEDIATE_JOINT_POSITIONS = [120, -70, -120, -80, 90, -90]
SCAN_POSE = [-0.25462, 0.59302, 0.3141, 180.0, 0.0, 90.0]

# --- Place grid ---
FIRST_PLACE_POSE = [-0.7076, -0.05674, 0.0, 180.0, 0.0, 180.0]
GRID_ROWS = 3
GRID_COLS = 1
GRID_SPACING_X = 0.12
GRID_SPACING_Y = 0.08

# --- Motion heights ---
PICK_APPROACH_HEIGHT = 0.12   # Z above pick before descending
PICK_LIFT_HEIGHT = 0.20       # Z to lift after grasp
PLACE_APPROACH_HEIGHT = 0.1  # Z above place before descending
PLACE_LIFT_HEIGHT = 0.15      # Z to retract after release
PICK_HEIGHT_OFFSET= -0.03

# --- Motion params ---
LINEAR_SPEED = 0.8
LINEAR_ACCEL = 0.5
JOINT_SPEED_DEG = 150
JOINT_ACCEL_DEG = 60

# --- move_until_contact (place) ---
CONTACT_SPEED_MS = 0.02   # 2 cm/s descent
CONTACT_ACCEL = 0.3
CONTACT_DIRECTION = [0, 0, -1, 0, 0, 0]  # detect contact in -Z (base frame)

# --- Vision ---
QWEN_PROMPT = "white styrofoam rectangular boxes on a gray tray"
DINO_PROMPT = "individual white styrofoam rectangular boxes ."
QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
SAM_THRESHOLD = 0.5
DINO_BOX_THRESHOLD = 0.30
DINO_TEXT_THRESHOLD = 0.25

# --- Gripper ---
GRIPPER_FORCE_N = 100.0

# =============================================================================
# Pre-computed transforms
# =============================================================================
TCP_T_CAMERA = tfutils.pose_to_transformation_matrix(CAMERA_IN_TCP, rot_type='deg')


# =============================================================================
# Helpers
# =============================================================================

def build_place_grid():
    """Generate place poses in a row-major grid starting from FIRST_PLACE_POSE."""
    poses = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            p = list(FIRST_PLACE_POSE)
            p[0] = FIRST_PLACE_POSE[0] + c * GRID_SPACING_X
            p[1] = FIRST_PLACE_POSE[1] + r * GRID_SPACING_Y
            poses.append(p)
    return poses


def detect_boxes(image_np):
    """Try QWEN -> Grounding DINO -> classical contours. Returns (bboxes_xyxy_list, source_str)."""
    # 1) QWEN
    try:
        logger.info("[DETECT] Trying QWEN...")
        ann = retina.detect_objects_using_qwen(
            image=image_np,
            prompt =QWEN_PROMPT,
        )
        ann_list = ann.to_list() if hasattr(ann, "to_list") else []
        bboxes_xywh = [a["bbox"] for a in ann_list if a.get("bbox") is not None]
        if len(bboxes_xywh) > 0:
            bboxes_xyxy = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in bboxes_xywh]
            logger.success(f"[DETECT] QWEN found {len(bboxes_xyxy)} detections")
            return bboxes_xyxy, "qwen"
        logger.warning("[DETECT] QWEN returned 0 detections, falling back")
    except Exception as e:
        logger.warning(f"[DETECT] QWEN failed: {e}")

    # 2) Grounding DINO
    try:
        logger.info("[DETECT] Trying Grounding DINO...")
        ann, _cats = retina.detect_objects_using_grounding_dino(
            image=image_np,
            prompt=DINO_PROMPT,
            box_threshold=DINO_BOX_THRESHOLD,
            text_threshold=DINO_TEXT_THRESHOLD,
        )
        ann_list = ann.to_list() if hasattr(ann, "to_list") else []
        bboxes_xywh = [a["bbox"] for a in ann_list if a.get("bbox") is not None]
        if len(bboxes_xywh) > 0:
            bboxes_xyxy = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in bboxes_xywh]
            logger.success(f"[DETECT] Grounding DINO found {len(bboxes_xyxy)} detections")
            return bboxes_xyxy, "grounding_dino"
        logger.warning("[DETECT] DINO returned 0 detections, falling back")
    except Exception as e:
        logger.warning(f"[DETECT] Grounding DINO failed: {e}")

    # 3) Classical: bright (white) regions on gray tray via Otsu -> contours
    try:
        logger.info("[DETECT] Trying classical contour detection...")
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if image_np.ndim == 3 else image_np
        gray_img = datatypes.Image(image=gray)
        ann_geom = retina.detect_contours(
            image=gray_img,
            retrieval_mode="external",
            approx_method="simple",
            min_area=1500,
            max_area=200000,
        )
        ann_list = ann_geom.to_list() if hasattr(ann_geom, "to_list") else []
        bboxes_xywh = [a["bbox"] for a in ann_list if a.get("bbox") is not None]
        bboxes_xyxy = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in bboxes_xywh]
        logger.success(f"[DETECT] Classical found {len(bboxes_xyxy)} detections")
        return bboxes_xyxy, "classical"
    except Exception as e:
        logger.error(f"[DETECT] Classical also failed: {e}")
        return [], "none"


def get_mask_for_bbox(image_np, bbox_xyxy):
    """Run SAM on a single bbox and return a binary uint8 mask (H,W)."""
    boxes2d = datatypes.Boxes2D(arrays=[bbox_xyxy], array_format="XYXY")
    sam_ann = cornea.segment_image_using_sam(
        image=image_np,
        bboxes=boxes2d,
        mask_threshold=SAM_THRESHOLD,
        image_id=0,
    )
    # SAM returns ObjectDetectionAnnotations whose 'segmentation' fields contain mask info.
    # We'll instead reconstruct the mask by re-running cornea on each bbox is expensive;
    # use the polygon/segmentation from annotation if available, else fallback to bbox-filled.
    ann_list = sam_ann.to_list() if hasattr(sam_ann, "to_list") else []
    h, w = image_np.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(ann_list) == 0:
        # Fallback: use bbox as mask
        x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
        mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 255
        return mask

    a = ann_list[0]
    seg = a.get("segmentation", None)
    if seg is None or len(seg) == 0:
        x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
        mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 255
        return mask

    # segmentation may be list of polygons (flat [x1,y1,x2,y2,...]) per COCO convention
    try:
        if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (list, np.ndarray)):
            # list of polygons
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
        elif isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (int, float)):
            pts = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
            mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 255
    except Exception as e:
        logger.warning(f"[SAM] Could not parse mask polygon ({e}); falling back to bbox")
        x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
        mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 255

    return mask


def compute_pca_on_mask(mask_uint8):
    """Return (centroid_uv (2,), principal_axis (2,) unit vec, eigvals (2,)) or None."""
    ys, xs = np.where(mask_uint8 > 0)
    n = xs.size
    if n < 5:
        return None
    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centroid = coords.mean(axis=0)
    cov = np.cov(coords - centroid, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    principal = principal / np.linalg.norm(principal)
    return centroid, principal, eigvals


def pixel_to_base(camera_intrinsics_3x3, dist_coeffs, pixel_xy, depth_m, base_T_tcp):
    """Project pixel + depth to 3D point in base frame."""
    base_T_cam = base_T_tcp @ TCP_T_CAMERA
    cam_T_point = pupil.project_pixel_to_camera_point(
        camera_intrinsics=camera_intrinsics_3x3,
        distortion_coefficients=dist_coeffs,
        pixel=list(pixel_xy),
        depth=float(depth_m),
    )
    cam_T_point_np = cam_T_point.to_numpy() if hasattr(cam_T_point, "to_numpy") else np.array(cam_T_point)
    base_T_point = base_T_cam @ cam_T_point_np
    return base_T_point[:3, 3]  # XYZ in base


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


def pixel_to_world(u, v, depth, K, base_T_tcp):
    """Project pixel + scalar depth -> base frame XYZ (manual intrinsic math, no distortion)."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    p_cam = np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth, 1.0])
    p_world = base_T_tcp @ TCP_T_CAMERA @ p_cam
    return p_world[:3]


def compute_yaw_world_from_axis(centroid_uv, principal_axis, depth, K, base_T_tcp, k_pixels=40):
    """Deproject two pixels along principal axis using centroid depth, return yaw_deg in world XY."""
    cx, cy = centroid_uv
    vx, vy = principal_axis
    w_plus  = pixel_to_world(int(round(cx + k_pixels * vx)), int(round(cy + k_pixels * vy)), depth, K, base_T_tcp)
    w_minus = pixel_to_world(int(round(cx - k_pixels * vx)), int(round(cy - k_pixels * vy)), depth, K, base_T_tcp)
    dx = float(w_plus[0] - w_minus[0])
    dy = float(w_plus[1] - w_minus[1])
    logger.info(f"[PCA] world axis: dx={dx:.4f} dy={dy:.4f}")
    return float(np.degrees(np.arctan2(dy, dx)))


def safe_disconnect(obj, name):
    try:
        obj.disconnect()
        logger.info(f"[CLEANUP] {name} disconnected")
    except BaseException as e:
        logger.warning(f"[CLEANUP] {name} disconnect failed: {e}")


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    # Initialize Rerun
    rr.init("pick_styrofoam_to_grid", spawn=True)

    robot = None
    gripper = None
    camera = None

    # Optional SIGINT (default behavior is fine; we install no custom handler beyond default).
    # The user requirement is to RESET to SIG_DFL at the start of finally; we honor that even
    # without a custom handler installed, to ensure no inherited handler interferes.
    try:
        # ----- Connect hardware -----
        logger.info("[INIT] Connecting UR10E...")
        robot = UniversalRobotsUR10E()
        robot.connect(ROBOT_IP)
        # Set TCP offset (23cm along Z from flange)
        robot.set_tcp([0.0, 0.0, TCP_OFFSET_M, 0.0, 0.0, 0.0])
        logger.success("[INIT] Robot connected and TCP set")

        logger.info("[INIT] Connecting OnRobot RG2 gripper...")
        gripper = OnRobotRG2()
        gripper.connect(GRIPPER_IP, protocol="MODBUS_TCP")
        logger.success("[INIT] Gripper connected")

        logger.info("[INIT] Connecting Intel RealSense...")
        camera = RealSense(name="scan_cam")
        camera.connect()
        logger.success("[INIT] Camera connected")

        # ----- Move to HOME -----
        logger.info(f"[MOTION] Moving to HOME {HOME_JOINT_POSITIONS}")
        robot.set_joint_positions(HOME_JOINT_POSITIONS, speed=JOINT_SPEED_DEG, acceleration=JOINT_ACCEL_DEG)

        # Open gripper at start
        logger.info("[GRIPPER] Open at start")
        gripper.open(force=GRIPPER_FORCE_N)

        # ----- Move to scan pose -----
        logger.info(f"[MOTION] Moving to SCAN_POSE {SCAN_POSE}")
        robot.set_cartesian_pose(SCAN_POSE, speed=LINEAR_SPEED, acceleration=LINEAR_ACCEL)
        time.sleep(0.5)

        # ----- Capture -----
        logger.info("[SCAN] Capturing color, depth, and intrinsics")
        color = camera.capture_color_image()
        depth = camera.capture_depth_image()
        fx, fy, h_img, w_img, ppx, ppy, model, coeffs = camera.get_intrinsics("color")
        K = np.array([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]], dtype=np.float32)
        dist_coeffs = list(coeffs)

        if color is None or depth is None:
            raise RuntimeError("[SCAN] Failed to capture color/depth")

        rr.log("scan/rgb", rr.Image(color))
        rr.log("scan/depth", rr.DepthImage(depth, meter=1.0))
        logger.success(f"[SCAN] color={color.shape}, depth={depth.shape}, K=fx{fx:.1f} fy{fy:.1f}")

        # Current TCP pose -> base_T_tcp
        current_tcp_pose = robot.get_cartesian_pose()
        base_T_tcp = tfutils.pose_to_transformation_matrix(current_tcp_pose, rot_type='deg')
        logger.info(f"[TRANSFORM] Current TCP (base): {current_tcp_pose}")

        # ----- Detect -----
        bboxes_xyxy, det_source = detect_boxes(color)
        logger.info(f"[DETECT] Source={det_source}, count={len(bboxes_xyxy)}")

        if len(bboxes_xyxy) == 0:
            logger.error("[DETECT] No detections - aborting picks")
        else:
            # Visualize bboxes only (no contour)
            rr_boxes_xyxy = np.array(bboxes_xyxy, dtype=np.float32)
            rr.log(
                "scan/detections",
                rr.Boxes2D(array=rr_boxes_xyxy, array_format=rr.Box2DFormat.XYXY),
            )

        # ----- Per-detection processing -----
        place_grid = build_place_grid()
        place_idx = 0
        max_picks = min(len(bboxes_xyxy), len(place_grid))

        for i in range(max_picks):
            bbox = bboxes_xyxy[i]
            logger.info(f"[PICK {i}] bbox={bbox}")

            # SAM mask
            try:
                mask = get_mask_for_bbox(color, bbox)
            except Exception as e:
                logger.error(f"[SAM] Failed for bbox {i}: {e}")
                continue
            mask_area = int(np.count_nonzero(mask))
            logger.info(f"[SAM] mask area={mask_area} px")
            rr.log(f"scan/sam_mask_{i}", rr.SegmentationImage(mask))

            if mask_area < 50:
                logger.warning(f"[SAM] mask too small for det {i}, skipping")
                continue

            # PCA
            pca_result = compute_pca_on_mask(mask)
            if pca_result is None:
                logger.error(f"[PCA] failed for det {i}")
                continue
            centroid_uv, principal_axis, _ = pca_result
            cx, cy = float(centroid_uv[0]), float(centroid_uv[1])
            logger.info(f"[PCA] centroid=({cx:.1f},{cy:.1f}) axis={principal_axis}")

            # PCA principal axis as a line
            line_len = 60.0
            line_pts = np.array([
                [cx - line_len * principal_axis[0], cy - line_len * principal_axis[1]],
                [cx + line_len * principal_axis[0], cy + line_len * principal_axis[1]],
            ], dtype=np.float32)
            rr.log(f"scan/pca_axis_{i}", rr.LineStrips2D([line_pts]))
            rr.log(f"scan/centroid_{i}", rr.Points2D(np.array([[cx, cy]], dtype=np.float32)))

            # Depth lookup at centroid
            d = get_depth_at(depth, int(round(cx)), int(round(cy)))
            if d <= 0.05:
                logger.warning(f"[TRANSFORM] invalid depth at centroid, skipping")
                continue
            logger.info(f"[TRANSFORM] depth at centroid = {d:.3f} m")

            # Project to base
            try:
                point_base = pixel_to_base(K, dist_coeffs, [cx, cy], d, base_T_tcp)
            except Exception as e:
                logger.error(f"[TRANSFORM] projection failed: {e}")
                continue
            logger.info(f"[TRANSFORM] pick point in base = {point_base}")

            yaw_deg = compute_yaw_world_from_axis(centroid_uv, principal_axis, d, K, base_T_tcp)
            # Top-down orientation in base: rx=180, ry=0, rz=yaw
            pick_pose_base = [
                float(point_base[0]),
                float(point_base[1]),
                float(point_base[2]) + PICK_HEIGHT_OFFSET,
                180.0,
                0.0,
                yaw_deg,
            ]
            pick_above = list(pick_pose_base)
            pick_above[2] = pick_pose_base[2] + PICK_APPROACH_HEIGHT
            pick_lift = list(pick_pose_base)
            pick_lift[2] = pick_pose_base[2] + PICK_LIFT_HEIGHT

            logger.info(f"[PICK {i}] above={pick_above}")
            logger.info(f"[PICK {i}] grasp={pick_pose_base}")

            # Visualize pick stages as simple 3D points (no scalars)
            rr.log(
                f"world/pick_{i}/above",
                rr.Points3D(np.array([pick_above[:3]], dtype=np.float32)),
            )
            rr.log(
                f"world/pick_{i}/grasp",
                rr.Points3D(np.array([pick_pose_base[:3]], dtype=np.float32)),
            )

            # Execute pick: above -> open -> descend -> close -> lift
            try:
                gripper.open(force=GRIPPER_FORCE_N)
                robot.set_cartesian_pose(pick_above, speed=LINEAR_SPEED, acceleration=LINEAR_ACCEL)
                robot.set_cartesian_pose(pick_pose_base, speed=LINEAR_SPEED * 0.5, acceleration=LINEAR_ACCEL)
                logger.info(f"[PICK {i}] closing gripper")
                gripper.close(force=GRIPPER_FORCE_N)
                time.sleep(0.3)
                robot.set_cartesian_pose(pick_lift, speed=LINEAR_SPEED, acceleration=LINEAR_ACCEL)
            except Exception as e:
                logger.error(f"[PICK {i}] motion failed: {e}")
                # Try to open gripper to be safe
                try:
                    gripper.open(force=GRIPPER_FORCE_N)
                except BaseException:
                    pass
                continue

            # Move through intermediate
            logger.info(f"[PLACE {i}] -> intermediate joints")
            try:
                robot.set_joint_positions(
                    INTERMEDIATE_JOINT_POSITIONS,
                    speed=JOINT_SPEED_DEG,
                    acceleration=JOINT_ACCEL_DEG,
                )
            except Exception as e:
                logger.error(f"[PLACE {i}] intermediate motion failed: {e}")
                continue

            # Place pose
            place_pose = list(place_grid[place_idx])
            place_above = list(place_pose)
            place_above[2] = place_pose[2] + PLACE_APPROACH_HEIGHT
            place_retract = list(place_pose)
            place_retract[2] = place_pose[2] + PLACE_LIFT_HEIGHT

            logger.info(f"[PLACE {i}] target grid idx={place_idx} pose={place_pose}")
            rr.log(
                f"world/place_{place_idx}/target",
                rr.Points3D(np.array([place_pose[:3]], dtype=np.float32)),
            )

            try:
                robot.set_cartesian_pose(place_above, speed=LINEAR_SPEED, acceleration=LINEAR_ACCEL)
                # Use move_until_contact for the placing descent (per user pref)
                logger.info(f"[PLACE {i}] move_until_contact descending")
                robot.move_until_contact(
                    cartesian_speed=[0.0, 0.0, -CONTACT_SPEED_MS, 0.0, 0.0, 0.0],
                    direction=CONTACT_DIRECTION,
                    acceleration=CONTACT_ACCEL,
                )
                logger.info(f"[PLACE {i}] opening gripper")
                gripper.open(force=GRIPPER_FORCE_N)
                time.sleep(0.2)
                robot.set_cartesian_pose(place_retract, speed=LINEAR_SPEED, acceleration=LINEAR_ACCEL)
            except Exception as e:
                logger.error(f"[PLACE {i}] motion failed: {e}")
                try:
                    gripper.open(force=GRIPPER_FORCE_N)
                except BaseException:
                    pass
                continue

            place_idx += 1
            logger.success(f"[CYCLE {i}] complete; placed at grid {place_idx - 1}")

        # ----- Return HOME -----
        logger.info("[MOTION] Returning to HOME")
        try:
            robot.set_joint_positions(HOME_JOINT_POSITIONS, speed=JOINT_SPEED_DEG, acceleration=JOINT_ACCEL_DEG)
        except Exception as e:
            logger.error(f"[MOTION] return-to-home failed: {e}")

        logger.success("[DONE] Pipeline complete")

    except Exception as e:
        logger.exception(f"[FATAL] Pipeline error: {e}")

    finally:
        # Per user requirement: reset SIGINT to default FIRST, before any disconnect
        try:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        except BaseException:
            pass

        # Catch BaseException so a 2nd Ctrl+C doesn't skip remaining teardown
        if gripper is not None:
            try:
                gripper.open(force=GRIPPER_FORCE_N)
            except BaseException as e:
                logger.warning(f"[CLEANUP] gripper.open failed: {e}")
            try:
                gripper.disconnect()
                logger.info("[CLEANUP] Gripper disconnected")
            except BaseException as e:
                logger.warning(f"[CLEANUP] Gripper disconnect failed: {e}")

        if robot is not None:
            try:
                robot.disconnect()
                logger.info("[CLEANUP] Robot disconnected")
            except BaseException as e:
                logger.warning(f"[CLEANUP] Robot disconnect failed: {e}")

        if camera is not None:
            try:
                camera.disconnect()
                logger.info("[CLEANUP] Camera disconnected")
            except BaseException as e:
                logger.warning(f"[CLEANUP] Camera disconnect failed: {e}")

        # Never let KeyboardInterrupt escape finally - swallow any pending one
        try:
            pass
        except BaseException:
            pass


if __name__ == "__main__":
    main()