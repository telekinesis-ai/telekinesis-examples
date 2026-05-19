# Pipeline: Scan-first pick-and-place of white styrofoam boxes
# Phase 1: Visit all scan poses, accumulate detections + deprojected world poses.
# Phase 2: Cluster world poses across all scans to deduplicate.
# Phase 3: Pick each clustered object, drop into collection box.
#
# Heavy logging + Rerun visualization at every stage.
# Robust signal handling: SIGINT reset before teardown, each disconnect guarded.

import signal
import sys
import time
import math
import numpy as np
import cv2
from loguru import logger

import rerun as rr

from telekinesis.synapse import utils as tfutils
from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E
from telekinesis.synapse.tools.parallel_grippers.onrobot import OnRobotRG2
from telekinesis.medulla.cameras.realsense import RealSense
from telekinesis import cornea


from telekinesis import retina, pupil
from datatypes import datatypes

# =====================================================================
# Tunable constants
# =====================================================================
ROBOT_IP = "192.168.2.2"
GRIPPER_IP = "192.168.1.1"
CAMERA_NAME = "realsense_top"

TCP_OFFSET = [0.0, 0.0, 0.23, 0.0, 0.0, 0.0]  # 23 cm tool offset
PAYLOAD_MASS = 1.0

# Hand-eye calibration
CAMERA_IN_TCP = [
    0.07520960896570618,
    -0.0352478269641629,
    -0.2162654145229983,
    -0.07505179364087063,
    0.8826477579985493,
    90.3598403373567,
]
TCP_T_CAMERA = tfutils.pose_to_transformation_matrix(CAMERA_IN_TCP, rot_type="deg")

# Scan grid
SCAN_START_POSE = [-0.25462, 0.59302, 0.3141, 180.0, 0.0, 90.0]
GRID_NX = 1
GRID_NY = 1
GRID_DX = 0.2
GRID_DY = -0.2

# Drop pose
DROP_POSE = [-0.740, 0.727, -0.2, -180.0, 0.0, 90.0]

# Pick parameters
APPROACH_HEIGHT = 0.15  # meters above object before descent
GRASP_Z = 0.015          # final pick z (meters, world frame); tune to table height
LIFT_HEIGHT = 0.20      # how high to lift after grasp
PICK_ORIENTATION = [180.0, 0.0, 90.0]  # gripper-down, aligned with scan poses

# Clustering
CLUSTER_XY_THRESHOLD = 0.012  # meters, treat detections within this XY radius as same object

# Motion params
MOVE_SPEED = 0.4
MOVE_ACC = 0.6

# Detection
QWEN_PROMPT = ("wrappers or napkins"
               )
GROUNDING_DINO_PROMPT = (
    "silver circle ."
)

# SAM/PCA tuning
SAM_AXIS_SAMPLE_PIXELS = 30  # pixel offset along principal axis for yaw computation
MIN_SAM_MASK_PX = 100        # masks smaller than this are rejected as noise

# Workspace bounds (sanity filter on deprojected points; world frame)
WS_X = (-1.0, 1.0)
WS_Y = (-1.0, 1.0)
WS_Z = (-0.05, 0.5)


INTERMEDIATE_JOINT_POSITIONS = [140, -100, -80, -90, 90, -70]
HOME_JOINT_POSITIONS = [120, -90, -90, -90, 90, -90]
# =====================================================================
# Helpers
# =====================================================================


def build_scan_poses(start, nx, ny, dx, dy):
    poses = []
    for iy in range(ny):
        # serpentine ordering for shorter travel
        x_indices = range(nx) if iy % 2 == 0 else range(nx - 1, -1, -1)
        for ix in x_indices:
            p = list(start)
            p[0] = start[0] + ix * dx
            p[1] = start[1] + iy * dy
            poses.append(p)
    return poses


def deproject_pixel_to_world(pixel_xy, depth, intrinsics, world_T_camera):
    """Deproject a pixel (cx, cy) to world coordinates using depth image.
    Returns world_T_point (4x4) or None if depth invalid."""
    cx = int(round(pixel_xy[0]))
    cy = int(round(pixel_xy[1]))
    H, W = depth.shape[:2]
    if not (0 <= cx < W and 0 <= cy < H):
        logger.warning(f"  pixel ({cx},{cy}) outside depth image {W}x{H}")
        return None
    # Sample a small patch median for robustness
    x0 = max(cx - 3, 0)
    x1 = min(cx + 4, W)
    y0 = max(cy - 3, 0)
    y1 = min(cy + 4, H)
    patch = depth[y0:y1, x0:x1]
    valid = patch[(patch > 0) & np.isfinite(patch)]
    if valid.size == 0:
        logger.warning(f"  no valid depth at pixel ({cx},{cy})")
        return None
    z = float(np.median(valid))

    fx, fy, _h, _w, ppx, ppy, _model, coeffs = intrinsics
    pixel = [float(cx), float(cy)]
    distortion = list(coeffs)

    cam_intrinsics = np.array([[fx, 0.0, ppx],
                               [0.0, fy, ppy],
                               [0.0, 0.0, 1.0]])

    world_T_pt_mat = pupil.project_pixel_to_world_point(
        camera_intrinsics=cam_intrinsics,
        distortion_coefficients=distortion,
        pixel=pixel,
        depth=z,
        world_T_camera=world_T_camera,
    )
    return np.array(world_T_pt_mat.to_numpy()) if hasattr(
        world_T_pt_mat, "to_numpy") else np.array(world_T_pt_mat)


def deproject_bbox_center_to_world(bbox_xywh, depth, intrinsics, world_T_camera):
    """Deproject the center of an XYWH bbox to world coordinates using depth image.
    Returns world_T_point (4x4) or None if depth invalid."""
    x, y, w, h = bbox_xywh
    cx = x + w / 2.0
    cy = y + h / 2.0
    return deproject_pixel_to_world((cx, cy), depth, intrinsics, world_T_camera)


def cluster_xy(points_world, threshold):
    """Greedy clustering by XY distance. points_world: list of np.array shape (3,)."""
    clusters = []  # list of list of indices
    used = [False] * len(points_world)
    for i in range(len(points_world)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(points_world)):
            if used[j]:
                continue
            dx = points_world[i][0] - points_world[j][0]
            dy = points_world[i][1] - points_world[j][1]
            if math.hypot(dx, dy) <= threshold:
                group.append(j)
                used[j] = True
        clusters.append(group)
    centroids = []
    for grp in clusters:
        pts = np.array([points_world[k] for k in grp])
        centroids.append(pts.mean(axis=0))
    return centroids, clusters


def detect_objects(rgb_image):
    """Try QWEN -> Grounding DINO -> classical contours. Returns Boxes2D-like list of [x,y,w,h] and label."""
    # 1) QWEN
    try:
        logger.info("  trying QWEN detector...")
        ann = retina.detect_objects_using_qwen(image=rgb_image, prompt=QWEN_PROMPT)
        items = ann.to_list() if hasattr(ann, "to_list") else []
        boxes = [it.get("bbox", [0, 0, 0, 0]) for it in items]
        if boxes:
            logger.info(f"  QWEN found {len(boxes)} candidate(s)")
            return boxes, "qwen"
        logger.info("  QWEN returned 0 detections, falling back")
    except Exception as e:
        logger.warning(f"  QWEN failed: {e}")

    # 2) Grounding DINO
    try:
        logger.info("  trying Grounding DINO detector...")
        ann, _cats = retina.detect_objects_using_grounding_dino(
            image=rgb_image, prompt=GROUNDING_DINO_PROMPT,
            box_threshold=0.25, text_threshold=0.25,
        )
        items = ann.to_list() if hasattr(ann, "to_list") else []
        boxes = [it.get("bbox", [0, 0, 0, 0]) for it in items]
        if boxes:
            logger.info(f"  Grounding DINO found {len(boxes)} candidate(s)")
            return boxes, "grounding_dino"
        logger.info("  Grounding DINO returned 0 detections, falling back to classical")
    except Exception as e:
        logger.warning(f"  Grounding DINO failed: {e}")

    # 3) Classical fallback: detect bright/white blobs via Otsu + contours
    try:
        logger.info("  trying classical contour detector...")
        seg_ann = retina.detect_contours(
            image=rgb_image,
            retrieval_mode="external",
            approx_method="simple",
            min_area=500,
            max_area=200000,
        )
        items = seg_ann.to_list() if hasattr(seg_ann, "to_list") else []
        boxes = [it.get("bbox", [0, 0, 0, 0]) for it in items]
        logger.info(f"  classical contours found {len(boxes)} candidate(s)")
        return boxes, "classical"
    except Exception as e:
        logger.warning(f"  classical detector failed: {e}")
        return [], "none"


def run_sam_on_bboxes(rgb_image, bboxes_xywh):
    """Run SAM with QWEN bboxes (converted to XYXY) as prompts. Returns list of binary masks (H,W) bool, one per bbox, or None on failure for a given bbox."""
    if not bboxes_xywh:
        return []
    try:
        # convert XYWH -> XYXY for SAM
        bboxes_xyxy = []
        for b in bboxes_xywh:
            x, y, w, h = b
            bboxes_xyxy.append([int(x), int(y), int(x + w), int(y + h)])
        logger.info(f"  running SAM on {len(bboxes_xyxy)} bboxes (XYXY)")

        sam_ann = cornea.segment_image_using_sam(
            rgb_image, bboxes=bboxes_xyxy, mask_threshold=0.5, image_id=0)
        items = sam_ann.to_list() if hasattr(sam_ann, "to_list") else []
        logger.info(f"  SAM returned {len(items)} annotation items")
        masks = []
        H, W = rgb_image.shape[:2]
        for j, it in enumerate(items):
            seg = it.get("segmentation", None)
            mask = None
            if isinstance(seg, dict):
                # RLE-style dict: try 'counts' decoded mask if present, else try array conversion
                # Many implementations include a binary mask under a key; fallback: skip
                if "mask" in seg:
                    mask = np.asarray(seg["mask"], dtype=bool)
                elif "counts" in seg and "size" in seg:
                    # we don't decode RLE here; comment the gap
                    # NOTE: RLE decoding not available via Telekinesis API; skipping
                    logger.warning(
                        f"    SAM item {j}: RLE segmentation, no decoder available, skipping")
                    mask = None
            elif isinstance(seg, (list, np.ndarray)):
                arr = np.asarray(seg)
                if arr.ndim == 2 and arr.shape == (H, W):
                    mask = arr.astype(bool)
                else:
                    try:
                        canvas = np.zeros((H, W), dtype=np.uint8)
                        polys = seg if (
                            isinstance(
                                seg, list) and len(seg) > 0 and isinstance(
                                seg[0], (list, np.ndarray))) else [seg]
                        for poly in polys:
                            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(canvas, [pts], 255)
                        mask = canvas.astype(bool)
                        logger.info(
                            f"    SAM item {j}: decoded polygon mask, pixel count = {int(mask.sum())}")
                    except Exception as poly_e:
                        logger.warning(
                            f"    SAM item {j}: unexpected segmentation shape {arr.shape}, could not decode as polygon: {poly_e}")
                        mask = None
            if mask is None:
                logger.warning(f"    SAM item {j}: could not extract mask")
            else:
                px = int(mask.sum())
                if px < MIN_SAM_MASK_PX:
                    logger.warning(
                        f"    SAM item {j}: mask too small ({px} px < {MIN_SAM_MASK_PX}), rejecting")
                    mask = None
                else:
                    logger.info(f"    SAM item {j}: mask pixel count = {px}")
            masks.append(mask)
        # pad/truncate to match bbox count
        while len(masks) < len(bboxes_xywh):
            masks.append(None)
        return masks[:len(bboxes_xywh)]
    except Exception as e:
        logger.warning(f"  SAM failed: {e}")
        return [None] * len(bboxes_xywh)


def compute_mask_pca(mask):
    """Compute PCA on mask True pixel coordinates.
    Returns (centroid_uv (2,), principal_axis (2,) unit vec, eigenvalues (2,)) or None."""
    if mask is None:
        return None
    ys, xs = np.where(mask)
    n = xs.size
    logger.info(f"    PCA: mask has {n} True pixels")
    if n < 5:
        logger.warning(f"    PCA: too few mask pixels ({n}), skipping")
        return None
    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)  # (N,2) (u,v)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending; take largest
    idx = int(np.argmax(eigvals))
    principal = eigvecs[:, idx]
    # normalize
    norm = np.linalg.norm(principal)
    if norm < 1e-9:
        logger.warning("    PCA: principal eigenvector has near-zero norm")
        return None
    principal = principal / norm
    logger.info(f"    PCA: centroid_uv={centroid}, eigvals={eigvals}, principal_axis={principal}")
    return centroid, principal, eigvals


def compute_yaw_world_from_axis(
        centroid_uv,
        principal_axis_uv,
        depth,
        intrinsics,
        world_T_camera,
        k_pixels):
    """Sample two pixels along principal axis around centroid, deproject both, compute world XY yaw (degrees)."""
    p_plus = (centroid_uv[0] + k_pixels * principal_axis_uv[0],
              centroid_uv[1] + k_pixels * principal_axis_uv[1])
    p_minus = (centroid_uv[0] - k_pixels * principal_axis_uv[0],
               centroid_uv[1] - k_pixels * principal_axis_uv[1])
    logger.info(f"    yaw: sampling p_plus={p_plus}, p_minus={p_minus} (k={k_pixels})")
    w_plus = deproject_pixel_to_world(p_plus, depth, intrinsics, world_T_camera)
    w_minus = deproject_pixel_to_world(p_minus, depth, intrinsics, world_T_camera)
    if w_plus is None or w_minus is None:
        logger.warning("    yaw: could not deproject one or both axis sample points")
        return None
    pt_plus = w_plus[:3, 3]
    pt_minus = w_minus[:3, 3]
    dx = pt_plus[0] - pt_minus[0]
    dy = pt_plus[1] - pt_minus[1]
    yaw_rad = math.atan2(dy, dx)
    yaw_deg = math.degrees(yaw_rad)
    logger.info(f"    yaw: world dx={dx}, dy={dy}, yaw_deg={yaw_deg}")
    return yaw_deg


# =====================================================================
# Main pipeline
# =====================================================================
def main():
    rr.init("scan_first_pick_place", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    robot = None
    gripper = None
    camera = None
    interrupted = {"flag": False}

    def sigint_handler(_sig, _frame):
        logger.warning("SIGINT received, requesting graceful shutdown...")
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        # ---------- Connect ----------
        logger.info("=== Phase 0: connecting hardware ===")
        robot = UniversalRobotsUR10E()
        robot.connect(ROBOT_IP)
        robot.set_tcp(TCP_OFFSET)
        robot.set_payload(PAYLOAD_MASS)
        logger.info(f"robot connected, current pose: {robot.get_cartesian_pose()}")

        gripper = OnRobotRG2()
        gripper.connect(GRIPPER_IP, protocol="MODBUS_TCP", verbose=True)
        logger.info("gripper connected, opening to home")
        gripper.open(force=40.0, asynchronous=False)

        camera = RealSense(name=CAMERA_NAME)
        camera.connect(warmup_frames=30)
        logger.info("camera connected and warmed up")

        robot.set_joint_positions(
            HOME_JOINT_POSITIONS)
        # ---------- Phase 1: scan all poses ----------
        logger.info("=== Phase 1: scanning grid ===")
        scan_poses = build_scan_poses(SCAN_START_POSE, GRID_NX, GRID_NY, GRID_DX, GRID_DY)
        logger.info(f"built scan grid with {len(scan_poses)} poses")

        for idx, p in enumerate(scan_poses):
            rr.log(f"world/scan_poses/{idx}",
                   rr.Points3D([p[:3]], radii=0.01, colors=[(0, 255, 255)]),
                   static=True)

        scans = []  # list of dicts: {pose, rgb, depth, detections, world_poses, yaws}

        for i, scan_pose in enumerate(scan_poses):
            if interrupted["flag"]:
                logger.warning("interrupt during scan phase, breaking")
                break
            logger.info(f"--- scan {i+1}/{len(scan_poses)} pose={scan_pose} ---")
            robot.set_cartesian_pose(scan_pose, speed=MOVE_SPEED, acceleration=MOVE_ACC)
            time.sleep(0.3)  # settle

            rgb = camera.capture_single_color_frame()
            depth = camera.capture_single_depth_frame()
            if rgb is None or depth is None:
                logger.warning(f"  scan {i}: failed to capture frames")
                continue
            rr.log(f"scan/{i}/rgb", rr.Image(rgb))
            rr.log(f"scan/{i}/depth", rr.DepthImage(depth, meter=1.0))

            # Compute world_T_camera for this scan pose
            tcp_pose_now = robot.get_cartesian_pose()
            base_T_tcp = tfutils.pose_to_transformation_matrix(tcp_pose_now, rot_type="deg")
            world_T_camera = base_T_tcp @ TCP_T_CAMERA
            logger.info(f"  world_T_camera computed (translation={world_T_camera[:3, 3]})")

            # Camera intrinsics for color stream
            try:
                intrinsics = camera.get_intrinsics("color")
            except Exception as e:
                logger.error(f"  failed to read intrinsics: {e}")
                continue

            # Detect
            bboxes, detector_used = detect_objects(rgb)
            logger.info(f"  detector={detector_used}, {len(bboxes)} bboxes")

            # Log bboxes to Rerun
            if bboxes:
                centers = []
                sizes = []
                for b in bboxes:
                    bx, by, bw, bh = b
                    centers.append([bx + bw / 2, by + bh / 2])
                    sizes.append([bw, bh])
                rr.log(
                    f"scan/{i}/rgb/bboxes",
                    rr.Boxes2D(centers=centers, sizes=sizes,
                               colors=[(255, 0, 0)] * len(bboxes)),
                )

            # Run SAM + PCA on the bboxes whenever a detector found something
            masks = [None] * len(bboxes)
            pca_results = [None] * len(bboxes)
            if detector_used in ("qwen", "grounding_dino") and bboxes:
                masks = run_sam_on_bboxes(rgb, bboxes)
                for j, m in enumerate(masks):
                    if m is None:
                        continue
                    # log mask as segmentation image
                    try:
                        seg = m.astype(np.uint16)
                        rr.log(f"scan/{i}/mask/{j}", rr.SegmentationImage(seg))
                    except Exception as e:
                        logger.warning(f"    failed to log SAM mask {j} to Rerun: {e}")
                    pca_results[j] = compute_mask_pca(m)

            # Deproject each detection
            world_poses = []
            yaws = []
            for j, bbox in enumerate(bboxes):
                pca = pca_results[j] if j < len(pca_results) else None
                yaw_deg = None
                if pca is not None:
                    centroid_uv, principal_axis_uv, _eigvals = pca
                    # Use mask centroid for deprojection
                    world_T_pt = deproject_pixel_to_world(
                        (centroid_uv[0], centroid_uv[1]),
                        depth, intrinsics, world_T_camera)
                    # Compute yaw from principal axis
                    yaw_deg = compute_yaw_world_from_axis(
                        centroid_uv, principal_axis_uv, depth,
                        intrinsics, world_T_camera, SAM_AXIS_SAMPLE_PIXELS)
                    # Log principal axis as 2D line on RGB
                    try:
                        p_plus = [centroid_uv[0] + SAM_AXIS_SAMPLE_PIXELS * principal_axis_uv[0],
                                  centroid_uv[1] + SAM_AXIS_SAMPLE_PIXELS * principal_axis_uv[1]]
                        p_minus = [centroid_uv[0] - SAM_AXIS_SAMPLE_PIXELS * principal_axis_uv[0],
                                   centroid_uv[1] - SAM_AXIS_SAMPLE_PIXELS * principal_axis_uv[1]]
                        rr.log(
                            f"scan/{i}/rgb/principal_axis/{j}",
                            rr.LineStrips2D([[p_minus, p_plus]],
                                            colors=[(0, 255, 0)]),
                        )
                    except Exception as e:
                        logger.warning(f"    failed to log principal axis {j} to Rerun: {e}")
                else:
                    # Fallback: bbox-center deprojection
                    world_T_pt = deproject_bbox_center_to_world(
                        bbox, depth, intrinsics, world_T_camera)

                if world_T_pt is None:
                    continue
                pt = world_T_pt[:3, 3]
                if not (WS_X[0] <= pt[0] <= WS_X[1]
                        and WS_Y[0] <= pt[1] <= WS_Y[1]
                        and WS_Z[0] <= pt[2] <= WS_Z[1]):
                    logger.info(f"    bbox {j}: world point {pt} outside workspace, skipping")
                    continue
                logger.info(f"    bbox {j}: deprojected to world {pt}, yaw_deg={yaw_deg}")
                world_poses.append(pt)
                yaws.append(yaw_deg)
                rr.log(f"world/scan_{i}/det_{j}",
                       rr.Points3D([pt], radii=0.012, colors=[(255, 128, 0)]))

            # Build detections list with mask/axis/yaw
            detections_full = []
            for j, bbox in enumerate(bboxes):
                pca = pca_results[j] if j < len(pca_results) else None
                principal_axis_pixel = pca[1] if pca is not None else None
                # find matching yaw from yaws list (yaws aligns with kept world_poses, not bboxes)
                detections_full.append({
                    "bbox": bbox,
                    "mask": masks[j] if j < len(masks) else None,
                    "principal_axis_pixel": principal_axis_pixel,
                })

            scans.append({
                "scan_index": i,
                "pose": scan_pose,
                "detections": detections_full,
                "world_poses": world_poses,
                "yaws": yaws,
                "detector": detector_used,
            })
            logger.info(f"  scan {i} accumulated {len(world_poses)} world poses, yaws={yaws}")

        # ---------- Phase 2: cluster ----------
        logger.info("=== Phase 2: clustering accumulated detections ===")
        all_world_poses = []
        all_yaws = []
        for s in scans:
            for p, y in zip(s["world_poses"], s["yaws"]):
                all_world_poses.append(p)
                all_yaws.append(y)
        logger.info(f"total raw detections across scans: {len(all_world_poses)}")
        if not all_world_poses:
            logger.warning("no detections to cluster, nothing to pick")
            return

        centroids, groups = cluster_xy(all_world_poses, CLUSTER_XY_THRESHOLD)
        logger.info(f"clustering produced {len(centroids)} unique objects "
                    f"(threshold={CLUSTER_XY_THRESHOLD} m)")

        # Compute per-cluster yaw via circular mean
        cluster_yaws = []
        for k, grp in enumerate(groups):
            yaws_in_cluster = [all_yaws[idx] for idx in grp if all_yaws[idx] is not None]
            if not yaws_in_cluster:
                cluster_yaw = None
            else:
                sin_sum = sum(math.sin(math.radians(y)) for y in yaws_in_cluster)
                cos_sum = sum(math.cos(math.radians(y)) for y in yaws_in_cluster)
                cluster_yaw = math.degrees(math.atan2(sin_sum, cos_sum))
            cluster_yaws.append(cluster_yaw)
            logger.info(f"  cluster {k}: centroid={centroids[k]}, size={len(grp)}, "
                        f"yaws={yaws_in_cluster}, mean_yaw_deg={cluster_yaw}")
            rr.log(f"world/clusters/{k}",
                   rr.Points3D([centroids[k]], radii=0.018, colors=[(0, 255, 0)]),
                   static=True)
            try:
                rr.log(f"world/clusters/{k}/yaw_text",
                       rr.TextLog(f"cluster {k} yaw_deg={cluster_yaw}"))
            except Exception as e:
                logger.warning(f"  failed to log cluster {k} yaw text: {e}")

        # ---------- Phase 3: pick & place ----------
        logger.info("=== Phase 3: picking clustered objects ===")
        for k, centroid in enumerate(centroids):
            if interrupted["flag"]:
                logger.warning("interrupt during pick phase, breaking")
                break
            logger.info(f"--- pick {k+1}/{len(centroids)} target={centroid} ---")

            obj_x, obj_y, _obj_z = float(centroid[0]), float(centroid[1]), float(centroid[2])

            cluster_yaw = cluster_yaws[k]
            if cluster_yaw is None:
                pick_orientation = list(PICK_ORIENTATION)
                logger.info(
                    f"  cluster {k}: no yaw available, using default PICK_ORIENTATION={pick_orientation}")
            else:
                pick_orientation = [180.0, 0.0, cluster_yaw]
                logger.info(f"  cluster {k}: pick orientation set to {pick_orientation}")

            approach_pose = [obj_x, obj_y, APPROACH_HEIGHT,
                             *pick_orientation]
            grasp_pose = [obj_x, obj_y, GRASP_Z, *pick_orientation]
            lift_pose = [obj_x, obj_y, APPROACH_HEIGHT + LIFT_HEIGHT,
                         *pick_orientation]

            rr.log(f"world/pick_{k}/approach",
                   rr.Points3D([approach_pose[:3]], radii=0.01,
                               colors=[(0, 0, 255)]))
            rr.log(f"world/pick_{k}/grasp",
                   rr.Points3D([grasp_pose[:3]], radii=0.01,
                               colors=[(255, 0, 255)]))

            # Open gripper, approach, descend (NO move_until_contact for picking)
            try:
                logger.info("  opening gripper")
                gripper.open(force=40.0, asynchronous=False)

                logger.info(f"  moving to approach {approach_pose}")
                robot.set_cartesian_pose(approach_pose,
                                         speed=MOVE_SPEED, acceleration=MOVE_ACC)

                logger.info(f"  descending to grasp {grasp_pose}")
                robot.set_cartesian_pose(grasp_pose,
                                         speed=0.15, acceleration=0.3)

                logger.info("  closing gripper to grasp")
                grasp_status = gripper.close(force=40.0, asynchronous=False)
                logger.info(f"  grasp status: {grasp_status}")

                logger.info(f"  lifting to {lift_pose}")
                robot.set_cartesian_pose(lift_pose,
                                         speed=MOVE_SPEED, acceleration=MOVE_ACC)

                robot.set_joint_positions(
                    INTERMEDIATE_JOINT_POSITIONS)

                logger.info(f"  moving to drop pose {DROP_POSE}")
                robot.set_cartesian_pose(DROP_POSE,
                                         speed=MOVE_SPEED, acceleration=MOVE_ACC)

                logger.info("  opening gripper to release")
                gripper.open(force=40.0, asynchronous=False)
                rr.log(f"world/drop_{k}",
                       rr.Points3D([DROP_POSE[:3]], radii=0.015,
                                   colors=[(255, 255, 0)]))
                logger.success(f"  pick {k+1} completed")
                robot.set_joint_positions(
                    INTERMEDIATE_JOINT_POSITIONS)
            except Exception as e:
                logger.error(f"  pick {k+1} failed: {e}")
                # try to recover gripper state
                try:
                    gripper.open(force=40.0, asynchronous=False)
                except Exception:
                    pass

        logger.info("=== pipeline complete ===")
        robot.set_joint_positions(
            HOME_JOINT_POSITIONS)

    except Exception as e:
        logger.exception(f"top-level error: {e}")
    finally:
        # Reset SIGINT BEFORE any disconnect so a second Ctrl+C does not skip teardown
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        logger.info("=== cleanup ===")

        try:
            if gripper is not None:
                logger.info("disconnecting gripper")
                gripper.disconnect()
        except BaseException as e:
            logger.error(f"gripper disconnect error: {e}")

        try:
            if robot is not None:
                logger.info("disconnecting robot")
                robot.disconnect()
        except BaseException as e:
            logger.error(f"robot disconnect error: {e}")

        try:
            if camera is not None:
                logger.info("disconnecting camera")
                camera.disconnect()
        except BaseException as e:
            logger.error(f"camera disconnect error: {e}")

        logger.info("teardown complete")


if __name__ == "__main__":
    main()
