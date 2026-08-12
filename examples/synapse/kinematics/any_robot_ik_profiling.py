"""IK time-profiling script.

Generates n_poses feasible target poses via FK on random joint configurations,
runs IK with profiling enabled on each, computes overall statistics, and saves
results to CSV.

Supports sweeping a single IK parameter (e.g. dt) over multiple values.
When sweeping, only overall statistics per parameter value are printed/saved.

python any_robot_ik_profiling.py --n-poses 1000
python any_robot_ik_profiling.py --test-all-robots --n-poses 1000

"""

import math
import itertools
import pathlib
import argparse
import csv
import inspect
import sys
import time

import numpy as np
import pinocchio
from loguru import logger

from telekinesis.synapse import utils
from telekinesis.synapse.robots.manipulators import (
    universal_robots,
    kuka,
    abb,
    motoman,
    neura_robotics,
    fanuc,
)

ROW_FIELDS = [
    "pose_index",
    "total_ik_call_ms",
    "raw_solver_ms",
    "check_types_ms",
    "pose_transform_ms",
    "num_seeds_tried",
    "winning_seed_index",
    "min_seed_time_ms",
    "max_seed_time_ms",
    "mean_seed_time_ms",
    "total_seed_time_ms",
    "min_seed_iters",
    "max_seed_iters",
    "mean_seed_iters",
    "linear_error_norm_meters",
    "angular_error_norm_rad",
]

SKIP_ROBOT_CLASS_NAMES = {
    "ABB",
    "Kuka",
    "UniversalRobots",
    "Motoman",
    "NeuraRobotics",
    "FrankaRobotics",
    "Fanuc",
}


def get_robot():
    robot = universal_robots.UniversalRobotsUR10E()
    robot.active_tcp = 'tool0'
    # robot = kuka.KukaKR6R9002()
    # robot = kuka.KukaKR150R31002()
    # robot = kuka.KukaLBRIIWA14R820()
    # robot = abb.AbbCRB15000595()
    # robot = abb.AbbIRB2400()
    return robot


def discover_robot_classes():
    """Collect concrete robot classes from imported manipulator modules."""
    # modules = [universal_robots]
    modules = [
        universal_robots,
        kuka,
        abb,
        motoman,
        neura_robotics,
        fanuc,
    ]  # franka_robotics
    robot_classes = []

    for module in modules:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if cls.__name__ in SKIP_ROBOT_CLASS_NAMES:
                continue
            robot_classes.append(cls)

    robot_classes.sort(key=lambda c: (c.__module__, c.__name__))
    return robot_classes


def save_results_csv(rows: list[dict], output_csv: pathlib.Path) -> None:
    """Save aggregated all-robot results to CSV."""
    if not rows:
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: _convert(v) for k, v in row.items()} for row in rows])

    logger.success(f"CSV results saved to {output_csv}")


def print_table(rows: list[dict], fields: list[str]) -> None:
    col_widths = {f: max(len(f), max(len(str(r[f])) for r in rows)) for f in fields}
    header = "  ".join(f"{f:<{col_widths[f]}}" for f in fields)
    separator = "  ".join("-" * col_widths[f] for f in fields)
    print("\n" + header + "\n" + separator)
    for row in rows:
        print("  ".join(f"{str(row[f]):<{col_widths[f]}}" for f in fields))


def run_single_robot(
    robot,
    n_poses: int,
    solver: str,
    rng: np.random.Generator,
    use_noised_joint_configs_as_qinit: bool,
    ik_param_sweep: dict | None,
):
    """Run the profiling flow for one robot and return result data + summary rows."""
    logger.info(f"Generating {n_poses} feasible poses via FK for {_get_robot_name(robot)}...")

    feasible_poses, joint_configs = _generate_feasible_poses(
        robot,
        n_poses=n_poses,
        mode="discretized",
        grid_points_per_joint=5,
        rng=rng,
    )

    csv_output = {}
    aggregate_rows = []

    if ik_param_sweep is None:
        rows = _run_profiling(
            robot,
            n_poses,
            feasible_poses,
            joint_configs,
            solver,
            use_noised_joint_configs_as_qinit=use_noised_joint_configs_as_qinit,
        )
        if not rows:
            logger.error(f"No successful IK solutions for {_get_robot_name(robot)} — skipping.")
            return None, []

        print_table(rows, ROW_FIELDS)
        stats = _compute_overall_stats(rows, n_poses, robot)
        _print_overall_stats(stats)

        csv_output["config"] = {
            "robot_name": _get_robot_name(robot),
            "robot_ndof": int(robot.ndof),
            "n_poses": n_poses,
            "solver": solver,
        }
        csv_output["overall"] = stats
        aggregate_rows.append(stats)
    else:
        param_name, param_values = next(iter(ik_param_sweep.items()))
        logger.info(f"Sweeping '{param_name}' over {param_values} for {_get_robot_name(robot)}")
        sweep_results = []

        for val in param_values:
            label = f"{_get_robot_name(robot)} | {param_name}={val}"
            logger.info(f"\n{'─' * 60}\n  {label}\n{'─' * 60}")

            rows = _run_profiling(
                robot,
                n_poses,
                feasible_poses,
                joint_configs,
                solver,
                ik_overrides={param_name: val},
                use_noised_joint_configs_as_qinit=use_noised_joint_configs_as_qinit,
            )

            if not rows:
                logger.warning(f"  No solutions for {label} — skipping.")
                continue

            stats = _compute_overall_stats(rows, n_poses, robot)
            stats[param_name] = val
            _print_overall_stats(stats, label=label)
            sweep_results.append(stats)
            aggregate_rows.append(stats)

        if not sweep_results:
            logger.error(f"All sweep values failed for {_get_robot_name(robot)} — skipping.")
            return None, []

        sweep_fields = [param_name] + [k for k in sweep_results[0] if k != param_name]
        print_table(sweep_results, sweep_fields)

        csv_output["config"] = {
            "robot_name": _get_robot_name(robot),
            "robot_ndof": int(robot.ndof),
            "n_poses": n_poses,
            "solver": solver,
            "ik_param_sweep": param_name,
            "sweep_values": [_convert(v) for v in param_values],
        }
        csv_output["sweep_results"] = sweep_results

    return csv_output, aggregate_rows


def _get_robot_name(robot) -> str:
    return robot.__class__.__name__


def _build_row(pose_index: int, timing: dict) -> dict:
    seed_times_s = timing["seed_times_s"]
    seed_times_ms = [t * 1000.0 for t in seed_times_s]
    seed_iters = timing["seed_iterations"]
    # col_times = timing["collision_check_times_s"]

    return {
        "pose_index": pose_index,
        "raw_solver_ms": round(timing["raw_solver_s"] * 1000.0, 6),
        "check_types_ms": round(timing["check_types_s"] * 1000.0, 6),
        "pose_transform_ms": round(timing["pose_transform_s"] * 1000.0, 6),
        "num_seeds_tried": timing["num_seeds_tried"],
        "winning_seed_index": timing["winning_seed_index"],
        "min_seed_time_ms": round(min(seed_times_ms), 6) if seed_times_ms else 0.0,
        "max_seed_time_ms": round(max(seed_times_ms), 6) if seed_times_ms else 0.0,
        "mean_seed_time_ms": round(float(np.mean(seed_times_ms)), 6) if seed_times_ms else 0.0,
        "total_seed_time_ms": round(sum(seed_times_ms), 6),
        "min_seed_iters": min(seed_iters) if seed_iters else 0,
        "max_seed_iters": max(seed_iters) if seed_iters else 0,
        "mean_seed_iters": round(float(np.mean(seed_iters)), 2) if seed_iters else 0.0,
        "linear_error_m": _convert(timing["linear_error_m"]),
        "angular_error_rad_vec": _convert(timing["angular_error_rad_vec"]),
        "linear_error_norm_meters": round(float(timing["linear_error_norm_meters"]), 6),
        "angular_error_norm_rad": round(float(timing["angular_error_norm_rad"]), 6),
    }


def _compute_overall_stats(rows: list[dict], n_poses_requested: int, robot) -> dict:
    def _avg(key):
        return round(float(np.mean([r[key] for r in rows])), 6)

    seeds_tried = [r["num_seeds_tried"] for r in rows]
    success_rate = 100.0 * len(rows) / n_poses_requested if n_poses_requested > 0 else 0.0

    return {
        "robot_name": _get_robot_name(robot),
        "robot_ndof": int(robot.ndof),
        "n_poses": int(n_poses_requested),
        "n_poses_solved": len(rows),
        "success_rate_%": round(success_rate, 2),
        "avg_lin_error_meters": _avg("linear_error_norm_meters"),
        "avg_ang_error_rad": _avg("angular_error_norm_rad"),
        "avg_total_ik_call_ms": _avg("total_ik_call_ms"),
        "avg_raw_solver_ms": _avg("raw_solver_ms"),
        "avg_check_types_ms": _avg("check_types_ms"),
        "avg_pose_transform_ms": _avg("pose_transform_ms"),
        "avg_total_seed_time_ms": _avg("total_seed_time_ms"),
        "avg_num_seeds_tried": round(float(np.mean(seeds_tried)), 2),
        # "min_num_seeds_tried": int(min(seeds_tried)),
        "max_num_seeds_tried": int(max(seeds_tried)),
    }


def _print_overall_stats(stats: dict, label: str = "") -> None:
    title = f"Overall Statistics{f' ({label})' if label else ''}"
    print(f"\n{'=' * len(title)}")
    print(title)
    print("=" * len(title))
    for key, val in stats.items():
        print(f"  {key:<30s}  {val}")
    print()


def _validate_ik_solution(
    robot,
    q_solution: np.ndarray,
    q_original: np.ndarray,
    pose_index: int,
    se3_tol: float = 1e-3,
) -> None:
    """Compare IK solution against the joint config that generated the target pose.

    First checks if joint configs match directly. If they differ, compares the
    FK poses in SE3 using the Lie-group log distance. Raises if poses differ.
    """
    # joint_limits, joint_configs, and IK output are all in degrees now,
    # so this validator stays in degrees end-to-end.
    q_solution = np.asarray(q_solution, dtype=float)

    if not robot.in_joint_limits(q_solution, verbose=True):
        raise RuntimeError(
            f"q solution limits: IK solution {q_solution} is not inside of joint limits"
        )

    if np.allclose(q_solution, q_original, atol=1e-4):
        return

    pose_solution = robot.forward_kinematics(q_solution)
    pose_original = robot.forward_kinematics(q_original)

    T_sol = utils.pose_to_transformation_matrix(pose_solution, rot_type="deg")
    T_orig = utils.pose_to_transformation_matrix(pose_original, rot_type="deg")

    se3_sol = pinocchio.SE3(T_sol[:3, :3], T_sol[:3, 3])
    se3_orig = pinocchio.SE3(T_orig[:3, :3], T_orig[:3, 3])

    diff = se3_orig.actInv(se3_sol)
    log_err = np.linalg.norm(pinocchio.log6(diff).vector)

    if log_err > se3_tol:
        raise RuntimeError(
            f"Pose {pose_index}: IK solution differs from original in SE3 "
            f"(log6 norm = {log_err:.6f}, tol = {se3_tol})"
        )


def _run_profiling(
    robot,
    n_poses: int,
    feasible_poses: list,
    joint_configs: list,
    solver: str,
    ik_overrides: dict | None = None,
    use_noised_joint_configs_as_qinit: bool = False,
) -> list[dict]:
    ik_kwargs: dict = {
        "profile": True,
        "solver": solver,
        #  'dt': 0.1
    }
    if ik_overrides:
        ik_kwargs.update(ik_overrides)

    rows = []
    for i, pose in enumerate(feasible_poses):
        if use_noised_joint_configs_as_qinit:
            # joint_configs[i] is already in degrees; IK expects q_init in
            # degrees. Add a small (~0.5°) jitter for warm-start variance.
            q_init = np.asarray(joint_configs[i], dtype=float)
            noise = np.random.normal(loc=0.0, scale=0.57, size=q_init.shape)
            ik_kwargs["q_init"] = q_init + noise
        else:
            ik_kwargs["q_init"] = None

        try:
            if solver == "tracik":
                t0 = time.perf_counter()
                q = robot.inverse_kinematics(target_pose=pose, **ik_kwargs)
                dt = time.perf_counter() - t0
                dt_ms = round(dt * 1000.0, 6)
                if q is None:
                    logger.warning(f"  Pose {i}: tracik failed in {dt_ms:.4f} ms")
                    continue
                _validate_ik_solution(robot, q, joint_configs[i], i)
                rows.append(
                    {
                        "pose_index": i,
                        "total_ik_call_ms": dt_ms,
                        "num_seeds_tried": 1,
                        "winning_seed_index": 0,
                        "min_seed_time_ms": dt_ms,
                        "max_seed_time_ms": dt_ms,
                        "mean_seed_time_ms": dt_ms,
                        "total_seed_time_ms": dt_ms,
                        "min_seed_iters": 0,
                        "max_seed_iters": 0,
                        "mean_seed_iters": 0.0,
                        "linear_error_m": 0.0,
                        "angular_error_rad_vec": [0.0, 0.0, 0.0],
                        "linear_error_norm_meters": 0.0,
                        "angular_error_norm_rad": 0.0,
                    }
                )
                logger.success(f"  Pose {i}: solved in {dt_ms:.4f} ms (tracik)")
            else:
                t0 = time.perf_counter()
                q, timing = robot.inverse_kinematics(target_pose=pose, **ik_kwargs)
                total_ik_call_ms = round((time.perf_counter() - t0) * 1000.0, 6)
                _validate_ik_solution(robot, q, joint_configs[i], i)
                row = _build_row(i, timing)
                row["total_ik_call_ms"] = total_ik_call_ms
                rows.append(row)
                logger.success(
                    f"  Pose {i}: total_ik_call {total_ik_call_ms:.4f} ms, "
                    f"solver {timing['raw_solver_s'] * 1000.0:.4f} ms, "
                    f"check_types {timing['check_types_s'] * 1000.0:.4f} ms, "
                    f"pose_transform {timing['pose_transform_s'] * 1000.0:.4f} ms "
                    f"({timing['num_seeds_tried']} seed(s), "
                    f"winning index {timing['winning_seed_index']}, "
                    f"linear error {timing['linear_error_norm_meters']:.6f} mm, "
                    f"angular error {timing['angular_error_norm_rad']:.6f} rad)"
                )
        except (RuntimeError, TypeError, ValueError) as e:
            logger.error(f"{robot.__class__.__name__}  Error during IK {e}")
    return rows


def _generate_feasible_poses(
    robot,
    n_poses: int,
    mode: str = "random",
    grid_points_per_joint: int | None = None,
    joint_grid_points: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> list:
    """
    Generate feasible end-effector poses by sampling the robot joint space.

    Parameters
    ----------
    robot
        Robot model with:
        - robot.joint_limits: array of shape (n_joints, 2)
        - robot.forward_kinematics(q) (q in degrees)
    n_poses : int
        Number of poses to generate (maximum used in discretized mode).
    mode : str
        "random"      -> sample uniformly in continuous joint space
        "discretized" -> sample from a discretized joint-space grid
    grid_points_per_joint : int | None
        Number of grid points for every joint in discretized mode.
        Example: 3 => [min, mid, max] for each joint.
    joint_grid_points : list[int] | None
        Per-joint number of grid points in discretized mode.
        Overrides grid_points_per_joint if provided.
        Example for 6 joints: [3, 3, 5, 3, 2, 2]
    rng : np.random.Generator | None
        Optional RNG for reproducibility.

    Returns
    -------
    list
        List of feasible poses computed via forward kinematics.
    """
    if rng is None:
        rng = np.random.default_rng()

    eps = 1e-4
    joint_limits = np.asarray(robot.joint_limits)
    joint_limits = joint_limits.copy()
    joint_limits[:, 0] += eps  # push lower bounds slightly inward
    joint_limits[:, 1] -= eps  # push upper bounds slightly inward
    n_joints = joint_limits.shape[0]

    if mode == "random":
        feasible_poses = []
        joint_configs = []
        for _ in range(n_poses):
            q_rand = rng.uniform(
                low=joint_limits[:, 0],
                high=joint_limits[:, 1],
            )
            # joint_limits are in degrees, so q_rand is in degrees too.
            pose = robot.forward_kinematics(q_rand)
            feasible_poses.append(pose)
            joint_configs.append(q_rand)
        return feasible_poses

    elif mode == "discretized":
        if joint_grid_points is not None:
            if len(joint_grid_points) != n_joints:
                raise ValueError(
                    f"joint_grid_points must have length {n_joints}, got {len(joint_grid_points)}"
                )
            points_per_joint = joint_grid_points
        else:
            if grid_points_per_joint is None:
                raise ValueError(
                    "For mode='discretized', provide either "
                    "grid_points_per_joint or joint_grid_points."
                )
            points_per_joint = [grid_points_per_joint] * n_joints

        # Build 1D grids for each joint
        joint_grids = []
        for j in range(n_joints):
            n_points = points_per_joint[j]
            if n_points < 1:
                raise ValueError(f"Grid size for joint {j} must be >= 1")
            q_min, q_max = joint_limits[j]
            if n_points == 1:
                # Use midpoint if only one point requested
                grid = np.array([(q_min + q_max) / 2.0])
            else:
                grid = np.linspace(q_min, q_max, n_points)
            joint_grids.append(grid)

        total_combinations = math.prod(len(g) for g in joint_grids)

        # Generate all combinations if small enough, otherwise randomly sample
        # from the discretized grid without constructing all poses first.
        feasible_poses = []
        joint_configs = []

        if total_combinations <= n_poses:
            for q in itertools.product(*joint_grids):
                q_arr = np.asarray(q, dtype=float)
                pose = robot.forward_kinematics(q_arr)
                feasible_poses.append(pose)
                joint_configs.append(q_arr)
        else:
            # Randomly sample points from the discrete grid
            for _ in range(n_poses):
                q_disc = np.array(
                    [grid[rng.integers(0, len(grid))] for grid in joint_grids],
                    dtype=float,
                )
                pose = robot.forward_kinematics(q_disc, verbose=True)
                feasible_poses.append(pose)
                joint_configs.append(q_disc)

        return feasible_poses, joint_configs

    else:
        raise ValueError("mode must be either 'random' or 'discretized'")


def _convert(v):
    """Convert numpy/non-standard types to plain Python for CSV serialization."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def main():
    """
    Run inverse kinematics (IK) profiling from the command line.

    Parses CLI arguments to configure and execute IK solver benchmarks for one
    or multiple robot models. Supports parameter sweeps, reproducible random
    sampling, and optional initialization strategies.

    Modes:
    - Single robot: profiles the default robot and saves results to CSV.
    - All robots: discovers available robot classes, profiles each, prints a
    summary table, and saves aggregated results to CSV.

    Outputs are written to the specified output directory (created if not
    provided explicitly).
    """
    parser = argparse.ArgumentParser(description="IK profiling script")
    default_output = pathlib.Path(__file__).parent / "outputs"

    parser.add_argument(
        "--n-poses",
        type=int,
        default=1_000,
        help="Number of poses (default: 1000)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="multi_start_clik",
        help="Solver type (default: multi_start_clik)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--use-noised-joint-configs-as-qinit",
        action="store_true",
        help="Use noised joint configs as qinit (default: False)",
    )
    parser.add_argument(
        "--test-all-robots",
        action="store_true",
        help="Profile all robot classes and save summary CSV",
    )
    parser.add_argument(
        "--output-folder",
        type=pathlib.Path,
        default=default_output,
        help="Output directory (default: script directory)",
    )

    args = parser.parse_args()

    # Configuration
    n_poses = args.n_poses
    solver = args.solver
    seed = args.seed
    use_noised_joint_configs_as_qinit = args.use_noised_joint_configs_as_qinit
    test_all_robots = args.test_all_robots
    output_dir = args.output_folder

    logger.info(f"use_noised_joint_configs_as_qinit: {use_noised_joint_configs_as_qinit}")

    # Detect if user explicitly passed the argument
    user_provided = "--output-folder" in sys.argv
    if not output_dir.exists():
        if user_provided:
            parser.error(f"--output-folder does not exist: {output_dir}")
            exit(1)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    # Output directory is now guaranteed to exist, either as default or user-provided
    output_dir = pathlib.Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=seed)
    # IK parameter sweep: set to None for a single run, or provide a dict
    # mapping one IK solver keyword argument to a list of values to benchmark.
    # Examples:
    # ik_param_sweep = {"dt": [0.1, .35, 0.5, 0.75]}
    # ik_param_sweep = {"it_max": [1000, 10000]}
    # ik_param_sweep = {"damp": [1e-15, 1e-14, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8]}
    #   ik_param_sweep = {"max_num_q_init_candidates": [50, 100, 300, 500]}
    #   ik_param_sweep = None  # single run with defaults
    ik_param_sweep = None

    if not test_all_robots:
        robot = get_robot()
        robot.setup_kinematics_solver(solver)
        result, result_rows = run_single_robot(
            robot=robot,
            n_poses=n_poses,
            solver=solver,
            rng=rng,
            use_noised_joint_configs_as_qinit=use_noised_joint_configs_as_qinit,
            ik_param_sweep=ik_param_sweep,
        )

        if result is None:
            return

        # Save overall stats as CSV
        if result_rows:
            output_csv = output_dir / "ik_profiling_overall.csv"
            save_results_csv(result_rows, output_csv)

        return

    # All-robots mode
    all_robot_classes = discover_robot_classes()
    all_result_rows = []

    logger.info(f"Discovered {len(all_robot_classes)} robot classes to test.")

    for RobotCls in all_robot_classes:
        robot_label = f"{RobotCls.__module__}.{RobotCls.__name__}"
        logger.info(f"\n{'=' * 80}\nTesting {robot_label}\n{'=' * 80}")

        try:
            robot = RobotCls()
        except Exception as e:
            logger.warning(f"Skipping {robot_label}: could not instantiate ({e})")
            continue

        try:
            robot_result, result_rows = run_single_robot(
                robot=robot,
                n_poses=n_poses,
                solver=solver,
                rng=np.random.default_rng(seed=seed),  # reset seed per robot for fair comparison
                use_noised_joint_configs_as_qinit=use_noised_joint_configs_as_qinit,
                ik_param_sweep=ik_param_sweep,
            )
        except Exception as e:
            logger.exception(f"Skipping {robot_label}: profiling failed ({e})")
            exit(1)
            continue

        if robot_result is None:
            continue

        all_result_rows.extend(result_rows)

    if not all_result_rows:
        logger.error("No robot produced successful results — nothing to save.")
        return

    # Final printed summary table
    summary_fields = list(all_result_rows[0].keys())
    print("\n" + "=" * 80)
    print("ALL ROBOTS SUMMARY")
    print("=" * 80)
    print_table(all_result_rows, summary_fields)

    # Save CSV
    output_csv = output_dir / "ik_profiling_results_all_robots.csv"
    save_results_csv(all_result_rows, output_csv)


if __name__ == "__main__":
    main()
