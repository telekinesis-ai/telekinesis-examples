"""
Streams joint targets at 125 Hz to oscillate the base joint (j0) sinusoidally with servo_joint.

Supports Universal Robots (UR).

Usage:
    python servo_joint.py [--ip <ROBOT_IP>]
"""

import argparse
import math
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Oscillate the base joint (j0) sinusoidally with servo_joint."""
    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # Servo-loop and trajectory parameters
    dt = 0.008          # 125 Hz servo loop
    amplitude = 2.0     # ±2 deg base oscillation
    period = 4.0        # seconds per full oscillation
    n_cycles = 2

    # ==================== Visualization (Optional) =============================
    # Live: subscribes to the robot's state topic and redraws as it moves.
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
        if ip:
            robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        # Hold all joints at their current values; only j0 will be modulated.
        # Sine starts at 0, so the first target equals ``center`` exactly —
        # no step at t=0.
        center = robot.get_joint_positions()
        logger.info(
            f"Oscillating j0 by ±{amplitude} deg around {center[0]:.3f} deg "
            f"({n_cycles} cycles, {period}s each)"
        )

        duration = period * n_cycles
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t >= duration:
                break

            theta = 2.0 * math.pi * t / period
            target = list(center)
            target[0] = center[0] + amplitude * math.sin(theta)

            # speed/acceleration are not used by UR's servoJ when time > 0,
            # but pass sane non-zero values to avoid controller-side validation
            # rejecting the script.
            robot.servo_joint(
                q=target,
                speed=60.0,
                acceleration=80.0,
                time=dt,
                lookahead_time=0.1,
                gain=300,
            )

            # Pace the loop. Sleep the remainder of this dt window.
            next_tick = t0 + (math.floor(t / dt) + 1) * dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

        robot.servo_stop()
        logger.success("servo_joint loop complete.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR10e servo_joint example")
    parser.add_argument("--ip", type=str, default=None,
                         help="UR robot IP address for real hardware, e.g. 192.168.1.100")
    args = parser.parse_args()

    main(ip=args.ip)
