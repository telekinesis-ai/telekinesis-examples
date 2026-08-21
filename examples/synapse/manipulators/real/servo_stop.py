"""
Streams a brief servo_joint move and interrupts it mid-stream with servo_stop.

Supports Universal Robots (UR).

Usage:
    python servo_stop.py [--ip <ROBOT_IP>]
"""

import argparse
import math
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Stream servo_joint targets for 1 second, then interrupt with servo_stop."""
    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    
    # Servo-loop and trajectory parameters
    dt = 0.008          # 125 Hz servo loop
    amplitude = 2.0     # ±2 deg base oscillation
    period = 4.0        # seconds for one complete sine cycle
    # Only stream for a fraction of ``period``, so the motion is interrupted
    # partway through the cycle (~1/4 cycle here) rather than completing it.
    stream_duration = 1.0  # stream servo targets for this long before stopping
    deceleration = 10.0    # deg/s² for servo_stop

    # ==================== Visualization (Optional) =============================
    # Live: subscribes to the robot's state topic and redraws as it moves.
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        # Hold all joints at their current values; only j0 will be modulated.
        center = robot.get_joint_positions()
        logger.info(
            f"Streaming j0 oscillation for {stream_duration}s, then servo_stop"
        )

        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t >= stream_duration:
                break

            theta = 2.0 * math.pi * t / period
            target = list(center)
            target[0] = center[0] + amplitude * math.sin(theta)

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

        # Interrupt the servo stream — controller ramps the joints down.
        robot.servo_stop(deceleration=deceleration)
        logger.success(f"servo_stop issued (deceleration={deceleration} deg/s²).")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR10e servo_stop example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="IP address of the UR robot (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
