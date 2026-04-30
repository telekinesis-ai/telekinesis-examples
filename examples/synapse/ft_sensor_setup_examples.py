"""F/T Sensor Setup examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def ft_rtde_input_enable_example(ip: str = "192.168.1.2"):
    """Enable RTDE-fed external F/T input with a 100-g sensor."""
    logger.info("Running ft_rtde_input_enable example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.ft_rtde_input_enable(enable=True, sensor_mass=0.1, sensor_measuring_offset=[0.0, 0.0, 0.01], sensor_cog=[0.0, 0.0, 0.005])
        logger.success("RTDE F/T input enabled.")
        robot.ft_rtde_input_enable(enable=False)
        logger.info("RTDE F/T input disabled.")

    finally:
        robot.disconnect()


def enable_external_ft_sensor_example(ip: str = "192.168.1.2"):
    """Enable the external F/T sensor (deprecated API) with 100-g sensor mass."""
    logger.info("Running enable_external_ft_sensor example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.enable_external_ft_sensor(enable=True, sensor_mass=0.1)
        logger.success("External F/T sensor enabled.")
        robot.enable_external_ft_sensor(enable=False)
        logger.info("External F/T sensor disabled.")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "ft_rtde_input_enable": lambda: ft_rtde_input_enable_example(ip),
        "enable_external_ft_sensor": lambda: enable_external_ft_sensor_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="F/T Sensor Setup Synapse examples")
    parser.add_argument("--ip", type=str, default="192.168.1.2")
    parser.add_argument("--example", type=str)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pause", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    examples = get_example_dict(args.ip)
    if args.list:
        for name in sorted(examples):
            logger.info(f"  - {name}")
        return
    if args.all:
        for name, fn in examples.items():
            logger.info(f"Running {name}...")
            try:
                fn()
            except Exception as e:
                logger.error(f"{name} FAILED: {type(e).__name__}: {e}")
            if args.pause:
                input("Press Enter for next example...")
        return
    if not args.example:
        logger.error("Provide --example, --list, or --all.")
        raise SystemExit(1)
    name = args.example.lower()
    if name not in examples:
        matches = difflib.get_close_matches(name, examples.keys(), n=3, cutoff=0.4)
        logger.error(f"Example '{name}' not found.")
        if matches:
            logger.error("Did you mean: " + ", ".join(matches))
        raise SystemExit(1)
    examples[name]()


if __name__ == "__main__":
    main()
