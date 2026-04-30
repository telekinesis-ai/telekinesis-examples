"""I/O and Configuration examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_digital_in_state_example(ip: str = "192.168.1.2"):
    """Read state of digital input 0."""
    logger.info("Running get_digital_in_state example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"DI 0 state: {robot.get_digital_in_state(0)}")

    finally:
        robot.disconnect()


def get_digital_out_state_example(ip: str = "192.168.1.2"):
    """Read state of digital output 0."""
    logger.info("Running get_digital_out_state example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"DO 0 state: {robot.get_digital_out_state(0)}")

    finally:
        robot.disconnect()


def get_payload_example(ip: str = "192.168.1.2"):
    """Read configured payload mass [kg]."""
    logger.info("Running get_payload example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Payload: {robot.get_payload()} kg")

    finally:
        robot.disconnect()


def get_payload_cog_example(ip: str = "192.168.1.2"):
    """Read configured payload centre-of-gravity vector [m]."""
    logger.info("Running get_payload_cog example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Payload CoG: {robot.get_payload_cog()}")

    finally:
        robot.disconnect()


def set_payload_example(ip: str = "192.168.1.2"):
    """Set payload to 2 kg with CoG 5 cm along tool Z."""
    logger.info("Running set_payload example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.set_payload(mass=2.0, cog=[0.0, 0.0, 0.05])
        logger.success(f"Payload set; readback: mass={robot.get_payload()}, cog={robot.get_payload_cog()}")

    finally:
        robot.disconnect()


def set_tcp_example(ip: str = "192.168.1.2"):
    """Set TCP offset to 10 cm along flange Z."""
    logger.info("Running set_tcp example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        tcp = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
        robot.set_tcp(tcp_offset=tcp)
        logger.success(f"TCP set; readback: {robot.get_tcp_offset()}")

    finally:
        robot.disconnect()


def get_step_time_example(ip: str = "192.168.1.2"):
    """Read controller step time [s]."""
    logger.info("Running get_step_time example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Step time: {robot.get_step_time()} s")

    finally:
        robot.disconnect()


def get_speed_scaling_example(ip: str = "192.168.1.2"):
    """Read current speed scaling factor (0..1)."""
    logger.info("Running get_speed_scaling example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Speed scaling: {robot.get_speed_scaling()}")

    finally:
        robot.disconnect()


def get_async_operation_progress_example(ip: str = "192.168.1.2"):
    """Read async-operation progress string."""
    logger.info("Running get_async_operation_progress example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Async progress: {robot.get_async_operation_progress()}")

    finally:
        robot.disconnect()


def get_freedrive_status_example(ip: str = "192.168.1.2"):
    """Read freedrive-status integer."""
    logger.info("Running get_freedrive_status example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Freedrive status: {robot.get_freedrive_status()}")

    finally:
        robot.disconnect()


def get_actual_digital_input_bits_example(ip: str = "192.168.1.2"):
    """Read all digital inputs as a bitmask."""
    logger.info("Running get_actual_digital_input_bits example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"DI bits: {robot.get_actual_digital_input_bits():#018b}")

    finally:
        robot.disconnect()


def get_actual_digital_output_bits_example(ip: str = "192.168.1.2"):
    """Read all digital outputs as a bitmask."""
    logger.info("Running get_actual_digital_output_bits example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"DO bits: {robot.get_actual_digital_output_bits():#018b}")

    finally:
        robot.disconnect()


def get_standard_analog_input_example(ip: str = "192.168.1.2"):
    """Read standard analog inputs 0 and 1."""
    logger.info("Running get_standard_analog_input example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"AI 0: {robot.get_standard_analog_input(0):.4f}")
        logger.success(f"AI 1: {robot.get_standard_analog_input(1):.4f}")

    finally:
        robot.disconnect()


def get_standard_analog_output_example(ip: str = "192.168.1.2"):
    """Read standard analog outputs 0 and 1."""
    logger.info("Running get_standard_analog_output example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"AO 0: {robot.get_standard_analog_output(0):.4f}")
        logger.success(f"AO 1: {robot.get_standard_analog_output(1):.4f}")

    finally:
        robot.disconnect()


def get_output_int_register_example(ip: str = "192.168.1.2"):
    """Read output integer register 18."""
    logger.info("Running get_output_int_register example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Output int reg 18: {robot.get_output_int_register(18)}")

    finally:
        robot.disconnect()


def get_output_double_register_example(ip: str = "192.168.1.2"):
    """Read output double register 18."""
    logger.info("Running get_output_double_register example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        logger.success(f"Output double reg 18: {robot.get_output_double_register(18):.6f}")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "get_digital_in_state": lambda: get_digital_in_state_example(ip),
        "get_digital_out_state": lambda: get_digital_out_state_example(ip),
        "get_payload": lambda: get_payload_example(ip),
        "get_payload_cog": lambda: get_payload_cog_example(ip),
        "set_payload": lambda: set_payload_example(ip),
        "set_tcp": lambda: set_tcp_example(ip),
        "get_step_time": lambda: get_step_time_example(ip),
        "get_speed_scaling": lambda: get_speed_scaling_example(ip),
        "get_async_operation_progress": lambda: get_async_operation_progress_example(ip),
        "get_freedrive_status": lambda: get_freedrive_status_example(ip),
        "get_actual_digital_input_bits": lambda: get_actual_digital_input_bits_example(ip),
        "get_actual_digital_output_bits": lambda: get_actual_digital_output_bits_example(ip),
        "get_standard_analog_input": lambda: get_standard_analog_input_example(ip),
        "get_standard_analog_output": lambda: get_standard_analog_output_example(ip),
        "get_output_int_register": lambda: get_output_int_register_example(ip),
        "get_output_double_register": lambda: get_output_double_register_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="I/O and Configuration Synapse examples")
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
