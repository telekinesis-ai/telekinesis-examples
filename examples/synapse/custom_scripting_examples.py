"""Custom Scripting examples for the Synapse SDK on Universal Robots UR10e."""

import argparse
import difflib

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def send_custom_script_example(ip: str = "192.168.1.2"):
    """Send an inline URScript program (textmsg)."""
    logger.info("Running send_custom_script example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        script = 'textmsg("hello from custom script")\n'
        robot.send_custom_script(script)
        logger.success("Inline URScript sent.")

    finally:
        robot.disconnect()


def send_custom_script_file_example(ip: str = "192.168.1.2"):
    """Send a URScript file (writes a temp file with a textmsg)."""
    logger.info("Running send_custom_script_file example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.script', delete=False, mode='w') as fh:
            fh.write('textmsg("hello from script file")\n')
            path = fh.name
        robot.send_custom_script_file(path)
        logger.success(f"Script file {path} sent.")

    finally:
        robot.disconnect()


def send_custom_script_function_example(ip: str = "192.168.1.2"):
    """Send a named URScript function."""
    logger.info("Running send_custom_script_function example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        robot.send_custom_script_function('my_func', 'textmsg("hello from function")')
        logger.success("Custom function sent.")

    finally:
        robot.disconnect()


def set_custom_script_file_example(ip: str = "192.168.1.2"):
    """Set the default URScript file path used by the controller wrapper."""
    logger.info("Running set_custom_script_file example...")
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.script', delete=False, mode='w') as fh:
            fh.write('# placeholder\n')
            path = fh.name
        robot.set_custom_script_file(path)
        logger.success(f"Custom script file path set to {path}.")

    finally:
        robot.disconnect()


def get_example_dict(ip: str = "192.168.1.2"):
    return {
        "send_custom_script": lambda: send_custom_script_example(ip),
        "send_custom_script_file": lambda: send_custom_script_file_example(ip),
        "send_custom_script_function": lambda: send_custom_script_function_example(ip),
        "set_custom_script_file": lambda: set_custom_script_file_example(ip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Custom Scripting Synapse examples")
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
