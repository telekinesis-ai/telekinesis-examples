"""Run all cornea examples."""

import importlib.util
import pathlib
import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<level>{level: <8}</level> | <level>{message}</level>")


def run_examples():
    """Discover and run all cornea examples."""
    examples_dir = pathlib.Path(__file__).parent
    examples = sorted([f for f in examples_dir.glob("*.py") if not f.name.startswith("_") and f.name != "run_cornea_examples.py"])

    if not examples:
        logger.error("No examples found")
        return 1

    logger.info(f"Running {len(examples)} cornea example(s)...")
    logger.info("=" * 60)

    successful = 0
    failed = 0

    for example_file in examples:
        try:
            example_name = example_file.stem
            logger.info(f"Running: {example_name}")

            # Load and execute the example
            spec = importlib.util.spec_from_file_location(example_name, example_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find and call the example function
            for attr_name in dir(module):
                if attr_name.endswith("_example") and callable(getattr(module, attr_name)):
                    getattr(module, attr_name)()
                    break

            logger.success(f"✓ {example_name} completed")
            successful += 1
        except Exception as e:
            logger.error(f"✗ {example_name} failed: {e}")
            failed += 1

        logger.info("-" * 60)

    logger.info("=" * 60)
    logger.info(f"Summary: {successful} successful, {failed} failed out of {len(examples)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_examples())
