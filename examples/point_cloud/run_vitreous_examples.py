"""Run all vitreous examples."""

import importlib.util
import pathlib
import sys
import time

from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<level>{level: <8}</level> | <level>{message}</level>")

# Helper names in each example module that we time separately. Everything else
# in the example function (chiefly the vitreous skill call) counts as "algorithm".
VISUALIZE_HELPER = "visualize"


def _instrument(module, accum):
    """Wrap the module's fetch/visualize helpers so their time accrues into `accum`."""

    def make_timer(key, fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                accum[key] += time.perf_counter() - start

        return wrapper

    if callable(getattr(module, VISUALIZE_HELPER, None)):
        setattr(
            module,
            VISUALIZE_HELPER,
            make_timer("visualize", getattr(module, VISUALIZE_HELPER)),
        )


def run_examples():
    """Discover and run all vitreous examples."""
    examples_dir = pathlib.Path(__file__).parent
    examples = sorted(
        [
            f
            for f in examples_dir.glob("*.py")
            if not f.name.startswith("_") and f.name != "run_vitreous_examples.py"
        ]
    )

    if not examples:
        logger.error("No examples found")
        return 1

    logger.info(f"Running {len(examples)} vitreous example(s)...")
    logger.info("=" * 60)

    successful = 0
    failed = 0
    timings = []

    for example_file in examples:
        example_name = example_file.stem
        accum = {"fetch": 0.0, "visualize": 0.0}
        start = time.perf_counter()
        try:
            logger.info(f"Running: {example_name}")

            # Load and execute the example
            spec = importlib.util.spec_from_file_location(example_name, example_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Time the fetch and visualize phases separately from the skill call
            _instrument(module, accum)

            # Find and call the example function
            for attr_name in dir(module):
                if attr_name.endswith("_example") and callable(
                    getattr(module, attr_name)
                ):
                    getattr(module, attr_name)()
                    break

            total = time.perf_counter() - start
            status = "PASS"
            logger.success(f"✓ {example_name} completed in {total:.2f}s")
            successful += 1
        except Exception as e:
            total = time.perf_counter() - start
            status = "FAIL"
            logger.error(f"✗ {example_name} failed in {total:.2f}s: {e}")
            failed += 1

        fetch = accum["fetch"]
        visualize = accum["visualize"]
        algorithm = max(total - fetch - visualize, 0.0)  # remainder = skill call
        timings.append((example_name, fetch, algorithm, visualize, total, status))
        logger.info("-" * 60)

    logger.info("=" * 60)
    logger.info(
        f"Summary: {successful} successful, {failed} failed out of {len(examples)}"
    )

    # Per-example time summary table (fetch / algorithm / visualize / total), slowest first
    name_width = max((len(name) for name, *_ in timings), default=7)
    name_width = max(name_width, len("Example"))
    cols = f"{'Fetch':>7} | {'Algo':>7} | {'Viz':>7} | {'Total':>7}"
    header = f"| {'Example':<{name_width}} | {cols} | {'Status':<6} |"
    rule = (
        f"|{'-' * (name_width + 2)}|{'-' * 9}|{'-' * 9}|{'-' * 9}|{'-' * 9}|{'-' * 8}|"
    )

    logger.info("Time summary in seconds (slowest first):")
    logger.info(rule)
    logger.info(header)
    logger.info(rule)
    tot_fetch = tot_algo = tot_viz = tot_total = 0.0
    for name, fetch, algorithm, visualize, total, status in sorted(
        timings, key=lambda t: t[4], reverse=True
    ):
        tot_fetch += fetch
        tot_algo += algorithm
        tot_viz += visualize
        tot_total += total
        row = f"{fetch:>7.2f} | {algorithm:>7.2f} | {visualize:>7.2f} | {total:>7.2f}"
        logger.info(f"| {name:<{name_width}} | {row} | {status:<6} |")
    logger.info(rule)
    total_row = (
        f"{tot_fetch:>7.2f} | {tot_algo:>7.2f} | {tot_viz:>7.2f} | {tot_total:>7.2f}"
    )
    logger.info(f"| {'TOTAL':<{name_width}} | {total_row} | {'':<6} |")
    logger.info(rule)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_examples())
