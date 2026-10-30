"""
Demonstrates how to visualize a spot-welding gun in Rerun for the Synapse SDK.

Supports Isaac Sim assets (companion mesh).

Usage:
    python visualize_rerun.py
"""

import rerun as rr

from telekinesis.synapse.tools.welding_guns import isaacsim


def main() -> None:
    """Visualizes a spot-welding gun in Rerun at the world origin."""

    #===================== Create Gun ==========================================
    gun = isaacsim.SpotWeldingGun()

    # =================== Visualization =========================================
    # USD tools upload their companion mesh once as a static asset.
    rr.init(f"telekinesis_synapse_{type(gun).__name__}", spawn=True)
    gun.visualize_rerun(recording_stream=rr.get_global_data_recording())


if __name__ == "__main__":
    main()
