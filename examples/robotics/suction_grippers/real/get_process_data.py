"""
Demonstrates reading the decoded process data of a suction gripper pump.

Only supported Supports Piab grippers on MODBUS_RTU protocol.

Usage:
    python get_process_data.py --serial-port COM3
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import piab


def main(serial_port: str) -> None:
    """Reads the decoded process data of a Piab gripper pump."""

    #===================== Create Gripper ======================================
    gripper = piab.PiabPiCobotElectric()

    # ==================== Run Skill ===========================================
    try:
        gripper.connect(serial_port=serial_port, protocol="MODBUS_RTU")
        data = gripper.get_process_data()
        logger.success(f"Vacuum pressure: {data.vacuum_pressure_kpa} kPa")
        logger.success(f"Part present: {data.part_present}, "
                       f"part secured: {data.part_secured}")
        logger.success(f"Energy saving: {data.energy_saving}, "
                       f"atmospheric pressure: {data.atmospheric_pressure}, "
                       f"automated function complete: {data.automated_function_complete}")
        logger.success(f"Motor stall: {data.motor_stall}, "
                       f"membrane service warning: {data.membrane_service_warning}, "
                       f"hours to membrane service: {data.hours_to_membrane_service}")
        logger.success(f"PCB temperature: {data.pcb_temperature_c} °C")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Piab gripper get process data")
    p.add_argument("--serial-port", dest="serial_port", default="COM3",
                   help="Serial port for MODBUS_RTU")
    args = p.parse_args()

    main(serial_port=args.serial_port)
