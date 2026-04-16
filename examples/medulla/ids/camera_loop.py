"""
Camera loop for the IDS camera.
"""

import time

from loguru import logger

from babyros import node
from telekinesis.medulla.cameras import ids


def main():
    camera = None
    try:
        camera = ids.IDS(name="my_ids_camera", serial_number="4108909352")
        camera.connect()
        logger.info("Camera Server is running... Press Ctrl+C to stop.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        if camera is not None:
            camera.disconnect()
            node.SessionManager.delete()
        logger.info("Completed clenup.")


if __name__ == "__main__":
    main()
