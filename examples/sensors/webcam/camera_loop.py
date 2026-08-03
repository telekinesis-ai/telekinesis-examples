"""
Camera loop for the Webcam.
"""

import time

from loguru import logger

from babyros import node
from telekinesis.medulla.cameras import webcam


def main():
    camera = None
    try:
        camera = webcam.Webcam(name="my_webcam", camera_id=0)
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
