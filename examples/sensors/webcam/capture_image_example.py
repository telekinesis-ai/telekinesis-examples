"""
A simple example demonstraing how to connect to a webcam camera, capture an
image and disconnect from the camera when finished.

Please run this example from a terminal to avoid issues with rerun's spawn mode.
"""

from loguru import logger

import rerun as rr

from telekinesis.medulla.cameras import webcam


def main():
    camera = webcam.Webcam(name="my_webcam")
    try:
        rr.init("IDS_Example", spawn=True)
        camera.connect(camera_id=0)

        image = camera.capture_image()

        rr.log("Single_Image_Capture", rr.Image(image.to_numpy()))
    except Exception as e:
        logger.error(
            f"Unable to capture image.Caught exception: {type(e).__name__}: {e}"
        )
    finally:
        camera.disconnect()


if __name__ == "__main__":
    main()
