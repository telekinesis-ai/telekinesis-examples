"""
A simple example demonstraing how to connect to a webcam camera, stream a video
and disconnect from the camera when finished.

Please run this example from a terminal to avoid issues with rerun's spawn mode.
"""

import time
from loguru import logger

import rerun as rr

from telekinesis.medulla.cameras import webcam


def main():
    camera = webcam.Webcam(
        name="my_webcam",
        camera_id=0,
    )
    try:
        rr.init("IDS_Example", spawn=True)
        camera.connect()

        end_time = time.time() + 10
        while time.time() < end_time:
            image = camera.capture_video_color_frame()
            rr.log("Contionous_Image_Capture", rr.Image(image))
    except Exception as e:
        logger.error(
            f"Unable to stream video.Caught exception: {type(e).__name__}: {e}"
        )
    finally:
        camera.disconnect()


if __name__ == "__main__":
    main()
