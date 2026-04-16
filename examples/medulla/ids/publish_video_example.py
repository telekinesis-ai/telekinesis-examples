"""
A simple example demonstraing how to use the IDS camera with BabyROS to stream
live video.

Please first start the camera loop in examples/ids_example/camera_loop.py
before running this example. It will instantiate the IDS camera class, connect
to the camera and enter a loop to keep the instance alive and responsive to
requests from this script.

Please run this example from a terminal to avoid issues with rerun's spawn mode.
"""

import time

from loguru import logger
import rerun as rr
import numpy as np

from babyros import node


def log_video(data: np.ndarray) -> None:
    """
    Log video frames to the rerun viewer.
    """
    rr.log(
        "Contionous_Image_Capture",
        rr.EncodedImage(contents=data, media_type="image/jpeg"),
    )


def main():
    rr.init("IDS_Example", spawn=True)

    name = "my_ids_camera"

    base_topic = f"medulla/v1/camera/ids/IDS/{name}"

    start_video_stream_publisher = node.Publisher(
        topic=f"{base_topic}/start_video_stream"
    )

    stop_video_stream_publisher = node.Publisher(
        topic=f"{base_topic}/stop_video_stream"
    )

    video_subscriber = node.Subscriber(
        topic=f"{base_topic}/video",
        callback=log_video,
    )

    try:
        start_video_stream_publisher.publish(data={})
        time.sleep(10)
        stop_video_stream_publisher.publish(data={})
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        start_video_stream_publisher.delete()
        stop_video_stream_publisher.delete()
        video_subscriber.delete()
        node.SessionManager.delete()
        logger.info("Completed clenup.")


if __name__ == "__main__":
    main()
