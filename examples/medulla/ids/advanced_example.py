"""
Advanced example demonstraing how to use the IDS camera with BabyROS to capture
single image and stream a live video.

The IDS class defined in this script is a helper class for communicating with
the camera. It uses BabyROS clients and publishers to send requests.

Please first start the camera loop in examples/ids_example/camera_loop.py
before running this example. It will instantiate the IDS camera class, connect
to the camera and enter a loop to keep the instance alive and responsive to
requests from this script.

Please run this example from a terminal to avoid issues with rerun's spawn mode.
"""

from typing import Any
import time

from loguru import logger
import rerun as rr
import numpy as np

from babyros import node


class IDS:
    """
    Helper class for communicating with the IDS camera over BabyROS.
    """

    def __init__(self, name: str, serial_number: str) -> None:
        self._name = name
        self._serial_number = serial_number
        self._base_topic = "medulla/v1/camera/ids/IDS/" + self._name
        self._connection_client = node.Client(
            topic=self._base_topic + "/connect"
        )

        self._capture_client = node.Client(topic=f"{self._base_topic}/capture")

        self._parameter_client = node.Client(
            topic=f"{self._base_topic}/parameter"
        )

        self._start_video_stream_publisher = node.Publisher(
            topic=f"{self._base_topic}/start_video_stream"
        )

        self._stop_video_stream_publisher = node.Publisher(
            topic=f"{self._base_topic}/stop_video_stream"
        )

    def connect(self) -> dict:
        """
        Send a request to the server to connect to the camera.

        Returns:
            Server response dict with a 'status' key.
        """
        logger.info("Sending request to connect")
        response = self._connection_client.request(data={"connect": True})
        logger.info(f"Response from connect request: {response}")
        return response

    def disconnect(self) -> dict:
        """
        Send a request to the server to disconnect from the camera.

        Returns:
            Server response dict with a 'status' key.
        """
        logger.info("Sending request to disconnect")
        response = self._connection_client.request(data={"connect": False})
        logger.info(f"Response from disconnect request: {response}")
        return response

    def capture(self) -> np.ndarray:
        """
        Capture a single image from the camera server.

        Returns:
            The captured frame as a NumPy array.

        Raises:
            RuntimeError: If the server returns no image.
        """
        image = self._capture_client.request()

        if isinstance(image, list) and len(image) > 0:
            image = image[0]

        if image is None:
            raise RuntimeError("No image received from camera server.")

        return np.array(image)

    def start_video_stream(self) -> None:
        """
        Send a trigger to the server to start capturing and publishing video
        frames.
        """
        self._start_video_stream_publisher.publish(data={})

    def stop_video_stream(self) -> None:
        """
        Send a trigger to the server to stop capturing and publishing video frames.
        """
        self._stop_video_stream_publisher.publish(data={})

    def get_parameter(self, name: str) -> Any:
        """
        Get the value of a camera parameter.
        """
        logger.info("Sending request to connect")
        response = self._parameter_client.request(
            data={"mode": "get", "name": name}
        )
        logger.info(f"Response from connect request: {response}")
        return response[0]["value"]

    def set_parameter(self, name: str, value: Any) -> Any:
        """
        Set the value of a camera parameter.

        Args:
            name (str): The name of the parameter to set.
            value (Any): The value to set the parameter to.
        Returns:
            The value that was set, as returned by the server.
        """
        logger.info("Sending request to connect")
        response = self._parameter_client.request(
            data={"mode": "set", "name": name, "value": value}
        )
        logger.info(f"Response from connect request: {response}")
        return response[0]["value"]

    def delete(self) -> None:
        """
        Release all BabyROS nodes.
        """
        self._connection_client.delete()
        self._capture_client.delete()
        self._parameter_client.delete()
        self._start_video_stream_publisher.delete()
        self._stop_video_stream_publisher.delete()


def log_image(data: np.ndarray) -> None:
    """
    Log image to the rerun viewer.
    """
    rr.log("Single_Image_Capture", rr.Image(data))


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

    camera = None
    video_subscriber = None
    try:
        name = "my_ids_camera"
        camera = IDS(name=name, serial_number="4108909352")
        base_topic = f"medulla/v1/camera/ids/IDS/{name}"

        video_subscriber = node.Subscriber(
            topic=f"{base_topic}/video",
            callback=log_video,
        )

        logger.debug(
            f"AcquisitionFrameRate: {camera.get_parameter('AcquisitionFrameRate')}"
        )
        logger.debug(f"ExposureTime: {camera.get_parameter('ExposureTime')}")
        logger.debug(
            f"DeviceLinkThroughputLimit: {camera.get_parameter('DeviceLinkThroughputLimit')}"
        )

        camera.set_parameter("ExposureTime", 35000.0)
        image = camera.capture()
        log_image(image)
        camera.set_parameter("DeviceLinkThroughputLimit", 290000000)
        camera.set_parameter("AcquisitionFrameRate", 12.0)
        camera.set_parameter("ExposureTime", 60000.0)

        camera.start_video_stream()
        time.sleep(5)
        camera.stop_video_stream()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        if camera is not None:
            camera.delete()
        if video_subscriber is not None:
            video_subscriber.delete()
        node.SessionManager.delete()
        logger.info("Cleanup complete.")


if __name__ == "__main__":
    main()
