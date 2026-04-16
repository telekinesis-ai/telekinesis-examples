"""
A simple example demonstraing how to connect to an IDS camera, get/set
parameters and stream a live video.

Please run this example from a terminal to avoid issues with rerun's spawn mode.

You can set the camera parameters in two ways: either by using the IDS peak
Cockpit software or by using the get_parameter and set_parameter methods in the
code.
1) Using the IDS peak Cockpit software:
- Open the IDS peak Cockpit software and connect to your camera.
- Adjust the parameters you want to set (e.g., "ExposureTime",
"AcquisitionFrameRate", etc.).
- Close the IDS peak Cockpit software. The parameters you set will be persist as
long the camera remains connect or the PC is not restarted
- When you run the code, it will run with the parameters you set in the GUI

2) Using the get_parameter and set_parameter methods in the code:
- You can use the get_parameter method to read the current value of a parameter
- You can use the set_parameter method to set a new value for a parameter.
- Currently supported parameters in the IDS camera class in medulla
(medulla/cameras/ids.py) are listed in the class variable
'parameter_name_to_type_map' of the IDS class.
"""

import time
from loguru import logger

import cv2
import rerun as rr

from telekinesis.medulla.cameras import ids


def main():
    camera = ids.IDS(
        name="my_ids_camera",
        serial_number="4108909352",
        load_factory_defaults=False,
    )
    try:
        rr.init("IDS_Example", spawn=True)
        camera.connect()

        logger.info(
            f"AcquisitionFrameRate: {camera.get_parameter('AcquisitionFrameRate')}"
        )
        logger.info(f"ExposureTime: {camera.get_parameter('ExposureTime')}")
        logger.info(
            f"DeviceLinkThroughputLimit: {camera.get_parameter('DeviceLinkThroughputLimit')}"
        )

        camera.set_parameter("ExposureTime", 35000.0)
        camera.set_parameter("DeviceLinkThroughputLimit", 290000000)
        camera.set_parameter("AcquisitionFrameRate", 14.0)

        end_time = time.time() + 10
        while time.time() < end_time:
            image = camera.capture_video_color_frame()
            if image is not None:
                _, encoded = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                rr.log(
                    "Contionous_Image_Capture",
                    rr.EncodedImage(contents=encoded, media_type="image/jpeg"),
                )

        # Alternativly you can log the unencoded images with cv2
        # frame_count = 0
        # start_time = time.time()
        # fps = 0
        # while True:
        #     image = camera.capture_video_color_frame()
        #     if image is not None:
        #         frame_count += 1
        #         elapsed = time.time() - start_time
        #         if elapsed >= 1.0:
        #             fps = frame_count / elapsed
        #             logger.info(f"FPS: {fps:.2f}")
        #             frame_count = 0
        #             start_time = time.time()
        #         cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #         cv2.putText(image, "Press 'q' to exit", (10, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #         cv2.imshow('Captured Image', image)
        #     else:
        #         logger.warning("Image is None.")
        #     if cv2.waitKey(1) == ord('q'):
        #         break
        # cv2.destroyAllWindows()
    except Exception as e:
        logger.error(
            f"Unable to run video capture."
            f"Caught exception: {type(e).__name__}: {e}"
        )
    finally:
        camera.disconnect()


if __name__ == "__main__":
    main()
