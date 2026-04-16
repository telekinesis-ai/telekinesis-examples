"""
A simple example demonstraing how to connect to an IDS camera, get/set
parameters, capture and image and disconnect from the camera when finished.

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

from loguru import logger
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

        logger.info(f"ExposureTime: {camera.get_parameter('ExposureTime')}")
        camera.set_parameter("ExposureTime", 35000.0)

        image = camera.capture_single_color_frame()

        rr.log("Single_Image_Capture", rr.Image(image))
    except Exception as e:
        logger.error(
            f"Unable to load capture image."
            f"Caught exception: {type(e).__name__}: {e}"
        )
    finally:
        camera.disconnect()


if __name__ == "__main__":
    main()
