"""
Example script to demonstrate usage of Image datatype.
"""

import time

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def segmentation_image_example():
    """
    Example function to demonstrate usage of Image datatype.
     - Create an Image data
     - Print the original data
    """
    # Create an Image data
    input_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    logger.info(f"Input Image shape: {input_image.shape}, dtype: {input_image.dtype}")
    my_segmentation_image = datatypes.SegmentationImage(input_image)
    logger.info(f"Original SegmentationImageImage: {my_segmentation_image}")

    # Access the underlying image data
    my_segmentation_image_data = my_segmentation_image.data
    my_segmentation_image_label_codes = my_segmentation_image.label_codes
    my_segmentation_image_number_label_codes = my_segmentation_image.number_of_labels
    my_segmentation_image_shape = my_segmentation_image_data.shape
    my_segmentation_image_dtype = my_segmentation_image_data.dtype
    my_segmentation_image_height = my_segmentation_image.height
    my_segmentation_image_width = my_segmentation_image.width
    my_segmentation_image_compression = my_segmentation_image.compression
    my_segmentation_image_numpy_array = my_segmentation_image.to_numpy()

    logger.info(f"Underlying SegmentationImage data: {my_segmentation_image_data}")
    logger.info(f"Underlying SegmentationImage label codes: {my_segmentation_image_label_codes}")
    logger.info(
        f"Underlying SegmentationImage number of label codes: {my_segmentation_image_number_label_codes}"
    )
    logger.info(f"Underlying SegmentationImage shape: {my_segmentation_image_shape}")
    logger.info(f"Underlying SegmentationImage dtype: {my_segmentation_image_dtype}")
    logger.info(f"Underlying SegmentationImage height: {my_segmentation_image_height}")
    logger.info(f"Underlying SegmentationImage width: {my_segmentation_image_width}")
    logger.info(f"Underlying SegmentationImage compression: {my_segmentation_image_compression}")
    logger.info(f"Underlying SegmentationImage numpy array: {my_segmentation_image_numpy_array}")

    logger.info("Visualizing with Rerun...")
    rr.init("segmentation_image_example", spawn=True)
    datatypes.visualize(
        my_segmentation_image,
        entity_path="/SegmentationImage/my_segmentation_image",
    )

    # Update the image data
    updated_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    my_segmentation_image.data = updated_image
    logger.info("Visualize updated image with entity path /SegmentationImage...")
    datatypes.visualize(
        my_segmentation_image,
        entity_path="/SegmentationImage/my_updated_segmentation_image",
    )

    # Operate Image with numpy
    my_segmentation_image_mean = np.mean(my_segmentation_image)
    logger.info(f"Mean pixel value of the image: {my_segmentation_image_mean}")

    my_segmentation_image_flipped = np.flipud(my_segmentation_image)
    logger.info(
        f"Flipped Image shape: {my_segmentation_image_flipped.shape}, dtype: {my_segmentation_image_flipped.dtype}"
    )

    # Serialize and deserialize the image
    serialization_start_time = time.perf_counter()
    serialized_image = datatypes.serialize(my_segmentation_image)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized_image = datatypes.deserialize(serialized_image)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(
        f"Deserialize Image matches with original: {deserialized_image == my_segmentation_image}"
    )

    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    segmentation_image_example()
