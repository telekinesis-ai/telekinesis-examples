"""
Example script to demonstrate usage of ImageBatch datatype.
"""

import time
from pathlib import Path

import numpy as np
from loguru import logger
import rerun as rr

from telekinesis import datatypes


def image_batch_example():
    """
    Example function to demonstrate usage of ImageBatch datatype.
        - Create an ImageBatch data
        - Access the underlying image batch data
        - Visualize the ImageBatch data using Rerun
        - Access a particular image from the batch
        - Operate on the indexed image
        - Serialize and deserialize the ImageBatch
    """
    # Create an ImageBatch data
    
    input_image_1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    ROOT_PATH = Path(__file__).parent
    input_image_2 = datatypes.Image.from_path(ROOT_PATH / "data/sample.jpg").to_numpy()
    images = [
        input_image_1,
        input_image_2,
    ]
    my_image_batch = datatypes.ImageBatch(images)
    logger.info(f"Original ImageBatch: {my_image_batch}")

    # Access the underlying image batch data
    my_image_batch_dtype = my_image_batch.dtypes
    my_image_batch_shape = my_image_batch.shapes
    my_image_batch_compression = my_image_batch.compressions
    my_image_batch_numpy = my_image_batch.to_numpy()

    logger.info(f"Underlying ImageBatch dtype: {my_image_batch_dtype}")
    logger.info(f"Underlying ImageBatch shapes: {my_image_batch_shape}")
    logger.info(f"Underlying ImageBatch compression: {my_image_batch_compression}")
    logger.info(f"Underlying ImageBatch numpy array: {my_image_batch_numpy}")

    # Visualize the ImageBatch with Rerun
    logger.info("Visualizing all images..")
    rr.init("image_batch_example", spawn=True)
    datatypes.visualize(my_image_batch, entity_path="/ImageBatch")

    # Access a particular image from the batch
    index = 1
    # Returns the image at the specified index as a new Image object
    my_image_1_from_batch = my_image_batch[index]
    logger.info(f"Image at index {index} from ImageBatch: {my_image_1_from_batch}")
    datatypes.visualize(my_image_1_from_batch, entity_path="/ImageBatch/Image_1")

    # Operate on the indexed image which is a new Image object
    # This does not change the original ImageBatch data
    gray_image = my_image_1_from_batch.to_grayscale()
    logger.info(f"Grayscale Image at index {index} from ImageBatch: {gray_image}")
    datatypes.visualize(gray_image, entity_path="/ImageBatch/Image_1/Grayscale")
    gray_image.save_to_path(ROOT_PATH / "data/grayscale_image.jpg")
    logger.info("Grayscale image saved to data/grayscale_image.jpg")

    # ImageBatch is immutable after construction -- there is no way to
    # replace an image in place. To change an image, build a new ImageBatch
    # with the updated image in place of the one you want to replace.
    index = 0
    updated_image = np.random.randint(0, 255, (1907, 512, 3), dtype=np.uint8)
    images[index] = updated_image
    my_image_batch = datatypes.ImageBatch(images)
    logger.info(f"Rebuilt ImageBatch at index 0: {my_image_batch}")

    # Serialize and deserialize the ImageBatch
    serialization_start_time = time.perf_counter()
    serialized_image_batch = datatypes.serialize(my_image_batch)
    serialization_end_time = time.perf_counter()

    deserialization_start_time = time.perf_counter()
    deserialized_image_batch = datatypes.deserialize(serialized_image_batch)["param_0"]
    deserialization_end_time = time.perf_counter()
    logger.info(
        f"Deserialize ImageBatch matches with original: {deserialized_image_batch == my_image_batch}"
    )
    logger.info(
        f"Serialization time: {(serialization_end_time - serialization_start_time) * 1000} ms"
    )
    logger.info(
        f"Deserialization time: {(deserialization_end_time - deserialization_start_time) * 1000} ms"
    )


if __name__ == "__main__":
    image_batch_example()
