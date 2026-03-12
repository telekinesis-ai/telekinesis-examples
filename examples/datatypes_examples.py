"""Example usage of custom datatypes for serialization and deserialization.

This script demonstrates how to use the custom datatypes defined in the library.
Data is accessed via to_dict(), to_list(), or to_numpy() where available.
"""

import numpy as np
import argparse
from loguru import logger
import difflib
import pathlib

from datatypes import datatypes
from datatypes.datatypes import PartState

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent


def bool_example():
    """
    Example for Bool datatype.

    Demonstrates:
    - constructing a boolean value
    - accessing the value 
    """
    # ------------------------------------------------
    # 1. Create Bool instance
    # ------------------------------------------------
    bool_val = datatypes.Bool(True)

    # ------------------------------------------------
    # 2. Access value
    # ------------------------------------------------
    logger.info("Bool value={}", bool_val.value)


def float_example():
    """
    Example for Float datatype.

    Demonstrates:
    - constructing a float value
    - accessing the value 
    """
    # ------------------------------------------------
    # 1. Create Float instance
    # ------------------------------------------------
    float_val = datatypes.Float(3.14)

    # ------------------------------------------------
    # 2. Access value
    # ------------------------------------------------
    logger.info("Float value={}", float_val.value)


def int_example():
    """
    Example for Int datatype.

    Demonstrates:
    - constructing an integer value
    - accessing the value 
    """
    # ------------------------------------------------
    # 1. Create Int instance
    # ------------------------------------------------
    int_val = datatypes.Int(42)

    # ------------------------------------------------
    # 2. Access value
    # ------------------------------------------------
    logger.info("Int value={}", int_val.value)


def imageformat_example():
    """
    Example for ImageFormat datatype.

    Demonstrates:
    - constructing an image format with dimensions and metadata
    - accessing format attributes 
    """
    # ------------------------------------------------
    # 1. Create ImageFormat instance
    # ------------------------------------------------
    image_format = datatypes.ImageFormat(
        width=2,
        height=2,
        pixel_format=1,
        channel_datatype=0,
        color_model=1,
    )

    # ------------------------------------------------
    # 2. Access format attributes
    # ------------------------------------------------
    logger.info("ImageFormat width={}", image_format.width)
    logger.info("ImageFormat height={}", image_format.height)
    logger.info("ImageFormat pixel_format={}", image_format.pixel_format)
    logger.info("ImageFormat channel_datatype={}", image_format.channel_datatype)
    logger.info("ImageFormat color_model={}", image_format.color_model)


def list_of_images_example():
    """
    Example for ListOfImages datatype.

    Demonstrates:
    - constructing a list of Image instances
    - accessing images via to_list
    - accessing pixel data via to_numpy on each Image
    """
    # ------------------------------------------------
    # 1. Create ListOfImages instance
    # ------------------------------------------------
    image_rgba = np.array([
        [[255, 0, 0, 255], [0, 255, 0, 255]],
        [[0, 0, 255, 255], [255, 255, 0, 255]],
    ], dtype=np.uint8)
    image_bgr = np.array([
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 255, 0]],
    ], dtype=np.uint8)
    img_1 = datatypes.Image(image=image_rgba)
    img_2 = datatypes.Image(image=image_bgr)
    list_of_images = datatypes.ListOfImages(image_list=[img_1, img_2])

    # ------------------------------------------------
    # 2. Access images via to_list
    # ------------------------------------------------
    images_list = list_of_images.to_list()
    logger.info("ListOfImages to_list num_images={}", len(images_list))
    logger.info("ListOfImages to_list first shape={}", images_list[0].to_numpy().shape)
    logger.info("ListOfImages to_list second shape={}", images_list[1].to_numpy().shape)


def image_example():
    """
    Example for Image datatype.

    Demonstrates:
    - constructing an Image from a numpy array (RGBA and BGR)
    - accessing pixel data via to_numpy
    - accessing format attributes 
    """
    # ------------------------------------------------
    # 1. Create Image instance (RGBA)
    # ------------------------------------------------
    image_rgba = np.array([
        [[255, 0, 0, 255], [0, 255, 0, 255]],
        [[0, 0, 255, 255], [255, 255, 0, 255]],
    ], dtype=np.uint8)
    img_rgba = datatypes.Image(image=image_rgba)

    # ------------------------------------------------
    # 2. Access data via to_numpy (RGBA case)
    # ------------------------------------------------
    numpy_array = img_rgba.to_numpy()
    logger.info("Image (RGBA) to_numpy shape={}", numpy_array.shape)
    logger.info("Image (RGBA) to_numpy dtype={}", numpy_array.dtype)
    logger.info("Image (RGBA) width={}", img_rgba.width)
    logger.info("Image (RGBA) height={}", img_rgba.height)
    logger.info("Image (RGBA) color_model={}", img_rgba.color_model)
    logger.info("Image (RGBA) channel_datatype={}", img_rgba.channel_datatype)

    # ------------------------------------------------
    # 3. Create Image instance (BGR)
    # ------------------------------------------------
    image_bgr = np.array([
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 255, 0]],
    ], dtype=np.uint8)
    img_bgr = datatypes.Image(image=image_bgr)

    # ------------------------------------------------
    # 4. Access data via to_numpy (BGR case)
    # ------------------------------------------------
    numpy_array_bgr = img_bgr.to_numpy()
    logger.info("Image (BGR) to_numpy shape={}", numpy_array_bgr.shape)
    logger.info("Image (BGR) to_numpy dtype={}", numpy_array_bgr.dtype)
    logger.info("Image (BGR) width={}", img_bgr.width)
    logger.info("Image (BGR) height={}", img_bgr.height)


def boxes3d_example():
    """
    Example for Boxes3D datatype.

    Demonstrates:
    - constructing 3D bounding boxes with half_sizes, centers, rotations
    - accessing data via to_dict
    """
    # ------------------------------------------------
    # 1. Create Boxes3D instance
    # ------------------------------------------------
    boxes_3d = datatypes.Boxes3D(
        half_sizes=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        centers=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        rotations_in_euler_angle=[[0.0, 0.0, 0.0]],
    )

    # ------------------------------------------------
    # 2. Access data via to_dict
    # ------------------------------------------------
    data = boxes_3d.to_dict()
    logger.info("Boxes3D to_dict half_sizes={}", data["half_sizes"])
    logger.info("Boxes3D to_dict centers={}", data["centers"])
    logger.info("Boxes3D to_dict rotations_in_euler_angle={}", data["rotations_in_euler_angle"])


def boxes2d_example():
    """
    Example for Boxes2D datatype.

    Demonstrates:
    - constructing 2D bounding boxes with half_sizes and centers
    - accessing data via to_dict
    """
    # ------------------------------------------------
    # 1. Create Boxes2D instance
    # ------------------------------------------------
    boxes_2d = datatypes.Boxes2D(
        half_sizes=[[1, 2], [3, 4]],
        centers=[[0.0, 0.0], [1.0, 1.0]],
    )

    # ------------------------------------------------
    # 2. Access data via to_dict
    # ------------------------------------------------
    data = boxes_2d.to_dict()
    logger.info("Boxes2D to_dict half_sizes={}", data["half_sizes"])
    logger.info("Boxes2D to_dict centers={}", data["centers"])


def mesh3d_example():
    """
    Example for Mesh3D datatype.

    Demonstrates:
    - constructing a 3D mesh with vertices, triangles, normals, and colors
    - accessing mesh attributes 
    """
    # ------------------------------------------------
    # 1. Create Mesh3D instance
    # ------------------------------------------------
    vertex_positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangle_indices = np.array([[0, 1, 2]], dtype=np.int32)
    vertex_normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    vertex_colors = np.array(
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8
    )
    mesh_3d = datatypes.Mesh3D(
        vertex_positions, triangle_indices, vertex_normals, vertex_colors
    )

    # ------------------------------------------------
    # 2. Access mesh attributes
    # ------------------------------------------------
    logger.info("Mesh3D vertex_positions shape={}", mesh_3d.vertex_positions.shape)
    logger.info("Mesh3D triangle_indices shape={}", mesh_3d.triangle_indices.shape)
    logger.info("Mesh3D vertex_normals shape={}", mesh_3d.vertex_normals.shape)
    logger.info("Mesh3D vertex_colors shape={}", mesh_3d.vertex_colors.shape if mesh_3d.vertex_colors is not None else None)

def points3d_example():
    """
    Example for Points3D datatype.

    Demonstrates:
    - constructing a 3D point cloud
    - accessing attributes via to_numpy
    """

    # ------------------------------------------------
    # 1. Create Points3D instance
    # ------------------------------------------------

    input_positions = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float32)

    # Optional
    input_colors = np.array([
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ], dtype=np.uint8)

    input_normals = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ], dtype=np.float32)

    points = datatypes.Points3D(
        positions=input_positions,
        normals=input_normals,
        colors=input_colors,
    )

    # ------------------------------------------------
    # 2. Access data via to_numpy
    # ------------------------------------------------

    positions = points.to_numpy("positions")
    colors = points.to_numpy("colors")
    normals = points.to_numpy("normals")

    logger.info("Points3D to_numpy positions shape={}", positions.shape)
    logger.info("Points3D to_numpy colors shape={}", colors.shape if colors is not None else None)
    logger.info("Points3D to_numpy normals shape={}", normals.shape if normals is not None else None)


def vector2d_example():
    """
    Example for Vector2D datatype.

    Demonstrates:
    - constructing a 2D vector
    - accessing data via to_list and to_numpy
    """
    # ------------------------------------------------
    # 1. Create Vector2D instance
    # ------------------------------------------------
    xy = np.array([1.0, 2.0], dtype=np.float32)
    vec_2d = datatypes.Vector2D(xy)

    # ------------------------------------------------
    # 2. Access data via to_list and to_numpy
    # ------------------------------------------------
    data_list = vec_2d.to_list()
    logger.info("Vector2D to_list x={}", data_list[0])
    logger.info("Vector2D to_list y={}", data_list[1])
    logger.info("Vector2D to_numpy shape={}", vec_2d.to_numpy().shape)


def vector3d_example():
    """
    Example for Vector3D datatype.

    Demonstrates:
    - constructing a 3D vector
    - accessing data via to_list and to_numpy
    """
    # ------------------------------------------------
    # 1. Create Vector3D instance
    # ------------------------------------------------
    xyz = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vec_3d = datatypes.Vector3D(xyz)

    # ------------------------------------------------
    # 2. Access data via to_list and to_numpy
    # ------------------------------------------------
    data_list = vec_3d.to_list()
    logger.info("Vector3D to_list x={}", data_list[0])
    logger.info("Vector3D to_list y={}", data_list[1])
    logger.info("Vector3D to_list z={}", data_list[2])
    logger.info("Vector3D to_numpy shape={}", vec_3d.to_numpy().shape)


def vector4d_example():
    """
    Example for Vector4D datatype.

    Demonstrates:
    - constructing a 4D vector
    - accessing data via to_list and to_numpy
    """
    # ------------------------------------------------
    # 1. Create Vector4D instance
    # ------------------------------------------------
    xyzw = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    vec_4d = datatypes.Vector4D(xyzw)

    # ------------------------------------------------
    # 2. Access data via to_list and to_numpy
    # ------------------------------------------------
    data_list = vec_4d.to_list()
    logger.info("Vector4D to_list x={}", data_list[0])
    logger.info("Vector4D to_list y={}", data_list[1])
    logger.info("Vector4D to_list z={}", data_list[2])
    logger.info("Vector4D to_list w={}", data_list[3])
    logger.info("Vector4D to_numpy shape={}", vec_4d.to_numpy().shape)


def position2d_example():
    """
    Example for Position2D datatype.

    Demonstrates:
    - constructing a 2D position (point in space)
    - accessing data via to_list and to_numpy
    """
    # ------------------------------------------------
    # 1. Create Position2D instance
    # ------------------------------------------------
    xy = np.array([10.0, 20.0], dtype=np.float32)
    pos_2d = datatypes.Position2D(xy)

    # ------------------------------------------------
    # 2. Access data via to_list and to_numpy
    # ------------------------------------------------
    data_list = pos_2d.to_list()
    logger.info("Position2D to_list x={}", data_list[0])
    logger.info("Position2D to_list y={}", data_list[1])
    logger.info("Position2D to_numpy shape={}", pos_2d.to_numpy().shape)


def position3d_example():
    """
    Example for Position3D datatype.

    Demonstrates:
    - constructing a 3D position (point in space)
    - accessing data via to_list and to_numpy
    """
    # ------------------------------------------------
    # 1. Create Position3D instance
    # ------------------------------------------------
    xyz = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pos_3d = datatypes.Position3D(xyz)

    # ------------------------------------------------
    # 2. Access data via to_list and to_numpy
    # ------------------------------------------------
    data_list = pos_3d.to_list()
    logger.info("Position3D to_list x={}", data_list[0])
    logger.info("Position3D to_list y={}", data_list[1])
    logger.info("Position3D to_list z={}", data_list[2])
    logger.info("Position3D to_numpy shape={}", pos_3d.to_numpy().shape)


def mat4x4_example():
    """
    Example for Mat4X4 datatype.

    Demonstrates:
    - constructing a 4x4 matrix
    - accessing matrix via to_list and matrix attribute
    """
    # ------------------------------------------------
    # 1. Create Mat4X4 instance
    # ------------------------------------------------
    matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    mat_4x4 = datatypes.Mat4X4(matrix)

    # ------------------------------------------------
    # 2. Access matrix via to_list and to_numpy
    # ------------------------------------------------
    matrix_list = mat_4x4.to_list()
    logger.info("Mat4X4 to_list row0={}", matrix_list[0])
    logger.info("Mat4X4 to_list row1={}", matrix_list[1])
    logger.info("Mat4X4 to_list row2={}", matrix_list[2])
    logger.info("Mat4X4 to_list row3={}", matrix_list[3])
    logger.info("Mat4X4 to_numpy shape={}", mat_4x4.to_numpy().shape)


def mat3x3_example():
    """
    Example for Mat3X3 datatype.

    Demonstrates:
    - constructing a 3x3 matrix
    - accessing matrix via to_list and matrix attribute
    """
    # ------------------------------------------------
    # 1. Create Mat3X3 instance
    # ------------------------------------------------
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    mat_3x3 = datatypes.Mat3X3(matrix)

    # ------------------------------------------------
    # 2. Access matrix via to_list and to_numpy
    # ------------------------------------------------
    matrix_list = mat_3x3.to_list()
    logger.info("Mat3X3 to_list row0={}", matrix_list[0])
    logger.info("Mat3X3 to_list row1={}", matrix_list[1])
    logger.info("Mat3X3 to_list row2={}", matrix_list[2])
    logger.info("Mat3X3 to_numpy shape={}", mat_3x3.to_numpy().shape)


def array_example():
    """
    Example for Array datatype (arbitrary ndarray).

    Demonstrates:
    - constructing Array from 1D, 2D, and 3D numpy arrays
    - accessing data via to_numpy and to_list after each case
    """
    # ------------------------------------------------
    # 1. Create Array instance (1D)
    # ------------------------------------------------
    arr_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    array_1d = datatypes.Array(arr_1d)

    # ------------------------------------------------
    # 2. Access data via to_numpy and to_list (1D case)
    # ------------------------------------------------
    data = array_1d.to_numpy()
    logger.info("Array (1D) to_numpy shape={}", data.shape)
    logger.info("Array (1D) to_numpy dtype={}", data.dtype)
    logger.info("Array (1D) to_list={}", array_1d.to_list())

    # ------------------------------------------------
    # 3. Create Array instance (2D)
    # ------------------------------------------------
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    array_2d = datatypes.Array(arr_2d)

    # ------------------------------------------------
    # 4. Access data via to_numpy and to_list (2D case)
    # ------------------------------------------------
    numpy_arr = array_2d.to_numpy()
    logger.info("Array (2D) to_numpy shape={}", numpy_arr.shape)
    logger.info("Array (2D) to_numpy dtype={}", numpy_arr.dtype)
    logger.info("Array (2D) to_list={}", array_2d.to_list())

    # ------------------------------------------------
    # 5. Create Array instance (3D)
    # ------------------------------------------------
    arr_3d = np.random.randn(2, 3, 4).astype(np.float32)
    array_3d = datatypes.Array(arr_3d)

    # ------------------------------------------------
    # 6. Access data via to_numpy and to_list (3D case)
    # ------------------------------------------------
    data_3d = array_3d.to_numpy()
    logger.info("Array (3D) to_numpy shape={}", data_3d.shape)
    logger.info("Array (3D) to_numpy dtype={}", data_3d.dtype)
    logger.info("Array (3D) to_list len={}", len(array_3d.to_list()))


def float32_example():
    """
    Example for Float datatype (float32).

    Demonstrates:
    - constructing a float value
    - accessing the value 
    """
    # ------------------------------------------------
    # 1. Create Float instance
    # ------------------------------------------------
    float_val = datatypes.Float(2.718)

    # ------------------------------------------------
    # 2. Access value
    # ------------------------------------------------
    logger.info("Float value={}", float_val.value)


def channeldatatype_example():
    """
    Example for ChannelDatatype enumeration.

    Demonstrates:
    - accessing enum name and value
    """
    # ------------------------------------------------
    # 1. Access ChannelDatatype enum
    # ------------------------------------------------
    channel_dtype = datatypes.ChannelDatatype.U8

    # ------------------------------------------------
    # 2. Access enum attributes 
    # ------------------------------------------------
    logger.info("ChannelDatatype name={}", channel_dtype.name)
    logger.info("ChannelDatatype value={}", channel_dtype.value)


def colormodel_example():
    """
    Example for ColorModel enumeration.

    Demonstrates:
    - accessing enum name, value, and num_channels
    """
    # ------------------------------------------------
    # 1. Access ColorModel enum
    # ------------------------------------------------
    color_model = datatypes.ColorModel.RGB

    # ------------------------------------------------
    # 2. Access enum attributes 
    # ------------------------------------------------
    logger.info("ColorModel name={}", color_model.name)
    logger.info("ColorModel value={}", color_model.value)
    logger.info("ColorModel num_channels={}", color_model.num_channels())


def points2d_example():
    """
    Example for Points2D datatype.

    Demonstrates:
    - constructing a 2D point cloud with positions and colors
    - accessing positions and colors 
    """
    # ------------------------------------------------
    # 1. Create Points2D instance
    # ------------------------------------------------
    positions = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0, 255], [0, 255, 0, 255]], dtype=np.uint8)
    points_2d = datatypes.Points2D(positions, colors=colors)

    # ------------------------------------------------
    # 2. Access positions and colors
    # ------------------------------------------------
    logger.info("Points2D positions shape={}", points_2d.positions.shape)
    logger.info("Points2D colors shape={}", points_2d.colors.shape if points_2d.colors is not None else None)


def linestrips2d_example():
    """
    Example for LineStrips2D datatype.

    Demonstrates:
    - constructing line strips from multiple strips
    - accessing strips via to_dict
    """
    # ------------------------------------------------
    # 1. Create LineStrips2D instance
    # ------------------------------------------------
    strip_1 = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 0.0]], dtype=np.float32
    )
    strip_2 = np.array(
        [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    strip_3 = np.array(
        [[4.0, 0.0], [4.5, 0.5], [5.0, 0.0], [5.5, 0.5], [6.0, 0.0]],
        dtype=np.float32,
    )
    line_strips_2d = datatypes.LineStrips2D(strips=[strip_1, strip_2, strip_3])

    # ------------------------------------------------
    # 2. Access strips via to_dict
    # ------------------------------------------------
    strips_dict = line_strips_2d.to_dict()
    strips_list = strips_dict["strips"]
    logger.info("LineStrips2D to_dict num_strips={}", len(strips_list))
    logger.info("LineStrips2D to_dict strip_0 shape={}", strips_list[0].shape)
    logger.info("LineStrips2D to_dict strip_1 shape={}", strips_list[1].shape)
    logger.info("LineStrips2D to_dict strip_2 shape={}", strips_list[2].shape)


def string_example():
    """
    Example for String datatype.

    Demonstrates:
    - constructing a string value
    - accessing the value 
    """
    # ------------------------------------------------
    # 1. Create String instance
    # ------------------------------------------------
    string_val = datatypes.String("Hello, Telekinesis!")

    # ------------------------------------------------
    # 2. Access value
    # ------------------------------------------------
    logger.info("String value={}", string_val.value)


def rgba32_example():
    """
    Example for Rgba32 datatype.

    Demonstrates:
    - constructing RGBA colors
    - accessing via to_numpy and int conversion
    - using Rgba32 with Points3D
    """
    # ------------------------------------------------
    # 1. Create Rgba32 instances
    # ------------------------------------------------
    red = datatypes.Rgba32([255, 0, 0, 255])
    green = datatypes.Rgba32([0, 255, 0, 255])
    blue = datatypes.Rgba32([0, 0, 255, 255])

    # ------------------------------------------------
    # 2. Access via to_numpy and int conversion
    # ------------------------------------------------
    logger.info("Rgba32 red to_numpy={}", red.to_numpy())
    logger.info("Rgba32 green to_numpy={}", green.to_numpy())
    logger.info("Rgba32 blue to_numpy={}", blue.to_numpy())
    logger.info("Rgba32 int(red)={}", int(red))

    # ------------------------------------------------
    # 3. Use Rgba32 with Points3D
    # ------------------------------------------------
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
    )
    points_3d = datatypes.Points3D(positions=positions, colors=[red, green, blue])
    logger.info("Points3D with Rgba32 to_numpy positions shape={}", points_3d.to_numpy("positions").shape)
    logger.info("Points3D with Rgba32 to_numpy colors shape={}", points_3d.to_numpy("colors").shape if points_3d.to_numpy("colors") is not None else None)


def points3d_list_example():
    """
    Example for ListOfPoints3D datatype.

    Demonstrates:
    - constructing a list of Points3D instances
    - accessing point clouds via to_list
    """
    # ------------------------------------------------
    # 1. Create ListOfPoints3D instance
    # ------------------------------------------------
    point_1 = datatypes.Points3D(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.array([[255, 0, 0, 255], [0, 255, 0, 255]], dtype=np.uint8),
    )
    point_2 = datatypes.Points3D(
        positions=np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32),
        colors=np.array([[0, 0, 255, 255], [255, 255, 0, 255]], dtype=np.uint8),
    )
    list_of_points_3d = datatypes.ListOfPoints3D(point3d_list=[point_1, point_2])

    # ------------------------------------------------
    # 2. Access point clouds via to_list
    # ------------------------------------------------
    points_list = list_of_points_3d.to_list()
    logger.info("ListOfPoints3D to_list num_items={}", len(points_list))
    logger.info("ListOfPoints3D to_list first positions shape={}", points_list[0].to_numpy("positions").shape)
    logger.info("ListOfPoints3D to_list second positions shape={}", points_list[1].to_numpy("positions").shape)


def categories_example():
    """
    Example for Categories datatype.

    Demonstrates:
    - constructing a list of category dicts
    - accessing categories via to_list
    """
    # ------------------------------------------------
    # 1. Create Categories instance
    # ------------------------------------------------
    categories = datatypes.Categories(categories=[
        {
            "id": 0,
            "name": "person",
            "supercategory": "human",
            "color": [220, 85, 96],
            "isthing": 1,
        },
        {
            "id": 1,
            "name": "bicycle",
            "supercategory": "vehicle",
            "color": [0, 255, 0],
            "isthing": 1,
        },
    ])

    # ------------------------------------------------
    # 2. Access categories via to_list
    # ------------------------------------------------
    category_list = categories.to_list()
    logger.info("Categories to_list len={}", len(category_list))
    logger.info("Categories to_list first id={}", category_list[0]["id"])
    logger.info("Categories to_list first name={}", category_list[0]["name"])
    logger.info("Categories to_list second name={}", category_list[1]["name"])


def object_detection_annotations_example():
    """
    Example for ObjectDetectionAnnotations datatype.

    Demonstrates:
    - constructing object detection annotations
    - accessing annotations via to_list
    """
    # ------------------------------------------------
    # 1. Create ObjectDetectionAnnotations instance
    # ------------------------------------------------
    annotations = datatypes.ObjectDetectionAnnotations(annotations=[
        {
            "id": 0,
            "image_id": 1,
            "category_id": 2,
            "segmentation": [[1, 2, 3], [1, 1]],
            "bbox": [0, 0, 0, 0],
            "area": 0,
            "iscrowd": 0,
            "score": None,
        },
        {
            "id": 1,
            "image_id": 1,
            "category_id": 3,
            "segmentation": [[4, 5, 6], [1, 1]],
            "bbox": [1, 1, 1, 1],
            "area": 1,
            "iscrowd": 0,
        },
    ])

    # ------------------------------------------------
    # 2. Access annotations via to_list
    # ------------------------------------------------
    annotation_list = annotations.to_list()
    logger.info("ObjectDetectionAnnotations to_list len={}", len(annotation_list))
    logger.info("ObjectDetectionAnnotations to_list first id={}", annotation_list[0]["id"])
    logger.info("ObjectDetectionAnnotations to_list first image_id={}", annotation_list[0]["image_id"])
    logger.info("ObjectDetectionAnnotations to_list first category_id={}", annotation_list[0]["category_id"])


def geometry_detection_annotations_example():
    """
    Example for GeometryDetectionAnnotations datatype.

    Demonstrates:
    - constructing geometry detection annotations (circle, ellipse, contour)
    - accessing annotations via to_list
    - handling None values in optional fields
    """
    # ------------------------------------------------
    # 1. Create GeometryDetectionAnnotations instance
    # ------------------------------------------------
    annotations = datatypes.GeometryDetectionAnnotations(annotations=[
        {
            "id": 0,
            "image_id": 0,
            "geometry": {"center": [0, 0], "radius": 0},
            "geometry_shape": "circle",
            "category_id": 1,
            "segmentation": [],
            "bbox": None,
            "area": 0,
            "iscrowd": 0,
        },
        {
            "id": 1,
            "image_id": 0,
            "geometry": {"center": [0, 0], "axes": [0, 0], "angle": 0},
            "geometry_shape": "ellipse",
            "category_id": 2,
            "segmentation": [],
            "bbox": None,
            "area": 0,
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 0,
            "geometry": {"points": [[0, 0], [1, 1], [2, 2]]},
            "geometry_shape": "contour",
            "category_id": 3,
            "segmentation": [],
            "bbox": None,
            "area": 0,
            "iscrowd": 0,
        },
    ])
    # ------------------------------------------------
    # 2. Access annotations via to_list
    # ------------------------------------------------
    geo_list = annotations.to_list()
    logger.info("GeometryDetectionAnnotations to_list len={}", len(geo_list))
    logger.info("GeometryDetectionAnnotations to_list first id={}", geo_list[0]["id"])
    logger.info("GeometryDetectionAnnotations to_list first geometry_shape={}", geo_list[0]["geometry_shape"])

    # ------------------------------------------------
    # 3. Handling None values in optional fields
    # ------------------------------------------------
    annotation_with_nones = datatypes.GeometryDetectionAnnotations(annotations=[{
        "id": 0,
        "image_id": None,
        "category_id": None,
        "geometry": None,
        "geometry_shape": None,
        "segmentation": None,
        "bbox": None,
        "area": None,
        "iscrowd": 0,
        "score": None,
    }])
    none_list = annotation_with_nones.to_list()
    logger.info("GeometryDetectionAnnotations with None to_list len={}", len(none_list))
    logger.info("GeometryDetectionAnnotations with None to_list first id={}", none_list[0]["id"])


def panoptic_segmentation_annotation_example():
    """
    Example for PanopticSegmentationAnnotation datatype.

    Demonstrates:
    - constructing a panoptic segmentation annotation
    - accessing image_id, labeled_mask, segments_info via to_dict
    """
    # ------------------------------------------------
    # 1. Create PanopticSegmentationAnnotation instance
    # ------------------------------------------------
    annotation = datatypes.PanopticSegmentationAnnotation(
        image_id=0,
        labeled_mask=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        segments_info=[{"id": 0, "category_id": 0, "bbox": np.array([0, 0, 1, 1])}],
    )

    # ------------------------------------------------
    # 2. Access data via to_dict
    # ------------------------------------------------
    data = annotation.to_dict()
    logger.info("PanopticSegmentationAnnotation to_dict image_id={}", data["image_id"])
    labeled_mask = data["labeled_mask"]
    logger.info("PanopticSegmentationAnnotation to_dict labeled_mask shape={}", labeled_mask.shape if hasattr(labeled_mask, "shape") else len(labeled_mask))
    logger.info("PanopticSegmentationAnnotation to_dict segments_info len={}", len(data["segments_info"]))


def circles_example():
    """
    Example for Circles datatype.

    Demonstrates:
    - constructing circles with centers and radii
    - accessing data via to_dict
    """
    # ------------------------------------------------
    # 1. Create Circles instance
    # ------------------------------------------------
    circles = datatypes.Circles(
        centers=[[0.1, 0.1], [0.2, 0.2]],
        radii=[0.1, 0.2],
    )

    # ------------------------------------------------
    # 2. Access data via to_dict
    # ------------------------------------------------
    data = circles.to_dict()
    logger.info("Circles to_dict centers={}", data["centers"])
    logger.info("Circles to_dict radii={}", data["radii"])


def ellipses_example():
    """
    Example for Ellipses datatype.

    Demonstrates:
    - constructing ellipses with centers, axes, and angles
    - accessing data via to_dict
    """
    # ------------------------------------------------
    # 1. Create Ellipses instance
    # ------------------------------------------------
    ellipses = datatypes.Ellipses(
        centers=[[0.1, 0.1], [0.2, 0.2]],
        axes=[[0.2, 0.1], [0.3, 0.2]],
        angles=[45.0, 30.0],
    )

    # ------------------------------------------------
    # 2. Access data via to_dict
    # ------------------------------------------------
    data = ellipses.to_dict()
    logger.info("Ellipses to_dict centers={}", data["centers"])
    logger.info("Ellipses to_dict axes={}", data["axes"])
    logger.info("Ellipses to_dict angles={}", data["angles"])


def lines_example():
    """
    Example for Lines datatype.

    Demonstrates:
    - constructing lines in rho-theta form
    - accessing data via to_dict
    """
    # ------------------------------------------------
    # 1. Create Lines instance
    # ------------------------------------------------
    lines = datatypes.Lines(rho=[1.0, 2.0], theta=[45.0, 90.0])

    # ------------------------------------------------
    # 2. Access data via to_dict
    # ------------------------------------------------
    data = lines.to_dict()
    logger.info("Lines to_dict rho={}", data["rho"])
    logger.info("Lines to_dict theta={}", data["theta"])


def part_example():
    """
    Example for Part datatype.

    Demonstrates:
    - constructing a Part with id, pose, dimensions, and state
    - accessing data via to_dict
    - minimal Part with optional fields omitted
    """
    # ------------------------------------------------
    # 1. Create Part instance
    # ------------------------------------------------
    part = datatypes.Part(
        id=1,
        pose=[100.0, 200.0, 50.0, 0.0, 0.0, 90.0],
        x_dim=10.5,
        y_dim=20.0,
        z_dim=5.0,
        state=PartState.VISIBLE,
    )

    # ------------------------------------------------
    # 2. Access data via to_dict
    # ------------------------------------------------
    part_dict = part.to_dict()
    logger.info("Part to_dict id={}", part_dict["id"])
    logger.info("Part to_dict pose={}", part_dict["pose"])
    logger.info("Part to_dict x_dim={}", part_dict["x_dim"])
    logger.info("Part to_dict y_dim={}", part_dict["y_dim"])
    logger.info("Part to_dict z_dim={}", part_dict["z_dim"])
    logger.info("Part to_dict state={}", part_dict["state"])

    # ------------------------------------------------
    # 3. Minimal Part (optional fields omitted)
    # ------------------------------------------------
    minimal_part = datatypes.Part(id=2, pose=[0, 0, 0, 0, 0, 0])
    minimal_dict = minimal_part.to_dict()
    logger.info("Minimal Part to_dict id={}", minimal_dict["id"])
    logger.info("Minimal Part to_dict x_dim={}", minimal_dict["x_dim"])
    logger.info("Minimal Part to_dict state={}", minimal_dict["state"])


def tray_example():
    """
    Example for Tray datatype.

    Demonstrates:
    - constructing a Tray with id, pose, dimensions, slots, and parts
    - accessing data via to_dict
    - empty Tray with no parts
    """
    # ------------------------------------------------
    # 1. Create Tray instance
    # ------------------------------------------------
    parts = [
        datatypes.Part(
            id=1,
            pose=[10, 20, 30, 0, 0, 0],
            x_dim=5.0,
            y_dim=5.0,
            z_dim=2.0,
            state=PartState.VISIBLE,
        ),
        datatypes.Part(
            id=2,
            pose=[40, 50, 60, 0, 0, 90],
            state=PartState.OCCLUDED,
        ),
    ]
    tray = datatypes.Tray(
        id=10,
        pose=[500.0, 300.0, 100.0, 0.0, 0.0, 0.0],
        x_dim=100.0,
        y_dim=80.0,
        z_dim=15.0,
        num_slots=4,
        parts=parts,
    )

    # ------------------------------------------------
    # 2. Access data via to_dict
    # ------------------------------------------------
    tray_dict = tray.to_dict()
    logger.info("Tray to_dict id={}", tray_dict["id"])
    logger.info("Tray to_dict pose={}", tray_dict["pose"])
    logger.info("Tray to_dict x_dim={}", tray_dict["x_dim"])
    logger.info("Tray to_dict y_dim={}", tray_dict["y_dim"])
    logger.info("Tray to_dict z_dim={}", tray_dict["z_dim"])
    logger.info("Tray to_dict num_slots={}", tray_dict["num_slots"])
    logger.info("Tray to_dict num_parts={}", len(tray_dict["parts"]))
    for idx, part_dict in enumerate(tray_dict["parts"]):
        logger.info("Tray parts[{}] to_dict id={}", idx, part_dict["id"])
        logger.info("Tray parts[{}] to_dict pose={}", idx, part_dict["pose"])
        logger.info("Tray parts[{}] to_dict state={}", idx, part_dict["state"])

    # ------------------------------------------------
    # 3. Empty Tray (no parts)
    # ------------------------------------------------
    empty_tray = datatypes.Tray(
        id=11,
        pose=[0, 0, 0, 0, 0, 0],
        x_dim=50.0,
        y_dim=40.0,
        z_dim=10.0,
        num_slots=0,
    )
    empty_dict = empty_tray.to_dict()
    logger.info("Empty Tray to_dict id={}", empty_dict["id"])
    logger.info("Empty Tray to_dict num_parts={}", len(empty_dict["parts"]))


def bin_example():
    """
    Example for Bin datatype.

    Demonstrates:
    - constructing a Bin with id, pose, and parts
    - accessing data via to_dict
    - empty Bin with no parts
    """
    # ------------------------------------------------
    # 1. Create Bin instance
    # ------------------------------------------------
    parts = [
        datatypes.Part(
            id=1,
            pose=[10, 20, 30, 45, 0, 0],
            x_dim=8.0,
            y_dim=8.0,
            z_dim=3.0,
            state=PartState.VISIBLE,
        ),
        datatypes.Part(id=2, pose=[40, 50, 60, 0, 30, 0]),
        datatypes.Part(
            id=3,
            pose=[70, 80, 90, 0, 0, 60],
            state=PartState.FAILED_GRASP,
        ),
    ]
    bin_obj = datatypes.Bin(
        id=20,
        pose=[1000.0, 500.0, 0.0, 0.0, 0.0, 0.0],
        parts=parts,
    )

    # ------------------------------------------------
    # 2. Access data via to_dict
    # ------------------------------------------------
    bin_dict = bin_obj.to_dict()
    logger.info("Bin to_dict id={}", bin_dict["id"])
    logger.info("Bin to_dict pose={}", bin_dict["pose"])
    logger.info("Bin to_dict num_parts={}", len(bin_dict["parts"]))
    for idx, part_dict in enumerate(bin_dict["parts"]):
        logger.info("Bin parts[{}] to_dict id={}", idx, part_dict["id"])
        logger.info("Bin parts[{}] to_dict pose={}", idx, part_dict["pose"])
        logger.info("Bin parts[{}] to_dict state={}", idx, part_dict["state"])

    # ------------------------------------------------
    # 3. Empty Bin (no parts)
    # ------------------------------------------------
    empty_bin = datatypes.Bin(id=21, pose=[0, 0, 0, 0, 0, 0])
    empty_dict = empty_bin.to_dict()
    logger.info("Empty Bin to_dict id={}", empty_dict["id"])
    logger.info("Empty Bin to_dict num_parts={}", len(empty_dict["parts"]))


datatype_example_dict = {
    "bool": bool_example,
    "float": float_example,
    "int": int_example,
    "boxes3d": boxes3d_example,
    "boxes2d": boxes2d_example,
    "mesh3d": mesh3d_example,
    "points3d": points3d_example,
    "vector3d": vector3d_example,
    "vector4d": vector4d_example,
    "imageformat": imageformat_example,
    "image": image_example,
    "mat4x4": mat4x4_example,
    "mat3x3": mat3x3_example,
    "float32": float32_example,
    "channeldatatype": channeldatatype_example,
    "colormodel": colormodel_example,
    "points2d": points2d_example,
    "linestrips2d": linestrips2d_example,
    "string": string_example,
    "rgba32": rgba32_example,
    "points3d_list": points3d_list_example,
    "categories": categories_example,
    "object_detection_annotations": object_detection_annotations_example,
    "panoptic_segmentation_annotation": panoptic_segmentation_annotation_example,
    "circles": circles_example,
    "ellipses": ellipses_example,
    "lines": lines_example,
    "geometry_detection_annotations": geometry_detection_annotations_example,
    "part": part_example,
    "tray": tray_example,
    "bin": bin_example,
}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all datatype examples")
    group.add_argument("datatype", nargs="?", type=str, help="Name of the datatype example to run")
    args = parser.parse_args()
    example_dict = datatype_example_dict

    # Track results for all examples
    example_results = {}

    if args.all:
        logger.info("Running all datatype examples...")
        for name, func in example_dict.items():
            logger.info(f"Running {name} example...")
            try:
                func()
                logger.info(f"{name} example completed.")
                example_results[name] = "✓ PASS"
            except Exception as e:
                logger.error(f"{name} example failed with error: {e}")
                example_results[name] = f"✗ FAIL: {str(e)[:50]}"
    else:
        datatype = args.datatype.lower()
        if datatype not in example_dict:
            logger.error(f"Datatype '{datatype}' not found.")
            close_matches = difflib.get_close_matches(
                datatype, example_dict.keys(), n=3, cutoff=0.4)
            if close_matches:
                logger.error(f"Did you mean one of these?")
                for match in close_matches:
                    logger.error(f"  - {match}")
            raise SystemExit(1)
        logger.info(f"Running {datatype} example...")
        try:
            example_dict[datatype]()
            logger.info(f"{datatype} example completed.")
            example_results[datatype] = "✓ PASS"
        except Exception as e:
            logger.error(f"{datatype} example failed with error: {e}")
            example_results[datatype] = f"✗ FAIL: {str(e)[:50]}"
            raise

    # Add table of failure/success at the end
    if example_results:
        logger.info("\n" + "=" * 70)
        logger.info("EXAMPLES - CONSOLIDATED TEST RESULTS")
        logger.info("=" * 70)
        max_name_len = max(len(name) for name in example_results.keys())
        for example_name, result in example_results.items():
            logger.info(f"{example_name:<{max_name_len}} | {result}")
        logger.info("=" * 70)

        # Count pass/fail
        passed = sum(1 for r in example_results.values() if r.startswith("✓"))
        failed = sum(1 for r in example_results.values() if r.startswith("✗"))
        logger.info(
            f"Total: {passed} passed, {failed} failed out of {len(example_results)} examples")
        logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()
