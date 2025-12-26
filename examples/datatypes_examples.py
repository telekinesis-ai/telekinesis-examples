"""Example usage of custom datatypes for serialization and deserialization.

This script demonstrates how to use the custom datatypes defined in the library for
serializing and deserializing data using PyArrow.

Example:
    # ...existing example code...
"""

import numpy as np
import argparse
from loguru import logger
import difflib
import pathlib

from datatypes import datatypes

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

def bool_example():
    """Example for Bool datatype."""
    b = datatypes.Bool(True)
    logger.info("Bool:", b.value)

def float_example():
    """Example for Float datatype."""
    f = datatypes.Float(3.14)
    logger.info("Float:", f.value)

def int_example():
    """Example for Int datatype."""
    i = datatypes.Int(42)
    logger.info("Int:", i.value)

def boxes3d_example():
    """Example for Boxes3D datatype."""
    half_size = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    center = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([255, 0, 255], dtype=np.uint8)
    rotation = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    b = datatypes.Boxes3D(half_size, center=center, colors=colors, rotation_in_euler_angles=rotation)
    logger.info("Boxes3D half_size:", b.half_size)


def imageformat_example():
    """Example for ImageFormat datatype."""
    # Example values for the format
    width = 2
    height = 2
    pixel_format = 1  # e.g., 1=RGB, 2=RGBA, etc. (user-defined)
    color_model = 1   # e.g., 1=RGB, 2=GRAY, etc. (user-defined)
    channel_datatype = 0  # 0=np.uint8, 1=np.float32, etc. (see _channel_datatype_to_numpy_and_arrow)
    fmt = datatypes.ImageFormat(width, height, pixel_format, color_model, channel_datatype)
    logger.info("ImageFormat:", fmt.__dict__)

def image_example():
    """Example for Image datatype."""
    # Example 1: RGBA format (4 channels)
    image_rgba = np.array([
        [[255, 0, 0, 255], [0, 255, 0, 255]],
        [[0, 0, 255, 255], [255, 255, 0, 255]]
    ], dtype=np.uint8)

    telekinesis_image_rgba = datatypes.Image(image=image_rgba)
    logger.info(f"Telekinesis image (RGBA): shape={telekinesis_image_rgba.to_numpy().shape}, dtype={telekinesis_image_rgba.to_numpy().dtype}")

    # Example 2: BGR format (same as cv2.imread returns)
    # cv2.imread returns images in BGR format with shape (height, width, 3) and dtype uint8
    # Creating a 2x2 BGR image manually
    image_bgr = np.array([
        [[255, 0, 0], [0, 255, 0]],      # Row 1: Blue pixel, Green pixel
        [[0, 0, 255], [255, 255, 0]]     # Row 2: Red pixel, Cyan pixel
    ], dtype=np.uint8)  # Shape: (2, 2, 3) - Height x Width x Channels (BGR)

    telekinesis_image_bgr = datatypes.Image(image=image_bgr)
    logger.info(f"Telekinesis image (BGR, cv2 format): shape={telekinesis_image_bgr.to_numpy().shape}, dtype={telekinesis_image_bgr.to_numpy().dtype}")


def mesh3d_example():
    """Example for Mesh3D datatype."""
    vertex_positions = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32)
    triangle_indices = np.array([[0,1,2]], dtype=np.int32)
    vertex_normals = np.array([[0,0,1],[0,0,1],[0,0,1]], dtype=np.float32)
    vertex_colors = np.array([[255,0,0,255],[0,255,0,255],[0,0,255,255]], dtype=np.uint8)
    m = datatypes.Mesh3D(vertex_positions, triangle_indices, vertex_normals, vertex_colors)
    logger.info("Mesh3D vertex_positions:", m.vertex_positions)

def points3d_example():
    """Example for Points3D datatype."""

    # Load points and colors
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    colors = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
        [0, 255, 255, 255],
    ]

    # Handle nan values

    input_positions = np.array(points, dtype=np.float32)
    input_colors = np.array(colors, dtype=np.uint8)
    input_radii = 0.1 # 1D array, shape (N,)

    points3d = datatypes.Points3D(positions=input_positions, 
                                  colors=input_colors, 
                                  radii=input_radii)

def vector3d_example():
    """Example for Vector3D datatype."""
    xyz = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v = datatypes.Vector3D(xyz)
    logger.info("Vector3D xyz:", v.xyz)

def vector4d_example():
    """Example for Vector4D datatype."""
    xyzw = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    v = datatypes.Vector4D(xyzw)
    logger.info("Vector4D xyzw:", v.xyzw)

def transform3d_example():
    """Example for Transform3D datatype."""
    translation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rotation = np.array([0.0, 0.0, np.pi/2], dtype=np.float32)  # 90 degrees around z
    scale = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    t = datatypes.Transform3D(translation, rotation, scale)
    logger.info(f"Transform3D translation: {t.translation}, rotation: {t.rotation_in_euler_angles}, scale: {t.scale}")

def mat4x4_example():
    """Example for Mat4X4 datatype."""
    matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    m = datatypes.Mat4X4(matrix)
    logger.info(f"Mat4X4 matrix:\n{m.matrix}")

def float32_example():
    """Example for Float32 datatype."""
    f = datatypes.Float(2.718)
    logger.info("Float value:", f.value)

def channeldatatype_example():
    """Example for ChannelDatatype enumeration."""
    dt = datatypes.ChannelDatatype.U8
    logger.info(f"ChannelDatatype: {dt}, value: {dt.value}")

def colormodel_example():
    """Example for ColorModel enumeration."""
    cm = datatypes.ColorModel.RGB
    logger.info(f"ColorModel: {cm}, value: {cm.value}, channels: {cm.num_channels()}")

def points2d_example():
    """Example for Points2D datatype."""
    positions = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0, 255], [0, 255, 0, 255]], dtype=np.uint8)
    p = datatypes.Points2D(positions, colors=colors)
    logger.info(f"Points2D positions: {p.positions}, colors: {p.colors}")

def string_example():
    """Example for String datatype."""
    s = datatypes.String("Hello, Telekinesis!")
    logger.info(f"String value: {s.value}")

def rgba32_example():
    """Example for Rgba32 datatype."""
    
    # Example 1: Create Rgba32 colors for R, G, B
    red = datatypes.Rgba32([255, 0, 0, 255])
    green = datatypes.Rgba32([0, 255, 0, 255])
    blue = datatypes.Rgba32([0, 0, 255, 255])
    
    logger.info(f"Red color (packed uint32): {red.rgba}")
    logger.info(f"Green color (packed uint32): {green.rgba}")
    logger.info(f"Blue color (packed uint32): {blue.rgba}")
    
    # Example 2: Use __int__() to convert to integer
    red_int = int(red)
    logger.info(f"Red as int: {red_int}")
    logger.info(f"Direct comparison: int(red) == red.rgba: {red_int == red.rgba}")
    
    # Create Points3D positions
    positions = np.array([
        [0.0, 0.0, 0.0],  # Red point
        [1.0, 0.0, 0.0],  # Green point
        [2.0, 0.0, 0.0],  # Blue point
    ], dtype=np.float32)
    
    # Create Points3D with Rgba32 colors (passed as list)
    points = datatypes.Points3D(
        positions=positions,
        colors=[red, green, blue],
        radii=0.2
    )
    
    logger.info("Visualized Points3D with Rgba32 colors in rerun")

def pinhole_intrinsics_example():
    """Example for Pinhole intrinsics."""
    # Example intrinsics matrix (focal length = 3, principal point at center)
    focal_length = 3.0
    image_width = 3
    image_height = 3

    pinhole = datatypes.Pinhole(focal_length=focal_length, width=image_width, height=image_height)


def pinhole_perspective_example():
    """Example for Pinhole perspective camera."""
    # Perspective camera parameters (same semantics as the Rerun example)
    fov_y = 0.7853982
    aspect_ratio = 1.7777778  # 16:9
    image_plane_distance = 0.1

    # Pick an actual resolution so your intrinsics are meaningful:
    width = 1280
    height = int(width / aspect_ratio)

    # 1) Construct your own Pinhole datatype
    tk_pinhole = datatypes.Pinhole(
        width=width,
        height=height,
        fov_y=fov_y,
        aspect_ratio=aspect_ratio,
        camera_xyz=None,  # you’re not using this yet for Rerun
        image_plane_distance=image_plane_distance,
    )
    logger.info("Logged pinhole perspective camera.")

def points3d_list_example():
    """Example for ListOfPoints3D datatype."""
    point1 = datatypes.Points3D(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.array([[255, 0, 0, 255], [0, 255, 0, 255]], dtype=np.uint8),
        radii=0.1
    )
    point2 = datatypes.Points3D(
        positions=np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32),
        colors=np.array([[0, 0, 255, 255], [255, 255, 0, 255]], dtype=np.uint8),
        radii=0.2
    )
    point_list = datatypes.ListOfPoints3D(point3d_list=[point1, point2])
    logger.info(f"ListOfPoints3D contains {len(point_list.point3d_list)} Points3D instances.")

datatype_example_dict = {
    "bool": bool_example,
    "float": float_example,
    "int": int_example,
    "boxes3d": boxes3d_example,
    "mesh3d": mesh3d_example,
    "points3d": points3d_example,
    "vector3d": vector3d_example,
    "vector4d": vector4d_example,
    "imageformat": imageformat_example,
    "image": image_example,
    "transform3d": transform3d_example,
    "mat4x4": mat4x4_example,
    "float32": float32_example,
    "channeldatatype": channeldatatype_example,
    "colormodel": colormodel_example,
    "points2d": points2d_example,
    "string": string_example,
    "rgba32": rgba32_example,
    "pinhole_intrinsics": pinhole_intrinsics_example,
    "pinhole_perspective": pinhole_perspective_example,
    "points3d_list": points3d_list_example,
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
            close_matches = difflib.get_close_matches(datatype, example_dict.keys(), n=3, cutoff=0.4)
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
        logger.info("\n" + "="*70)
        logger.info("EXAMPLES - CONSOLIDATED TEST RESULTS")
        logger.info("="*70)
        max_name_len = max(len(name) for name in example_results.keys())
        for example_name, result in example_results.items():
            logger.info(f"{example_name:<{max_name_len}} | {result}")
        logger.info("="*70)
        
        # Count pass/fail
        passed = sum(1 for r in example_results.values() if r.startswith("✓"))
        failed = sum(1 for r in example_results.values() if r.startswith("✗"))
        logger.info(f"Total: {passed} passed, {failed} failed out of {len(example_results)} examples")
        logger.info("="*70 + "\n")

if __name__ == "__main__":
    main()