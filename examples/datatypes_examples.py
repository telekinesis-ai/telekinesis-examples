"""
Examples demonstrating the use of various datatypes with the Telekinesis Vitreous SDK.

This script shows how to create and send different data types such as Bool, Int, Float, String, Points3D, and Boxes3D.
It includes realistic and lean (minimal) examples, and a command-line interface to run each example.
"""
import argparse
import difflib
import numpy as np
from loguru import logger

from datatypes import datatypes
from telekinesis import vitreous

def send_all_datatypes_example():
	"""
	Demonstrates creation and usage of all supported datatypes in the Telekinesis SDK.

	This function creates and sends example instances of the following datatypes:
		1. Bool
		2. Int
		3. Float
		4. String
		5. Points3D
		6. Boxes3D
		7. Boxes2D
		8. Mesh3D
		9. Points2D
		10. Vector3D
		11. Vector4D
		12. Transform3D
		13. Mat4X4
		14. Rgba32
		15. Image
		16. ListOfPoints3D
		17. PanopticSegmentationAnnotation
		18. ObjectDetectionAnnotations
		19. Categories
		20. Circles
		21. Ellipses
		22. Lines
		23. GeometryDetectionAnnotations
		24. LineStrips2D

	The function simulates realistic scenarios for each datatype, such as large point clouds, dense bounding boxes, high-resolution images, and complex annotations, and prints the returned values for verification.
	"""

	# ========================= 1. Bool =========================
	bool_value = datatypes.Bool(True)

	# ========================= 2. Int =========================
	int_value = datatypes.Int(42)

	# ========================= 3. Float =========================
	float_value = datatypes.Float(3.14)

	# ========================= 4. String =========================
	string_value = datatypes.String("Hello, Vitreous!")

	# ========================= 5. Points3D (10M points) =========================
	points = np.random.rand(10000000, 3).astype(np.float32)
	colors = np.random.randint(0, 255, size=(10000000, 4), dtype=np.uint8)
	normals = np.random.rand(10000000, 3).astype(np.float32)
	radii = 0.1
	points3d = datatypes.Points3D(positions=points, colors=colors, normals=normals, radii=radii)

	# ========================= 6. Boxes3D =========================
	# Maximum realistic scenario: Dense urban scene detection
	# Example: 1000 bounding boxes representing extreme density scenario
	# Use case: Dense crowd monitoring, aerial urban imagery, or large-scale warehouse
	# This represents industry maximum for real-time object detection systems
	num_boxes = 1000
	
	# Generate realistic 3D bounding boxes for various object types
	# Half sizes: Random but realistic dimensions for people, vehicles, objects
	# People: ~0.3m x 0.3m x 0.9m, Vehicles: ~1.0-2.5m x 0.8-1.0m x 1.2-1.8m
	# Objects/packages: ~0.2-0.8m per dimension
	half_sizes = np.column_stack([
		np.random.uniform(0.2, 2.5, num_boxes),  # Width (x): 0.4m - 5.0m
		np.random.uniform(0.2, 1.0, num_boxes),  # Depth (y): 0.4m - 2.0m
		np.random.uniform(0.3, 1.8, num_boxes),  # Height (z): 0.6m - 3.6m
	]).astype(np.float32)
	
	# Centers: Distribute across a large urban scene (50m x 50m area, height 0-10m)
	centers = np.column_stack([
		np.random.uniform(-25.0, 25.0, num_boxes),  # X: spread across 50m
		np.random.uniform(-25.0, 25.0, num_boxes),  # Y: spread across 50m
		np.random.uniform(0.5, 10.0, num_boxes),     # Z: height from ground to 10m
	]).astype(np.float32)
	
	# Rotations: Random orientations (realistic for objects in urban scenes)
	rotations = np.column_stack([
		np.random.uniform(0.0, 0.2, num_boxes),      # Slight pitch variation
		np.random.uniform(0.0, 6.28, num_boxes),     # Full yaw rotation (0-360°)
		np.random.uniform(0.0, 0.2, num_boxes),      # Slight roll variation
	]).astype(np.float32)
	
	box3d = datatypes.Boxes3D(half_sizes=half_sizes, 
						   centers=centers, 
						   rotations_in_euler_angle=rotations)

	# ========================= 7. Boxes2D =========================
	# 1000 bounding boxes (maximum realistic: dense image object detection)
	# Simulates detection results from a 4K image with many objects (e.g., crowd monitoring, OCR, dense urban scene)
	# Format: [x_min, y_min, x_max, y_max] for each box
	num_boxes_2d = 1000
	
	# Generate realistic bounding boxes spread across a 4K image (3840x2160)
	image_width, image_height = 3840, 2160
	
	# Random box positions across the image
	x_mins = np.random.uniform(0, image_width - 200, num_boxes_2d)
	y_mins = np.random.uniform(0, image_height - 200, num_boxes_2d)
	
	# Random box sizes (typical object detection box sizes: 20-200 pixels)
	box_widths = np.random.uniform(20, 200, num_boxes_2d)
	box_heights = np.random.uniform(20, 200, num_boxes_2d)
	
	x_maxs = np.clip(x_mins + box_widths, 0, image_width)
	y_maxs = np.clip(y_mins + box_heights, 0, image_height)
	
	# Stack into (N, 4) format: each row is [x_min, y_min, x_max, y_max]
	arrays = np.column_stack([x_mins, y_mins, x_maxs, y_maxs]).astype(np.float32)

	box2d = datatypes.Boxes2D(arrays=arrays)

	# ========================= 8. Mesh3D =========================
	# Maximum realistic scenario: High-resolution 3D scanned object
	# Industry standard: Production CAD models, high-quality 3D scans, game assets
	# Upper limit for real-time applications: 1-5 million triangles
	# Example: Detailed 3D scanned industrial part, character mesh, or architectural model
	
	# Generate a high-resolution mesh (1 million triangles = 3 million vertices in triangle list)
	# In real meshes, vertices are shared, so ~500K unique vertices for 1M triangles
	num_vertices = 500000  # 500K vertices (realistic for high-quality 3D scan)
	num_triangles = 1000000  # 1M triangles (industry upper limit for real-time use)
	
	# Vertex positions: Realistic coordinates for a medium-sized object (e.g., 1m x 1m x 1m)
	# Distributed across object space with some variation
	vertex_positions = np.random.uniform(-0.5, 0.5, (num_vertices, 3)).astype(np.float32)
	
	# Triangle indices: Each triangle references 3 vertices
	# Must be valid indices into vertex_positions array
	# Shape must be (M, 3) where M is the number of triangles
	triangle_indices = np.random.randint(0, num_vertices, (num_triangles, 3), dtype=np.int32)
	
	# Vertex normals: Unit vectors pointing outward from surface
	# In real meshes, these are computed from triangle faces or measured via scanning
	vertex_normals = np.random.randn(num_vertices, 3).astype(np.float32)
	# Normalize to unit vectors
	norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
	vertex_normals = vertex_normals / norms
	
	# Vertex colors: RGBA values for each vertex
	# In real meshes, these come from texture maps or vertex coloring
	vertex_colors = np.random.randint(0, 255, size=(num_vertices, 4), dtype=np.uint8)
	vertex_colors[:, 3] = 255  # Full opacity
	
	mesh = datatypes.Mesh3D(
		vertex_positions=vertex_positions,
		triangle_indices=triangle_indices,
		vertex_normals=vertex_normals,
		vertex_colors=vertex_colors,
	)

	# ========================= 9. Points2D =========================
	# Maximum realistic scenario: Dense 2D feature detection
	# Example: 1 million keypoints from high-resolution image feature extraction
	# Use case: SLAM, visual odometry, panoramic image stitching, dense optical flow
	# This represents industry maximum for dense feature detection on 4K images
	num_points_2d = 1000000  # 1M points (upper limit for dense feature detection)
	
	# Positions distributed across 4K image space (3840x2160)
	positions = np.column_stack([
		np.random.uniform(0, 3840, num_points_2d),  # X coordinates
		np.random.uniform(0, 2160, num_points_2d),  # Y coordinates
	]).astype(np.float32)
	
	# Colors: Feature strength or classification visualization
	colors = np.random.randint(0, 255, size=(num_points_2d, 4), dtype=np.uint8)
	colors[:, 3] = 255  # Full opacity
	
	# Radii: Single float value for all points (typical for feature visualization)
	radii = 2.5  # Medium size radius for visualization
	
	points2d = datatypes.Points2D(positions=positions, 
								  colors=colors, 
								  radii=radii)

	# ========================= 10. Vector3D =========================
	vector3d_array = np.random.rand(3).astype(np.float32)
	vector3d = datatypes.Vector3D(xyz=vector3d_array)

	# ========================= 11. Vector4D =========================
	vector4d_array = np.random.rand(4).astype(np.float32)
	vector4d = datatypes.Vector4D(xyzw=vector4d_array)

	# 12. Mat4X4
	matrix = np.eye(4, dtype=np.float32)
	mat4x4 = datatypes.Mat4X4(matrix=matrix)

	# 13. Rgba32
	color = np.array([255, 0, 0, 255], dtype=np.uint8)
	color = datatypes.Rgba32(rgba=color)

	# 14. Image - 4K UHD resolution (3840x2160, industry standard high resolution)
	# This represents a typical high-resolution camera image with RGB channels
	image_array = np.random.randint(0, 255, size=(2160, 3840, 3), dtype=np.uint8)
	image = datatypes.Image(image=image_array)

	# 15. List of points3d - Maximum realistic scenario
	# Two separate 10M point clouds (e.g., before/after scans, multi-view reconstruction)
	points = np.random.rand(10000000, 3).astype(np.float32)
	colors = np.random.randint(0, 255, size=(10000000, 4), dtype=np.uint8)
	normals = np.random.rand(10000000, 3).astype(np.float32)
	radii_3d = 0.1  # Radius for 3D points

	points3d_1 = datatypes.Points3D(positions=points, colors=colors, normals=normals, radii=radii_3d)
	points3d_2 = datatypes.Points3D(positions=points, colors=colors, normals=normals, radii=radii_3d)
	points3d_list = [points3d_1, points3d_2]
	points3d_list = datatypes.ListOfPoints3D(point3d_list=points3d_list)

	# 16. Panoptic segmentation annotation - Maximum realistic scenario
	# 4K UHD panoptic segmentation with dense instance labeling
	# Use case: Autonomous driving, satellite imagery analysis, medical imaging
	# Industry maximum: 4K resolution with 500-1000 segment instances per frame
	num_segments = 500  # 500 segments (realistic maximum for dense urban/medical scenes)
	
	# Generate 4K labeled mask (3840x2160) with unique segment IDs
	# Using uint16 to support up to 65,535 unique segment IDs
	labeled_mask = np.random.randint(0, num_segments, size=(2160, 3840), dtype=np.uint16)
	
	# Generate segment info for each segment (bounding boxes, categories, etc.)
	segments_info = []
	for seg_id in range(1, num_segments + 1):
		# Random bounding box within 4K frame
		x_min = np.random.randint(0, 3640)
		y_min = np.random.randint(0, 1960)
		width = np.random.randint(20, 200)
		height = np.random.randint(20, 200)
		
		segments_info.append({
			"id": seg_id,
			"category_id": np.random.randint(1, 81),  # COCO has 80 categories
			"bbox": np.array([x_min, y_min, width, height])
		})
	
	panoptic_segmentation_annotation = datatypes.PanopticSegmentationAnnotation(
		image_id=0,
		labeled_mask=labeled_mask,
		segments_info=segments_info
	)

	# 17. ObjectDetectionAnnotation - Maximum realistic scenario
	# Dense object detection annotations for 4K image (e.g., COCO-style annotations)
	# Use case: Training data for object detection, instance segmentation
	# Industry maximum: 100-200 object instances per image in dense scenes
	num_objects = 200  # 200 objects (maximum for very dense scenes like crowds)
	
	object_annotations = []
	for obj_id in range(1, num_objects + 1):
		# Random bounding box within 4K frame
		x_min = np.random.randint(0, 3640)
		y_min = np.random.randint(0, 1960)
		width = np.random.randint(20, 300)
		height = np.random.randint(20, 300)
		area = width * height
		
		# Generate polygon segmentation (simplified: rectangle as polygon)
		segmentation = [[
			x_min, y_min,
			x_min + width, y_min,
			x_min + width, y_min + height,
			x_min, y_min + height
		]]
		
		object_annotations.append({
			"id": obj_id,
			"image_id": 0,
			"category_id": np.random.randint(1, 81),  # COCO has 80 categories
			"segmentation": segmentation,
			"bbox": [x_min, y_min, width, height],
			"area": area,
			"iscrowd": 0,
		})
	
	object_detection_annotations = datatypes.ObjectDetectionAnnotations(object_annotations)

	# 18. Categories - Maximum realistic scenario
	# Complete category taxonomy for large-scale datasets
	# Use case: COCO (80 categories), OpenImages (600 categories), ADE20K (150 categories)
	# Industry maximum: 1000+ categories for comprehensive object/scene understanding
	num_categories = 150  # 150 categories (realistic for scene understanding like ADE20K)
	
	# Generate comprehensive category list with hierarchical structure
	category_list = []
	supercategories = ["object", "stuff", "vehicle", "animal", "food", "furniture", 
					   "electronic", "appliance", "indoor", "outdoor", "person", "accessory"]
	
	for cat_id in range(1, num_categories + 1):
		category_list.append({
			"id": cat_id,
			"name": f"Category_{cat_id}",
			"supercategory": supercategories[cat_id % len(supercategories)],
			"isthing": 1 if cat_id % 3 != 0 else 0,  # Mix of thing/stuff categories
			"color": [
				np.random.randint(0, 255),
				np.random.randint(0, 255),
				np.random.randint(0, 255)
			]
		})
	
	categories = datatypes.Categories(categories=category_list)

	# 19. Circles - Maximum realistic scenario: Dense circle detection
	# Use case: Industrial inspection (bolts, holes, bearings), cell counting, coin detection
	# Industry maximum: 10,000 circles detected in high-resolution inspection images
	num_circles = 10000  # 10K circles (maximum for dense inspection scenarios)
	
	# Circle centers distributed across 4K image
	circle_centers = np.column_stack([
		np.random.uniform(0, 3840, num_circles),  # X coordinates
		np.random.uniform(0, 2160, num_circles),  # Y coordinates
	]).astype(np.float32)
	
	# Circle radii: Varying sizes (e.g., 5-100 pixels for different sized objects)
	circle_radii = np.random.uniform(5.0, 100.0, num_circles).astype(np.float32)
	
	circle = datatypes.Circles(centers=circle_centers, radii=circle_radii)

	# 20. Ellipses - Maximum realistic scenario: Dense ellipse detection
	# Use case: Biological cell analysis, quality control, gauge reading, eye tracking
	# Industry maximum: 5,000 ellipses for dense biological/industrial analysis
	num_ellipses = 5000  # 5K ellipses (maximum for dense analysis)
	
	# Ellipse centers distributed across 4K image
	ellipse_centers = np.column_stack([
		np.random.uniform(0, 3840, num_ellipses),  # X coordinates
		np.random.uniform(0, 2160, num_ellipses),  # Y coordinates
	]).astype(np.float32)
	
	# Ellipse axes: Major and minor axis lengths
	ellipse_axes = np.column_stack([
		np.random.uniform(10.0, 150.0, num_ellipses),  # Major axis
		np.random.uniform(5.0, 100.0, num_ellipses),   # Minor axis
	]).astype(np.float32)
	
	# Ellipse rotation angles (0 to 2π radians)
	ellipse_angles = np.random.uniform(0.0, 6.28, num_ellipses).astype(np.float32)
	
	ellipse = datatypes.Ellipses(centers=ellipse_centers, axes=ellipse_axes, angles=ellipse_angles)

	# 21. Lines - Maximum realistic scenario: Dense line detection
	# Use case: Document analysis, road lane detection, architectural plans, Hough transform
	# Industry maximum: 50,000 lines for detailed architectural/document analysis
	num_lines = 50000  # 50K lines (maximum for dense line detection)
	
	# Line parameters in Hough space (rho, theta)
	# rho: distance from origin (0 to diagonal of 4K image ≈ 4405 pixels)
	# theta: angle (0 to π radians)
	line_rhos = np.random.uniform(0.0, 4405.0, num_lines).astype(np.float32)
	line_thetas = np.random.uniform(0.0, 3.14159, num_lines).astype(np.float32)
	
	line = datatypes.Lines(rho=line_rhos, theta=line_thetas)

	# 22. GeometryDetectionAnnotations - Maximum realistic scenario: Comprehensive geometry detection
	num_geometries = 1000  # 1K geometry detections (maximum for comprehensive analysis)
	
	geometry_annotations = []
	for _ in range(num_geometries):
		# Randomly select geometry type
		geom_type = np.random.choice(["circle", "ellipse", "line", "contour"])

		if geom_type == "circle":
			geometry_annotations.append({
				"id": np.random.randint(1, 10000),
				"image_id": np.random.randint(0, 1000),
				"category_id": np.random.randint(1, 150),
				"geometry": {
					"center": [np.random.uniform(0, 3840), np.random.uniform(0, 2160)],
					"radius": np.random.uniform(5.0, 100.0)
				},
				"geometry_shape": "circle",
			})
		elif geom_type == "ellipse":
			geometry_annotations.append({
				"id": np.random.randint(1, 10000),
				"image_id": np.random.randint(0, 1000),
				"category_id": np.random.randint(1, 150),
				"geometry": {
					"center": [np.random.uniform(0, 3840), np.random.uniform(0, 2160)],
					"axes": [np.random.uniform(10.0, 150.0), np.random.uniform(5.0, 100.0)],
					"angle": np.random.uniform(0.0, 6.28)
				},
				"geometry_shape": "ellipse",
			})
		elif geom_type == "line":
			geometry_annotations.append({
				"id": np.random.randint(1, 10000),
				"image_id": np.random.randint(0, 1000),
				"category_id": np.random.randint(1, 150),
				"geometry": {
					"rho": np.random.uniform(0.0, 4405.0),
					"theta": np.random.uniform(0.0, 3.14159)
				},
				"geometry_shape": "line",
			})
		elif geom_type == "contour":
			geometry_annotations.append({
				"id": np.random.randint(1, 10000),
				"image_id": np.random.randint(0, 1000),
				"category_id": np.random.randint(1, 150),
				"geometry": {
					"points": [np.random.uniform(0, 3840, 2).tolist() for _ in range(np.random.randint(10, 200))]
				},
				"geometry_shape": "contour",
			})

	geometry_detection_annotations = datatypes.GeometryDetectionAnnotations(geometry_annotations)

	# 23. LineStrips2D - Maximum realistic scenario
	# Contours detected from high-resolution image analysis
	# Use case: Object detection, edge detection, manufacturing quality control, medical imaging
	# Industry maximum: 500 contours with varying complexity (10-200 points each)
	num_contours = 500  # 500 contours (maximum for comprehensive scene analysis)
	strips = []
	
	for _ in range(num_contours):
		# Random number of points per contour (simple to complex shapes)
		num_points = np.random.randint(10, 200)  # 10-200 points per contour
		
		# Generate contour points within 4K image bounds (3840x2160)
		# Each contour represents a closed or open shape detected in the image
		contour_points = np.column_stack([
			np.random.uniform(0, 3840, num_points),  # x coordinates
			np.random.uniform(0, 2160, num_points)   # y coordinates
		]).astype(np.float32)
		
		strips.append(contour_points)
	
	contour = datatypes.LineStrips2D(strips=strips)


	response_data = vitreous._send_all_datatypes(
								param_1=bool_value,
								param_2=int_value,
								param_3=float_value,
								param_4=string_value,
								param_5=vector3d,
								param_6=vector4d,
								param_7=mat4x4,
								param_8=color,
								param_9=image,
								param_10=box3d,
								param_11=box2d,
								param_12=mesh,
								param_13=points2d,
								param_14=points3d,
								param_15=points3d_list,
								param_16=panoptic_segmentation_annotation,
								param_17=object_detection_annotations,
								param_18=categories,
								param_19=circle,
								param_20=ellipse,
								param_21=line,
								param_22=geometry_detection_annotations,
								param_23=contour,
							)
	
	# ========== 1. Bool ==========
	logger.success(f"param_1 Bool value: {response_data['param_1'].value}")
	print("="*150)

	# ========== 2. Int ==========
	logger.success(f"param_2 Int value: {response_data['param_2'].value}")
	print("="*150)

	# ========== 3. Float ==========
	logger.success(f"param_3 Float value: {response_data['param_3'].value}")
	print("="*150)

	# ========== 4. String ==========
	logger.success(f"param_4 String value: {response_data['param_4'].value}")
	print("="*150)

	# ========== 5. Vector3D ==========
	logger.success(f"param_5 Vector3D xyz: {response_data['param_5'].xyz}")
	print("="*150)

	# ========== 6. Vector4D ==========
	logger.success(f"param_6 Vector4D xyzw: {response_data['param_6'].xyzw}")
	print("="*150)

	# ========== 7. Mat4X4 ==========
	logger.success(f"param_7 Mat4X4 matrix: {response_data['param_7'].matrix}")
	print("="*150)

	# ========== 8. Rgba32 ==========
	logger.success(f"param_8 Rgba32: {response_data['param_8'].rgba}")
	print("="*150)

	# ========== 9. Image ==========
	logger.success(f"param_9 Image nparray: {(response_data['param_9'])}")
	print("="*150)

	# ========== 10. Boxes3D ==========
	logger.success(f"param_10 Boxes3D half_sizes: {response_data['param_10'].half_sizes}")
	logger.success(f"param_10 Boxes3D centers: {response_data['param_10'].centers}")
	print("="*150)

	# ========== 11. Boxes2D ==========
	logger.success(f"param_11 Boxes2D half_sizes: {response_data['param_11'].half_sizes}")
	logger.success(f"param_11 Boxes2D centers: {response_data['param_11'].centers}")
	print("="*150)

	# ========== 12. Mesh3D ==========
	logger.success(f"param_12 Mesh3D vertex_positions: {response_data['param_12'].vertex_positions}")
	logger.success(f"param_12 Mesh3D triangle_indices: {response_data['param_12'].triangle_indices}")
	logger.success(f"param_12 Mesh3D vertex_normals: {response_data['param_12'].vertex_normals}")
	logger.success(f"param_12 Mesh3D vertex_colors: {response_data['param_12'].vertex_colors}")
	print("="*150)

	# ========== 13. Points2D ==========
	logger.success(f"param_13 Points2D positions: {response_data['param_13'].positions}")
	logger.success(f"param_13 Points2D colors: {response_data['param_13'].colors}")
	logger.success(f"param_13 Points2D radii: {response_data['param_13'].radii}")
	print("="*150)

	# ========== 14. Points3D ==========
	logger.success(f"param_14 Points3D positions: {response_data['param_14'].positions}")
	logger.success(f"param_14 Points3D colors: {response_data['param_14'].colors}")
	logger.success(f"param_14 Points3D radii: {response_data['param_14'].radii}")
	print("="*150)

	# ========== 15. ListOfPoints3D ==========
	for i, point3d in enumerate(response_data['param_15'].point3d_list):
		logger.success(f"param_15 ListOfPoints3D[{i}] positions: {point3d.positions}")
		logger.success(f"param_15 ListOfPoints3D[{i}] colors: {point3d.colors}")
		logger.success(f"param_15 ListOfPoints3D[{i}] radii: {point3d.radii}")
	print("="*150)

	# ========== 16. PanopticSegmentationAnnotation ==========
	logger.success(f"param_16 PanopticSegmentationAnnotation image_id: {response_data['param_16'].image_id}")
	logger.success(f"param_16 PanopticSegmentationAnnotation labeled_mask: {response_data['param_16'].labeled_mask.to_numpy()}")
	logger.success(f"param_16 PanopticSegmentationAnnotation segments_info: {response_data['param_16'].segments_info.to_list()}")
	print("="*150)

	# ========== 17. ObjectDetectionAnnotations ==========
	logger.success(f"param_17 ObjectDetectionAnnotations: {response_data['param_17'].to_list()}")
	print("="*150)

	# ========== 18. Categories ==========
	logger.success(f"param_18 Categories: {response_data['param_18'].to_list()}")
	print("="*150)

	# ========== 19. Circles ==========
	logger.success(f"param_19 Circles centers: {response_data['param_19'].centers}")
	logger.success(f"param_19 Circles radii: {response_data['param_19'].radii}")
	print("="*150)

	# ========== 20. Ellipses ==========
	logger.success(f"param_20 Ellipses centers: {response_data['param_20'].centers}")
	logger.success(f"param_20 Ellipses axes: {response_data['param_20'].axes}")
	logger.success(f"param_20 Ellipses angles: {response_data['param_20'].angles}")
	print("="*150)

	# ========== 21. Lines ==========
	logger.success(f"param_21 Lines rho: {response_data['param_21'].rho}")
	logger.success(f"param_21 Lines theta: {response_data['param_21'].theta}")
	print("="*150)

	# ========== 22. GeometryDetectionAnnotations ==========
	logger.success(f"param_22 GeometryDetectionAnnotations: {response_data['param_22'].to_list()}")
	print("="*150)

	# ========== 23. LineStrips2D ==========
	logger.success(f"param_23 LineStrips2D strips: {response_data['param_23'].strips}")
	print("="*150)

def send_all_datatypes_lean_example():
	"""
	Lean version of send_all_datatypes_example with minimal dummy data.

	All datatypes use the simplest possible data (size 1 or minimal complexity).
	This is for testing and demonstration purposes, not real-world scenarios.
	"""
	

	# ========================= 1. Bool =========================
	bool_value = datatypes.Bool(True)

	# ========================= 2. Int =========================
	int_value = datatypes.Int(42)

	# ========================= 3. Float =========================
	float_value = datatypes.Float(3.14)

	# ========================= 4. String =========================
	string_value = datatypes.String("Hello, Vitreous!")

	# ========================= 5. Points3D =========================
	points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
	colors = np.array([[255, 0, 0, 255]], dtype=np.uint8)
	normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
	radii = 0.1
	points3d = datatypes.Points3D(positions=points, colors=colors, normals=normals, radii=radii)

	# ========================= 6. Boxes3D =========================
	half_sizes = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
	centers = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
	rotations = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
	box3d = datatypes.Boxes3D(half_sizes=half_sizes, centers=centers, rotations_in_euler_angle=rotations)

	# ========================= 7. Boxes2D =========================
	arrays = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
	box2d = datatypes.Boxes2D(arrays=arrays)

	# ========================= 8. Mesh3D =========================
	vertex_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
	triangle_indices = np.array([[0, 1, 2]], dtype=np.int32)
	vertex_normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
	vertex_colors = np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)
	mesh = datatypes.Mesh3D(
		vertex_positions=vertex_positions,
		triangle_indices=triangle_indices,
		vertex_normals=vertex_normals,
		vertex_colors=vertex_colors,
	)

	# ========================= 9. Points2D =========================
	positions = np.array([[100.0, 100.0]], dtype=np.float32)
	colors = np.array([[255, 0, 0, 255]], dtype=np.uint8)
	radii = 2.5
	points2d = datatypes.Points2D(positions=positions, colors=colors, radii=radii)

	# ========================= 10. Vector3D =========================
	vector3d_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
	vector3d = datatypes.Vector3D(xyz=vector3d_array)

	# ========================= 11. Vector4D =========================
	vector4d_array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
	vector4d = datatypes.Vector4D(xyzw=vector4d_array)

	# ========================= 12. Mat4X4 =========================
	matrix = np.eye(4, dtype=np.float32)
	mat4x4 = datatypes.Mat4X4(matrix=matrix)

	# ========================= 13. Rgba32 =========================
	color = np.array([255, 0, 0, 255], dtype=np.uint8)
	color = datatypes.Rgba32(rgba=color)

	# ========================= 14. Image =========================
	image_array = np.array([[[255, 0, 0]]], dtype=np.uint8)
	image = datatypes.Image(image=image_array)

	# ========================= 15. ListOfPoints3D =========================
	points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
	colors = np.array([[255, 0, 0, 255]], dtype=np.uint8)
	normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
	radii_3d = 0.1
	points3d_1 = datatypes.Points3D(positions=points, colors=colors, normals=normals, radii=radii_3d)
	points3d_list = [points3d_1]
	points3d_list = datatypes.ListOfPoints3D(point3d_list=points3d_list)

	# ========================= 16. PanopticSegmentationAnnotation =========================
	labeled_mask = np.array([[1]], dtype=np.uint16)
	segments_info = [{
		"id": 1,
		"category_id": 1,
		"bbox": np.array([0, 0, 1, 1])
	}]
	panoptic_segmentation_annotation = datatypes.PanopticSegmentationAnnotation(
		image_id=0,
		labeled_mask=labeled_mask,
		segments_info=segments_info
	)

	# ========================= 17. ObjectDetectionAnnotations =========================
	object_annotations = [{
		"id": 1,
		"image_id": 0,
		"category_id": 1,
		"segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]],
		"bbox": [10.0, 10.0, 40.0, 40.0],
		"area": 1600.0,
		"iscrowd": 0,
	}]
	object_detection_annotations = datatypes.ObjectDetectionAnnotations(object_annotations)

	# ========================= 18. Categories =========================
	category_list = [{
		"id": 1,
		"name": "object",
		"supercategory": "thing",
		"isthing": 1,
		"color": [255, 0, 0]
	}]
	categories = datatypes.Categories(categories=category_list)

	# ========================= 19. Circles =========================
	circle_centers = np.array([[100.0, 100.0]], dtype=np.float32)
	circle_radii = np.array([50.0], dtype=np.float32)
	circle = datatypes.Circles(centers=circle_centers, radii=circle_radii)

	# ========================= 20. Ellipses =========================
	ellipse_centers = np.array([[100.0, 100.0]], dtype=np.float32)
	ellipse_axes = np.array([[60.0, 40.0]], dtype=np.float32)
	ellipse_angles = np.array([0.0], dtype=np.float32)
	ellipse = datatypes.Ellipses(centers=ellipse_centers, axes=ellipse_axes, angles=ellipse_angles)

	# ========================= 21. Lines =========================
	line_rhos = np.array([100.0], dtype=np.float32)
	line_thetas = np.array([1.57], dtype=np.float32)
	line = datatypes.Lines(rho=line_rhos, theta=line_thetas)

	# ========================= 22. GeometryDetectionAnnotations =========================
	geometry_annotations = [{
		"id": 1,
		"image_id": 0,
		"category_id": 1,
		"geometry": {
			"center": [100.0, 100.0],
			"radius": 50.0
		},
		"geometry_shape": "circle",
	},
	{
		"id": 2,
		"image_id": 0,
		"category_id": 1,
		"geometry": {
			"points": [[150.0, 150.0], [250.0, 250.0]]
		},
		"geometry_shape": "contour",
	}
]
	geometry_detection_annotations = datatypes.GeometryDetectionAnnotations(geometry_annotations)

	# ========================= 23. LineStrips2D =========================
	strips = [np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0]], dtype=np.float32)]
	contour = datatypes.LineStrips2D(strips=strips)

	response_data = vitreous._send_all_datatypes(
								param_1=bool_value,
								param_2=int_value,
								param_3=float_value,
								param_4=string_value,
								param_5=vector3d,
								param_6=vector4d,
								param_7=mat4x4,
								param_8=color,
								param_9=image,
								param_10=box3d,
								param_11=box2d,
								param_12=mesh,
								param_13=points2d,
								param_14=points3d,
								param_15=points3d_list,
								param_16=panoptic_segmentation_annotation,
								param_17=object_detection_annotations,
								param_18=categories,
								param_19=circle,
								param_20=ellipse,
								param_21=line,
								param_22=geometry_detection_annotations,
								param_23=contour,
							)
	
	# Bool
	print("="*150)
	logger.success(f"param_1 Bool value: {response_data['param_1'].value}")
	print("="*150)

	# Int
	logger.success(f"param_2 Int value: {response_data['param_2'].value}")
	print("="*150)

	# Float
	logger.success(f"param_3 Float value: {response_data['param_3'].value}")
	print("="*150)

	# String
	logger.success(f"param_4 String value: {response_data['param_4'].value}")
	print("="*150)

	# Vector3D
	logger.success(f"param_5 Vector3D xyz: {response_data['param_5'].xyz}")
	print("="*150)

	# Vector4D
	logger.success(f"param_6 Vector4D xyzw: {response_data['param_6'].xyzw}")
	print("="*150)

	# Mat4X4
	logger.success(f"param_7 Mat4X4 matrix: {response_data['param_7'].matrix}")
	print("="*150)

	# Rgba32
	logger.success(f"param_8 Rgba32: {response_data['param_8'].rgba}")
	print("="*150)

	# Image
	logger.success(f"param_9 Image nparray: {(response_data['param_9'])}")
	print("="*150)

	# Boxes3D
	logger.success(f"param_10 Boxes3D half_sizes: {response_data['param_10'].half_sizes}")
	logger.success(f"param_10 Boxes3D centers: {response_data['param_10'].centers}")
	print("="*150)

	# Boxes2D
	logger.success(f"param_11 Boxes2D half_sizes: {response_data['param_11'].half_sizes}")
	logger.success(f"param_11 Boxes2D centers: {response_data['param_11'].centers}")
	print("="*150)

	# Mesh3D
	logger.success(f"param_12 Mesh3D vertex_positions: {response_data['param_12'].vertex_positions}")
	logger.success(f"param_12 Mesh3D triangle_indices: {response_data['param_12'].triangle_indices}")
	logger.success(f"param_12 Mesh3D vertex_normals: {response_data['param_12'].vertex_normals}")
	logger.success(f"param_12 Mesh3D vertex_colors: {response_data['param_12'].vertex_colors}")
	print("="*150)

	# Points2D
	logger.success(f"param_13 Points2D positions: {response_data['param_13'].positions}")
	logger.success(f"param_13 Points2D colors: {response_data['param_13'].colors}")
	logger.success(f"param_13 Points2D radii: {response_data['param_13'].radii}")
	print("="*150)

	# Points3D
	logger.success(f"param_14 Points3D positions: {response_data['param_14'].positions}")
	logger.success(f"param_14 Points3D colors: {response_data['param_14'].colors}")
	logger.success(f"param_14 Points3D radii: {response_data['param_14'].radii}")
	print("="*150)

	# ListOfPoints3D
	for i, point3d in enumerate(response_data['param_15'].point3d_list):
		logger.success(f"param_15 ListOfPoints3D[{i}] positions: {point3d.positions}")
		logger.success(f"param_15 ListOfPoints3D[{i}] colors: {point3d.colors}")
		logger.success(f"param_15 ListOfPoints3D[{i}] radii: {point3d.radii}")
	print("="*150)

	# PanopticSegmentationAnnotation
	logger.success(f"param_16 PanopticSegmentationAnnotation image_id: {response_data['param_16'].image_id}")
	logger.success(f"param_16 PanopticSegmentationAnnotation labeled_mask: {response_data['param_16'].labeled_mask.to_numpy()}")
	logger.success(f"param_16 PanopticSegmentationAnnotation segments_info: {response_data['param_16'].segments_info.to_list()}")
	print("="*150)

	# ObjectDetectionAnnotations
	logger.success(f"param_17 ObjectDetectionAnnotations: {response_data['param_17'].to_list()}")
	print("="*150)

	# Categories
	logger.success(f"param_18 Categories: {response_data['param_18'].to_list()}")
	print("="*150)

	# Circles
	logger.success(f"param_19 Circles centers: {response_data['param_19'].centers}")
	logger.success(f"param_19 Circles radii: {response_data['param_19'].radii}")
	print("="*150)

	# Ellipses
	logger.success(f"param_20 Ellipses centers: {response_data['param_20'].centers}")
	logger.success(f"param_20 Ellipses axes: {response_data['param_20'].axes}")
	logger.success(f"param_20 Ellipses angles: {response_data['param_20'].angles}")
	print("="*150)

	# Lines
	logger.success(f"param_21 Lines rho: {response_data['param_21'].rho}")
	logger.success(f"param_21 Lines theta: {response_data['param_21'].theta}")
	print("="*150)

	# GeometryDetectionAnnotation
	logger.success(f"param_22 GeometryDetectionAnnotations: {response_data['param_22'].to_list()}")
	print("="*150)

	# LineStrips2D
	logger.success(f"param_23 LineStrips2D strips: {response_data['param_23'].strips}")
	print("="*150)

def get_example_dict():
	"""
	Returns a dictionary mapping example names (without _example suffix) to their functions.
	Used for command-line selection of examples.
	"""
	return {
		# Test for all datatypes sending
		"send_all_datatypes": send_all_datatypes_example,
		"send_all_datatypes_lean": send_all_datatypes_lean_example,
	}

def parse_args():
	"""
	Parse command line arguments for running examples.
	Returns:
		argparse.Namespace: Parsed arguments.
	"""
	parser = argparse.ArgumentParser(description="Run vitreous examples")
	parser.add_argument(
		"--example",
		type=str,
		help="Name of the example to run (without _example suffix) or use --list to see all available examples",
	)
	parser.add_argument(
		"--list", action="store_true", help="List all available examples"
	)
	return parser.parse_args()

def main():
	"""
	Main entry point for running the selected example from the command line.
	Handles argument parsing, example selection, and error reporting.
	"""
	args = parse_args()

	example_dict = get_example_dict()

	if args.list:
		logger.success("Available examples:")
		for example_name in sorted(example_dict.keys()):
			logger.success(f"  - {example_name}")
		return

	if not args.example:
		logger.error(
			"Please provide --example or use --list to see all available examples."
		)
		raise SystemExit(1)

	args.example = args.example.lower()

	if args.example not in example_dict:
		logger.error(f"Example '{args.example}' not found.")

		# Find the one that is nearest in name
		close_matches = difflib.get_close_matches(
			args.example, example_dict.keys(), n=3, cutoff=0.4
		)

		if close_matches:
			logger.error("Did you mean one of these?")
			for match in close_matches:
				logger.error(f"  - {match}")

		raise SystemExit(1)

	logger.success(f"Running {args.example} example...")
	example_dict[args.example]()
	logger.success(f"{args.example} example completed.")

if __name__ == "__main__":
	main()
