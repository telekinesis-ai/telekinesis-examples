import argparse
import difflib
import pathlib
import numpy as np
from loguru import logger
import rerun as rr
from rerun import blueprint as rrb
from scipy.spatial.transform import Rotation as R
import cv2
import time

from telekinesis import vitreous
from datatypes import datatypes, io

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
# Data directory path - points to the telekinesis-data git submodule
# To use a custom data location, change this path
DATA_DIR = ROOT_DIR / "telekinesis-data"

# Calculation examples
	  
def calculate_axis_aligned_bounding_box_example():
	"""
	Computes the axis-aligned bounding box (AABB) of a point cloud.

	Finds the smallest box aligned with coordinate axes that contains all points.
	Returns min/max coordinates along each axis.
	"""

	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "can_vertical_1_raw_preprocessed.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	axis_aligned_bounding_box = vitreous.calculate_axis_aligned_bounding_box(point_cloud=point_cloud)
	logger.success(
		f"Calculated axis-aligned bounding box for {len(point_cloud.positions)} points: with half-size: {axis_aligned_bounding_box.half_size} and center: {axis_aligned_bounding_box.center}"
	)

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("calculate_axis_aligned_bounding_box", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([ 530.98295975, -654.07763437,  128.73893843])
	look_target = np.array([-39.32096225, -77.51841498, 602.26898493])
	eye_up = np.array([ 0.02837839, -0.57508985, -0.8175979 ])
 
	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,  
		speed=0.0,  
		tracking_entity=None, 
	)

	# Send blueprint
	rr.send_blueprint(
		rrb.Blueprint(
			rrb.Horizontal(
				rrb.Spatial3DView(
					name="Input Point Cloud",
					origin="input_point_cloud",
					background=background,
					eye_controls=eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information
				),
				rrb.Spatial3DView(
					name="Axis-Aligned Bounding Box Overlay",
					origin="aabb_overlay",
					background=background,
					eye_controls=eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information
				),
			)
		)
	)

	# Log the axis-aligned bounding box as a box and overlay it on the point cloud
	rr.log("aabb_overlay", rr.Points3D(positions=point_cloud.positions,
		   colors=point_cloud.colors))
	quaternions = R.from_euler('xyz', axis_aligned_bounding_box.rotation_in_euler_angles, degrees=True).as_quat()

	rr.log("aabb_overlay", rr.Boxes3D(
			half_sizes=axis_aligned_bounding_box.half_size,
			centers=axis_aligned_bounding_box.center,
			colors=np.array([[0, 255, 0]]),  # Green color for bounding box
			quaternions=quaternions
	))
	# Log the input point cloud under input_point_cloud
	rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
		   colors=point_cloud.colors))


def calculate_oriented_bounding_box_example():
	"""
	Computes the oriented bounding box (OBB) of a point cloud.

	Finds the smallest box (in any orientation) that contains all points.
	Returns box parameters including center, extents, and rotation angles.
	"""

	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "can_vertical_1_raw_obb_preprocessed.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	result_bbox = vitreous.calculate_oriented_bounding_box(point_cloud=point_cloud)
	logger.success(
		f"Calculated oriented bounding box for {len(point_cloud.positions)} points with half-size: {result_bbox.half_size}, center: {result_bbox.center}, rotation_in_euler_angles: {result_bbox.rotation_in_euler_angles}"
	)

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("calculate_oriented_bounding_box", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([ 1800.43642, 0.23659945, -500.517691])
	look_target = np.array([ 0.43641739, 0.23659945, -0.51769066])
	eye_up = np.array([ 0., 0., -1.])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,  
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(
		rrb.Blueprint(
			rrb.Horizontal(
				rrb.Spatial3DView(
					name="Input Point Cloud",
					origin="input_point_cloud",
					background=background,
					eye_controls=eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information
				),
				rrb.Spatial3DView(
					name="Oriented Bounding Box Overlay",
					origin="obb_overlay",
					background=background,
					eye_controls=eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information
				),
			)
		)
	)

	# Log the oriented bounding box as a box and overlay it on the point cloud
	# Log the input point cloud under input_point_cloud
	rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
		   colors=point_cloud.colors))
	rr.log("obb_overlay", rr.Points3D(positions=point_cloud.positions,
		   colors=point_cloud.colors))
	
	# Log the oriented bounding box
	quaternions = R.from_euler('xyz', result_bbox.rotation_in_euler_angles, degrees=True).as_quat()
	rr.log("obb_overlay", rr.Boxes3D(
		half_sizes=result_bbox.half_size,
		centers=result_bbox.center,
		colors=np.array([[0, 255, 0]]),  # Green color for bounding box
		quaternions=quaternions
	))


def calculate_plane_normal_example():
	"""
	Extracts the normal vector from plane coefficients.

	Demonstrates extracting and normalizing the normal vector from plane equation
	coefficients (ax + by + cz + d = 0).
	"""

	# ===================== Operation ==========================================

	# Execute operation
	normal_vector = vitreous.calculate_plane_normal(plane_coefficients=[0.0, 0.0, 1.0, 0.0])
	logger.success(
		f"Calculated normal vector to {normal_vector}"
	)

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("calculate_plane_normal", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Spatial3DView(name="Plane with Normal Vector", origin="plane_visualization"),
	))

	# Extract plane parameters: ax + by + cz + d = 0
	plane_point = np.array([0, 0, 0])

	# Create two orthogonal vectors in the plane
	if abs(normal_vector[2]) < 0.9:
		u = np.cross(normal_vector, np.array([0, 0, 1]))
	else:
		u = np.cross(normal_vector, np.array([1, 0, 0]))
	u = u / np.linalg.norm(u)
	v = np.cross(normal_vector, u)
	v = v / np.linalg.norm(v)

	# Create a grid of points on the plane to visualize it
	plane_size = 100.0
	grid_density = 20
	plane_points = []
	for i in np.linspace(-plane_size, plane_size, grid_density):
		for j in np.linspace(-plane_size, plane_size, grid_density):
			point = plane_point + i * u + j * v
			plane_points.append(point)

	# Log the plane as points
	rr.log("plane_visualization/plane", rr.Points3D(
		np.array(plane_points),
		colors=[[200, 200, 200]] * len(plane_points),
		radii=[0.5] * len(plane_points)
	))

	# Log the plane point (origin on the plane)
	rr.log("plane_visualization/plane_point", rr.Points3D(
		np.array([plane_point]),
		colors=[[255, 255, 0]],
		radii=[2.0]
	))

	# Log the normal vector as an arrow from the plane point
	normal_length = 50.0
	rr.log("plane_visualization/normal_vector", rr.Arrows3D(
		origins=np.array([plane_point]),
		vectors=np.array([normal_vector * normal_length]),
		colors=np.array([[255, 0, 0]])
	))


def calculate_point_cloud_centroid_example():
	"""
	Computes the geometric center (centroid) of a point cloud.

	Calculates the mean position of all points in the cloud.
	"""

	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "zivid_large_pcb_inspection_cropped_preprocessed.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	centroid = vitreous.calculate_point_cloud_centroid(point_cloud=point_cloud)
	logger.success(
		f"Calculated centroid {centroid} for {len(point_cloud.positions)} points"
	)

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("calculate_point_cloud_centroid", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([250., 375., 250.])
	look_target = np.array([0, 0, 0])
	eye_up = np.array([0., 0., 1.])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,  
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(
		rrb.Blueprint(
			rrb.Horizontal(
				rrb.Spatial3DView(
					name="Input Point Cloud (Box-Filtered)",
					origin="input",
					background=background,
					eye_controls=eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information
				),
				rrb.Spatial3DView(
					name="Base Points + Frames Overlay",
					origin="output",
					background=background,
					eye_controls=eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information
				),
			)
		)
	)

	# Log input point cloud
	rr.log("input", rr.ViewCoordinates.RDB, static=True)
	rr.log("input", rr.Points3D(positions=point_cloud.positions,
		   colors=point_cloud.colors))
	
	# Log output point cloud and centroid
	rr.log("output", rr.ViewCoordinates.RDB, static=True)
	# Object cloud
	rr.log("output/object", rr.Points3D(positions=point_cloud.positions,
		   colors=point_cloud.colors))
	# Centroid point
	rr.log("output/centroid", rr.Points3D(positions=centroid, colors=(255, 0, 0)))

	
	# Log world-aligned frame axes (identity orientation)
	frame_scale = 100
	x_axis = np.array([frame_scale, 0, 0])
	y_axis = np.array([0, frame_scale, 0])
	z_axis = np.array([0, 0, frame_scale])

	axes_single = np.stack([x_axis, y_axis, z_axis], 0)
	axis_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

	origins = np.repeat(centroid.reshape(1, 3), 3, axis=0)
	vectors = np.tile(axes_single, (1, 1))
	colors = np.tile(axis_colors, (1, 1))

	rr.log("output/base_frames", rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))


def calculate_points_in_point_cloud_example():
	"""
	Counts the number of points in a point cloud.

	Simple utility that returns the total point count.
	"""

	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "can_vertical_1_raw.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	num_points = vitreous.calculate_points_in_point_cloud(point_cloud=point_cloud)
	logger.success(f"Counted {num_points.value} points in point cloud")

# Clustering examples

def cluster_point_cloud_using_dbscan_example():
	"""
	Clusters a point cloud using the DBSCAN density-based clustering algorithm.

	DBSCAN identifies clusters of points that are closely packed together,
	separating distinct objects or regions.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "zivid_bottles_10_preprocessed.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")
	
	# Execute operation
	clusters = vitreous.cluster_point_cloud_using_dbscan(
		point_cloud=point_cloud,
		max_distance=20,
		min_points=50,
	)
	logger.success(
		f"Clustered point cloud with {len(point_cloud.positions)} points using DBSCAN into {clusters.__len__()} clusters"
	)

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("cluster_point_cloud_using_dbscan", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([227.10778553, 335.22947723, 305.59192904])
	look_target = np.array([-22.89221447, -39.77052277,  55.59192904])
	eye_up = np.array([0., 0., 1.])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,  
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	num_output_clouds = len(clusters) if isinstance(clusters, list) else 1
	output_views = [
		rrb.Spatial3DView(name=f"Output {i+1}", origin=f"dbscan_cluster/cluster_{i+1}")
		for i in range(num_output_clouds)
	]
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(name="Input Point Cloud",
							  origin="input_point_cloud",
							  eye_controls=eye_controls,
							  background=background,
							  spatial_information=spatial_information,
							  line_grid=line_grid),
			rrb.Spatial3DView(name="Output Point Cloud", origin="dbscan_cluster",
							  eye_controls=eye_controls,
							  background=background,
							  spatial_information=spatial_information,
							  line_grid=line_grid),
		)
	))

	# Log the input point cloud under input_point_cloud
	rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
		   colors=point_cloud.colors))  

	# Log each output point cloud under its own path
	for i, cluster in enumerate(clusters.to_list()):
		rr.log(f"dbscan_cluster/cluster_{i+1}", rr.Points3D(positions=cluster.positions,
				   colors=cluster.colors))


def cluster_point_cloud_based_on_density_jump_example():
	"""
	Splits a point cloud into regions based on density discontinuities.

	Detects and splits point clouds at locations where point density changes
	dramatically.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "mug_preprocessed.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	clusters = vitreous.cluster_point_cloud_based_on_density_jump(
		point_cloud=point_cloud,
		num_nearest_neighbors=5,
		neighborhood_radius=0.05,
		is_point_cloud_linear=False,
		projection_axis=[0.0, 0.0, 1.0],   
	)
	logger.success(
		"Split point cloud based on density jump"
	)

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("cluster_point_cloud_based_on_density_jump", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([-2.5, 2.5, 5.])
	look_target = np.array([ 9.22595333e-18, -1.26803568e-15, -3.16024984e-17])
	eye_up = np.array([0., 1., 0.])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,  
		speed=0.0,  
		tracking_entity=None,  
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(name="Input Point Cloud",
							  origin="input_point_cloud",
							  eye_controls=eye_controls,
							  background=background,
							  spatial_information=spatial_information,
							  line_grid=line_grid),
			rrb.Spatial3DView(name="Output Point Cloud", origin="splitted_clusters",
							  eye_controls=eye_controls,
							  background=background,
							  spatial_information=spatial_information,
							  line_grid=line_grid),
		)
	))

	rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
		   colors=point_cloud.colors))

	# Log each output point cloud under its own path
	for i, cluster in enumerate(clusters.to_list()):
			rr.log(f"splitted_clusters/cluster_{i+1}", rr.Points3D(positions=cluster.positions,
				   colors=cluster.colors))

# Conversion examples

def convert_mesh_to_point_cloud_example():
	"""
	Converts a triangle mesh to a point cloud via surface sampling.

	Samples points on the mesh surface using uniform or Poisson disk sampling.
	"""

	# ===================== Operation ==========================================

	# Load mesh
	filepath = str(DATA_DIR / "meshes" / "gear_box.glb")
	mesh = io.load_mesh(filepath=filepath)
	logger.success(f"Loaded mesh with {len(mesh.vertex_positions)} vertices")

	# Execute operation
	point_cloud = vitreous.convert_mesh_to_point_cloud(
		mesh=mesh,
		num_points=10000,
		sampling_method="poisson_disk",
		initial_sampling_factor=5,
		initial_point_cloud=None,
		use_triangle_normal=False,
	)
	logger.success(
		f"Converted mesh with {len(mesh.vertex_positions)} vertices to point cloud"
	)


	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("convert_mesh_to_point_cloud", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	look_target = np.array([0.08670620230211261, 0.030510931062866967, -0.09899483804857363])
	offset = np.array([99.91329379769789, 149.96948906893712, 100.09899483804857])
	camera_eye_position = look_target + offset
	eye_up = np.array([0.0, 0.0, 1.0])
	zoom_out_factor = 4

	vec = camera_eye_position - look_target
	dir_vec = vec / np.linalg.norm(vec)
	overview_position = look_target + dir_vec * (np.linalg.norm(vec) * zoom_out_factor)

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(name="Input Mesh", origin="input_mesh",
							  background=background,
							  eye_controls=eye_controls,
							  line_grid=line_grid,
							  spatial_information=spatial_information),
			rrb.Spatial3DView(name="Output Point Cloud",
							  origin="output_point_cloud",
							  background=background,
							  eye_controls=eye_controls,
							  line_grid=line_grid,
							  spatial_information=spatial_information),
		)
	))

	# Log the input mesh under input_mesh
	rr.log("input_mesh", rr.Mesh3D(
		vertex_positions=mesh.vertex_positions,
		triangle_indices=mesh.triangle_indices,
		vertex_colors=mesh.vertex_colors,
		vertex_normals=mesh.vertex_normals,
		albedo_factor=[0.8, 0.8, 0.8, 1.0],
	))

	# Log the output point cloud under output_point_cloud
	rr.log("output_point_cloud", rr.Points3D(positions=point_cloud.positions,
											 colors=point_cloud.colors))

# Mesh creation examples

def create_cylinder_mesh_example():
	"""
	Creates a parametric cylinder mesh.

	Generates a cylinder with specified radius, height, and resolution.
	"""
	# ===================== Operation ==========================================

	# Execute operation
	cylinder_mesh = vitreous.create_cylinder_mesh(
		radius=0.01,
		height=0.02,
		radial_resolution=20,
		height_resolution=4,
		retain_base=False,
		vertex_tolerance=1e-6,
		transformation_matrix=np.eye(4, dtype=np.float32),
		compute_vertex_normals=True,
	)
	logger.success("Created cylinder mesh")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("create_cylinder_mesh", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=True)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([0.02670, 0.04005, 0.02670])
	look_target = np.array([0, 0, 0])
	eye_up = np.array([0., 0., 1.])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(
		rrb.Blueprint(
			rrb.Spatial3DView(
				name="Cylinder Mesh",
				origin="cylinder_mesh",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information
			),
		)
	)

	rr.log("cylinder_mesh", rr.Mesh3D(
		vertex_positions=cylinder_mesh.vertex_positions,
		triangle_indices=cylinder_mesh.triangle_indices,
		vertex_normals=cylinder_mesh.vertex_normals,
		albedo_factor=[0.8, 0.8, 0.8, 1.0],
	))


def create_plane_mesh_example():
	"""
	Creates a rectangular plane mesh (thin box).

	Generates a flat rectangular surface with specified dimensions.
	"""
	# ===================== Operation ==========================================

	# Execute operation
	plane_mesh = vitreous.create_plane_mesh(
		transformation_matrix=np.eye(4, dtype=np.float32),
		x_dimension=0.01,
		y_dimension=0.01,
		z_dimension=0.00001,
		compute_vertex_normals=True,
	)
	logger.success("Created plane mesh")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("create_plane_mesh", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([0.02670, 0.04005, 0.02670])
	look_target = np.array([0, 0, 0])
	eye_up = np.array([0., 0., 1.])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(
		rrb.Blueprint(
			rrb.Spatial3DView(
				name="Plane Mesh",
				origin="plane_mesh",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information
			),
		)
	)


	rr.log("plane_mesh", rr.Mesh3D(
		vertex_positions=plane_mesh.vertex_positions,
		triangle_indices=plane_mesh.triangle_indices,
		vertex_normals=plane_mesh.vertex_normals,
		albedo_factor=[0.8, 0.8, 0.8, 1.0],
	))


def create_sphere_mesh_example():
	"""
	Creates a UV sphere mesh.

	Generates a spherical mesh with specified radius and resolution.
	"""
	# ===================== Operation ==========================================

	# Execute operation
	sphere_mesh = vitreous.create_sphere_mesh(
		transformation_matrix=np.eye(4, dtype=np.float32),
		radius=0.01,
		resolution=20,
		compute_vertex_normals=True   
	)
	logger.success("Created sphere mesh")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("create_sphere_mesh", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([0.02670, 0.04005, 0.02670])
	look_target = np.array([0, 0, 0])
	eye_up = np.array([0., 0., 1.])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(
		rrb.Blueprint(
			rrb.Spatial3DView(
				name="Sphere Mesh",
				origin="sphere_mesh",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information
			),
		)
	)

	rr.log("sphere_mesh", rr.Mesh3D(
		vertex_positions=sphere_mesh.vertex_positions,
		triangle_indices=sphere_mesh.triangle_indices,
		vertex_normals=sphere_mesh.vertex_normals,
		albedo_factor=[0.8, 0.8, 0.8, 1.0],
	))


def create_torus_mesh_example():
	"""
	Creates a torus (donut shape) mesh.

	Generates a parametric torus with specified major/minor radii and resolution.
	"""
	# ===================== Operation ==========================================

	# Execute operation
	torus_mesh = vitreous.create_torus_mesh(
		transformation_matrix=np.eye(4, dtype=np.float32),
		torus_radius=0.01,
		tube_radius=0.005,
		radial_resolution=20,
		tubular_resolution=10,
		compute_vertex_normals=True,
	)
	logger.success(
		"Created torus mesh"
	)

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("create_torus_mesh", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([0.02670, 0.04005, 0.02670])
	look_target = np.array([0, 0, 0])
	eye_up = np.array([0., 0., 1.])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(
		rrb.Blueprint(
			rrb.Spatial3DView(
				name="Torus Mesh",
				origin="torus_mesh",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information
			),
		)
	)

	rr.log("torus_mesh", rr.Mesh3D(
		vertex_positions=torus_mesh.vertex_positions,
		triangle_indices=torus_mesh.triangle_indices,
		vertex_normals=torus_mesh.vertex_normals,
		albedo_factor=[0.8, 0.8, 0.8, 1.0],
	))

# Estimation examples

def estimate_principal_axis_within_radius_example():
	"""
	Estimates the principal component axis of a point cloud neighborhood.

	Uses PCA to find the dominant direction in a local neighborhood around a
	reference point.
	"""
	
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "mug_preprocessed.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	reference_point = np.array([0., 0., -0.52], dtype=np.float32)
	neighborhood_radius = .25

	# Execute operation
	local_principal_axis = vitreous.estimate_principal_axis_within_radius(
		point_cloud=point_cloud,
		neighborhood_radius=neighborhood_radius,
		reference_point=reference_point,
	)
	logger.success(
		"Estimated principal axis within radius"
	)

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("estimate_principal_axis_within_radius", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()
  	
   # Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([-1.25, 1.25, 2.5])
	look_target = np.array([ 9.22595333e-18, -1.26803568e-15, -3.16024984e-17])
	eye_up = np.array([0., 1., 0.])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)


	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(name="Input Point Cloud",
							  origin="input_point_cloud",
							  eye_controls=eye_controls,
							  background=background,
							  spatial_information=spatial_information,
							  line_grid=line_grid),
		)
	))

	rr.log("input_point_cloud", rr.Points3D(np.asarray(point_cloud.positions),
		   colors=(np.asarray(point_cloud.colors))))
	# Visualize reference point (green sphere)
	rr.log("input_point_cloud/reference_point", rr.Points3D(
		np.array([reference_point]),
		colors=np.array([[255, 0, 0]]),
		radii=[.03]
	))

	# Visualize neighborhood sphere boundary (yellow)
	u = np.linspace(0, 2 * np.pi, 60)
	v = np.linspace(0, np.pi, 40)
	x = neighborhood_radius * np.outer(np.cos(u), np.sin(v)) + reference_point[0]
	y = neighborhood_radius * np.outer(np.sin(u), np.sin(v)) + reference_point[1]
	z = neighborhood_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + reference_point[2]
	sphere_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
	rr.log("input_point_cloud/neighborhood_sphere", rr.Points3D(
		sphere_points,
		colors=[[255, 255, 0]] * len(sphere_points),
		# radii=[1.0] * len(sphere_points)
	))


	# Visualize principal axis arrow (red)
	rr.log("input_point_cloud/arrow", rr.Arrows3D(
		origins=np.array(reference_point),
		vectors=np.array(local_principal_axis),
		colors=np.array([[255, 0, 0]]),
		radii=0.1
	))


def estimate_principal_axes_example():
	"""
	Computes the principal axes of a point cloud using PCA.

	Finds the orthogonal axes along which the point cloud has maximum variance.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "zivid_large_pcb_inspection_cropped_preprocessed.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	principal_axes = vitreous.estimate_principal_axes(
		point_cloud=point_cloud,
		method="obb",
	)
	logger.success("Estimated principal axes")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("estimate_principal_axes", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([250., 375., 250.])
	look_target = np.array([0, 0, 0])
	eye_up = np.array([0., 0., 1.])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(
		rrb.Blueprint(
			rrb.Horizontal(
				rrb.Spatial3DView(
					name="Input Point Cloud", origin="input_point_cloud",
					background=background,
					eye_controls=eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information
				),
				rrb.Spatial3DView(
					name="Principal Axes Overlay", origin="principal_axes_overlay",
					background=background,
					eye_controls=eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information
				),
			)
		)
	)

	# Log the input point cloud under input_point_cloud
	rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
		   colors=point_cloud.colors))

	# Calculate the centroid of the point cloud

	# Principal axes is a (3, 3) matrix where each column is an axis (unit vector)
	# Scale the axes to be visible (adjust scale_factor based on point cloud size)
	points = point_cloud.positions
	bbox_size = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
	scale_factor = bbox_size * 0.5  # Scale to 30% of bounding box diagonal

	# Extract the three principal axes (columns) and scale them
	axis1 = principal_axes[:, 0] * scale_factor  # First (largest variance)
	axis2 = principal_axes[:, 1] * scale_factor  # Second
	axis3 = principal_axes[:, 2] * scale_factor  # Third (smallest variance)

	# Log the principal axes as arrows originating from the centroid
	rr.log("principal_axes_overlay/points", rr.Points3D(positions=point_cloud.positions,
		   colors=point_cloud.colors))
	
	rr.log("principal_axes_overlay/axes", rr.Arrows3D(
		origins=look_target,
		vectors=axis1,
		colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
		radii=4))  # RGB for 1st, 2nd, 3rd axes - radii controls arrow thickness

# Filtering examples

def filter_point_cloud_using_pass_through_filter_example():
	"""
	Filters points within axis-aligned min/max ranges.

	Keeps only points where each coordinate (x, y, z) falls within specified
	min/max bounds.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "mounts_3_raw.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	x_min, y_min, z_min, x_max, y_max, z_max = np.array([-185.0, -164.0, 450.0, 230.0, 164.0, 548.0])

	# Execute operation
	filtered_point_cloud = vitreous.filter_point_cloud_using_pass_through_filter(
		x_min=x_min,
		x_max=x_max,
		y_min=y_min,
		y_max=y_max,
		z_min=z_min,
		z_max=z_max,
		point_cloud=point_cloud,
	)
	logger.success("Filtered points using axis-aligned range")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("filter_point_cloud_using_pass_through_filter", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	look_target = np.array([17.246607005843458, -10.312582127251696, 495.8964079473356])
	offset = np.array([640.2690590947808, -332.28717547581937, -727.8142110040502])
	camera_eye_position = look_target + offset
	eye_up = np.array([0.040600170502887244, 0.009404387964181355, -0.9991312144268918])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=camera_eye_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud",
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Filtered Point Cloud",
				origin="filtered_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Log the input point cloud 
	rr.log("input_point_cloud/points", rr.Points3D(positions=point_cloud.positions,
			   colors=point_cloud.colors))

	# Log the passthrough filter box on the same view
	box_corners = np.array([
		[x_min, y_min, z_min],
		[x_max, y_min, z_min],
		[x_max, y_max, z_min],
		[x_min, y_max, z_min],
		[x_min, y_min, z_max],
		[x_max, y_min, z_max],
		[x_max, y_max, z_max],
		[x_min, y_max, z_max],
	])
	box_lines = np.array([
		[0, 1], [1, 2], [2, 3], [3, 0],
		[4, 5], [5, 6], [6, 7], [7, 4],
		[0, 4], [1, 5], [2, 6], [3, 7],
	])
	rr.log("input_point_cloud/filter_box", rr.LineStrips3D([box_corners[line]
		   for line in box_lines], colors=np.array([[255, 0, 0]])))

	# Log the filtered point cloud with color handling
	rr.log("filtered_point_cloud", rr.Points3D(positions=filtered_point_cloud.positions,
			   colors=filtered_point_cloud.colors))


def filter_point_cloud_using_bounding_box_example():
	"""
	Filters points within an axis-aligned bounding box.

	Keeps only points that fall within the specified 3D box defined by
	min/max coordinates along each axis.
	"""

	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "plastic_2_raw.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	# Create Box
	x_min, y_min, z_min, x_max, y_max, z_max = np.array([-163, -100, 470, 150, 100, 544])
	center = np.array(
		[[(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]],
		dtype=np.float32,
	)
	half_size = np.array(
		[[(x_max - x_min) / 2, (y_max - y_min) / 2, (z_max - z_min) / 2]],
		dtype=np.float32,
	)
	colors = [(255, 0, 0)]
	bbox = datatypes.Boxes3D(half_size=half_size, center=center, colors=colors)

	# Filter point cloud using bounding box
	filtered_point_cloud = vitreous.filter_point_cloud_using_bounding_box(
		point_cloud=point_cloud, bbox=bbox
	)

	logger.success(
		f"Filtered {len(filtered_point_cloud.positions)} points using bounding box"
	)


	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("filter_point_cloud_using_bounding_box", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	look_target = np.array([-16.837628596149877, 12.493554094779665, 516.8440399654662])
	offset = np.array([190.90496669008934, -508.22473543952464, -406.8421301004321])
	camera_eye_position = look_target + offset
	eye_up = np.array([0.17020111661232978, -0.054036235719827574, -0.9839266563789942])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=camera_eye_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud", 
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Filtered Point Cloud",
				origin="filtered_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Log the input point cloud under input_point_cloud
	rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
			   colors=point_cloud.colors))
	# Log the filtered point cloud with color handling
	rr.log("filtered_point_cloud", rr.Points3D(positions=filtered_point_cloud.positions,
			   colors=filtered_point_cloud.colors))

	# add the bbox to the point cloud
	rr.log(
		"input_point_cloud/bbox",
		rr.Boxes3D(
			half_sizes=bbox.half_size,
			centers=bbox.center,
			colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], 
		)
	)


def filter_point_cloud_using_cylinder_base_removal_example():
	"""
	Removes the base faces from a cylindrical mesh.

	Identifies and removes triangles that form the flat base(s) of a cylinder,
	leaving only the curved side surface.
	"""
	# ===================== Operation ==========================================

	# Load mesh
	filepath = str(DATA_DIR / "meshes" / "beer_can.glb")
	mesh = io.load_mesh(filepath=filepath)
	logger.success("Loaded mesh")
	
	# Execute operation
	filtered_mesh = vitreous.filter_point_cloud_using_cylinder_base_removal(
		mesh=mesh,
		compute_vertex_normals=True,
		distance_threshold=0.005
	)

	logger.success("Filtered mesh using cylinder base removal")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("filter_point_cloud_using_cylinder_base_removal", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	verts = np.asarray(filtered_mesh.vertex_positions)
	bbox_min = verts.min(axis=0)
	bbox_max = verts.max(axis=0)
	mesh_center = 0.5 * (bbox_min + bbox_max)

	look_target = mesh_center
	eye_up = np.array([0.0, 0.0, 0.1])
	offset = eye_up * 2
	position = look_target + offset
	
	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Mesh",
				origin="input_mesh",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Filtered Mesh",
				origin="filtered_mesh",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Log the input mesh under input_mesh
	rr.log("input_mesh", rr.Mesh3D(
		vertex_positions=mesh.vertex_positions,
		triangle_indices=mesh.triangle_indices,
		vertex_colors=mesh.vertex_colors,
		vertex_normals=mesh.vertex_normals,
		albedo_factor=[0.8, 0.8, 0.8, 1.0],
	))

	# Log the output filtered mesh under filtered_mesh
	rr.log("filtered_mesh", rr.Mesh3D(
		vertex_positions=filtered_mesh.vertex_positions,
		triangle_indices=filtered_mesh.triangle_indices,
		vertex_colors=filtered_mesh.vertex_colors,
		vertex_normals=filtered_mesh.vertex_normals,
		albedo_factor=[0.8, 0.8, 0.8, 1.0],
	))
	

def filter_point_cloud_using_mask_example():
	"""
	Filters a structured point cloud using a 2D binary mask.

	Applies a 2D image mask to an organized point cloud, keeping only points
	where the corresponding pixel is True.
	"""
	# ===================== Operation ==========================================

	# Load point cloud and mask
	filepath1 = str(DATA_DIR / "point_clouds" / "can_vertical_6_raw.ply")
	print("Loading point cloud from: ", filepath1)
	point_cloud = io.load_point_cloud(filepath=filepath1, 
											remove_duplicated_points=False, 
											remove_infinite_points=False, 
											remove_nan_points=False)

	filepath2 = str(DATA_DIR / "images" / "can_vertical_6_mask.png")
	mask = cv2.imread(filepath2, cv2.IMREAD_GRAYSCALE)


	logger.success("Loaded point cloud and mask")
   
	# Execute operation
	result_point_cloud = vitreous.filter_point_cloud_using_mask(
		point_cloud=point_cloud, 
		mask=mask,
	)
	logger.success("Filtered points using mask")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("filter_point_cloud_using_mask", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([865, -865, 165])
	look_target = np.array([-9.09364389, -78.71465444, 598.47233982])
	eye_up = np.array([0.02736525, -0.56736208, -0.82301361])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud",
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Vertical(
				rrb.Spatial2DView(name="Binary Mask", origin="binary_mask"),
			),
			rrb.Spatial3DView(
				name="Masked Point Cloud",
				origin="masked_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Log the input point cloud
	rr.log("input_point_cloud", rr.ViewCoordinates.RDB, static=True)
	rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
			   colors=point_cloud.colors))
	
	# Log and mask
	rr.log("binary_mask", rr.Image(mask, color_model="L"))

	# Log the filtered point cloud
	rr.log("masked_point_cloud", rr.ViewCoordinates.RDB, static=True)
	rr.log("masked_point_cloud", rr.Points3D(positions=result_point_cloud.positions,
			   colors=result_point_cloud.colors))


def filter_point_cloud_using_oriented_bounding_box_example():
	"""
	Filters points within an oriented (rotated) bounding box.

	Keeps only points within a 3D box that can be rotated to any orientation.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "can_vertical_3_downsampled.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	x_min = -205.65248652
	y_min = -112.59310319
	z_min = 554.42936219
	x_max = 121.88022318
	y_max = -17.60647882
	z_max = 698.54912862
	rot_x = -38.1245801
	rot_y = -7.89877607
	rot_z = -7.74440359    

	half_size = np.array(
		[[(x_max - x_min) / 2, (y_max - y_min) / 2, (z_max - z_min) / 2]],
		dtype=np.float32,
	)
	center = np.array(
		[[(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]],
		dtype=np.float32,
	)
	rotation_in_euler_angles = np.array([[rot_x, rot_y, rot_z]], dtype=np.float32)
	oriented_bbox = datatypes.Boxes3D(
		half_size=half_size,
		center=center,
		rotation_in_euler_angles=rotation_in_euler_angles,
	)

	# Filter point cloud using oriented bounding box
	filtered_point_cloud = vitreous.filter_point_cloud_using_oriented_bounding_box(
		point_cloud=point_cloud, oriented_bbox=oriented_bbox
	)
	logger.success("Filtered points using oriented bounding box")


	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("filter_point_cloud_using_oriented_bounding_box", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	look_target = np.array([-39.32096224866192, -77.51841497655289, 602.2689849331848])
	offset = np.array([587.536048681736, -530.2253280469823, -473.4666019099898])
	camera_eye_position = look_target + offset
	eye_up = np.array([0.02844979765608562, -0.5751413943408177, -0.8175591633203237])

	logger.success(f"Camera eye position: {camera_eye_position}, look target: {look_target}, eye up: {eye_up}")

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=camera_eye_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud",
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Filtered Point Cloud",
				origin="filtered_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Log the input point cloud under input_point_cloud
	rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
											colors=point_cloud.colors))
	
	# Log the output point cloud under output_point_cloud
	rr.log("filtered_point_cloud", rr.Points3D(positions=filtered_point_cloud.positions,
												colors=filtered_point_cloud.colors))
	
	# Log the oriented bounding box on the input view
	quaternions = R.from_euler('xyz', oriented_bbox.rotation_in_euler_angles, degrees=True).as_quat()
	rr.log("input_point_cloud/oriented_bbox", rr.Boxes3D(
		half_sizes=oriented_bbox.half_size,
		centers=oriented_bbox.center,
		quaternions=quaternions,
		colors=[(255, 0, 0)],
	))

	# # Visualize the oriented bounding box as line strips on the input
	# # Convert OBB parameters to 8 corners
	# x_min, y_min, z_min, x_max, y_max, z_max, roll, pitch, yaw = oriented_bbox

	# # Convert angles from degrees to radians
	# roll_rad = np.deg2rad(roll)
	# pitch_rad = np.deg2rad(pitch)
	# yaw_rad = np.deg2rad(yaw)

	# # Build rotation matrix from Euler angles
	# R = R_scipy.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad]).as_matrix()

	# # Center and half-extents in LOCAL frame
	# center_local = np.array([(x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2])
	# half_extent = np.array([(x_max - x_min)/2, (y_max - y_min)/2, (z_max - z_min)/2])

	# # 8 corners in LOCAL frame (centered at origin)
	# corners_local = np.array([
	#     [-half_extent[0], -half_extent[1], -half_extent[2]],
	#     [half_extent[0], -half_extent[1], -half_extent[2]],
	#     [half_extent[0],  half_extent[1], -half_extent[2]],
	#     [-half_extent[0],  half_extent[1], -half_extent[2]],
	#     [-half_extent[0], -half_extent[1],  half_extent[2]],
	#     [half_extent[0], -half_extent[1],  half_extent[2]],
	#     [half_extent[0],  half_extent[1],  half_extent[2]],
	#     [-half_extent[0],  half_extent[1],  half_extent[2]],
	# ])

	# # Transform to WORLD frame
	# obb_corners = (R @ corners_local.T).T + center_local

	# # Create line strips for the 12 edges of the box
	# obb_lines = [
	#     np.stack([obb_corners[0], obb_corners[1]]),
	#     np.stack([obb_corners[1], obb_corners[2]]),
	#     np.stack([obb_corners[2], obb_corners[3]]),
	#     np.stack([obb_corners[3], obb_corners[0]]),
	#     np.stack([obb_corners[4], obb_corners[5]]),
	#     np.stack([obb_corners[5], obb_corners[6]]),
	#     np.stack([obb_corners[6], obb_corners[7]]),
	#     np.stack([obb_corners[7], obb_corners[4]]),
	#     np.stack([obb_corners[0], obb_corners[4]]),
	#     np.stack([obb_corners[1], obb_corners[5]]),
	#     np.stack([obb_corners[2], obb_corners[6]]),
	#     np.stack([obb_corners[3], obb_corners[7]]),
	# ]
	# line_colors = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (len(obb_lines), 1))
	# rr.log("input_point_cloud/oriented_bbox", rr.LineStrips3D(obb_lines, colors=line_colors))


def filter_point_cloud_using_plane_defined_by_point_normal_proximity_example():
	"""
	Filters points near a plane defined by a point and normal vector.

	Keeps points within a distance threshold of a plane specified by a point
	on the plane and its normal vector.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "can_vertical_3_downsampled.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	filtered_point_cloud = (
		vitreous.filter_point_cloud_using_plane_defined_by_point_normal_proximity(
			distance_threshold=4.0,
			point_cloud=point_cloud,
			plane_point=[-15.74520074, 319.25105712, 454.3114797],
			plane_normal=[0.028344755192329624, -0.5747207168510667, -0.8178585895344518],
		)
	)
	logger.success("Filtered points using plane defined by point and normal")


	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("filter_point_cloud_using_plane_defined_by_point_normal_proximity", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([ 508.26353625, -457.76445726,  289.53896696])
	look_target = np.array([ 21.82833776,  -6.47603561, 684.12881138])
	eye_up = np.array([ 0.02846775, -0.57502007, -0.81764387])


	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud",
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Filtered Point Cloud",
				origin="filtered_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))


	# Log the input point cloud under input_point_cloud + plane
	rr.log("input_point_cloud", rr.Points3D(point_cloud.positions,
			   colors=point_cloud.colors))

	# Log the output point cloud
	rr.log("filtered_point_cloud", rr.Points3D(np.asarray(filtered_point_cloud.positions),
			   colors=filtered_point_cloud.colors))


def filter_point_cloud_using_plane_proximity_example():
	"""
	Filters points near a plane defined by coefficients.

	Keeps points within a distance threshold of a plane specified by its
	equation coefficients (ax + by + cz + d = 0).
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "can_vertical_3_downsampled.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute 
	filtered_point_cloud = vitreous.filter_point_cloud_using_plane_proximity(
		distance_threshold=4.0,
		point_cloud=point_cloud,
		plane_coefficients=[0.028344755192329624, -0.5747207168510667, -0.8178585895344518, 555.4890362620131]
	)
	logger.success("Filtered points using plane proximity")


	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("filter_point_cloud_using_plane_proximity", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([ 508.26353625, -457.76445726,  289.53896696])
	look_target = np.array([ 21.82833776,  -6.47603561, 684.12881138])
	eye_up = np.array([ 0.02846775, -0.57502007, -0.81764387])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud",
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Filtered Point Cloud",
				origin="filtered_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))


	# Visualize the plane as a mesh
	# Log the input point cloud under input_point_cloud + plane
	rr.log("input_point_cloud", rr.Points3D(point_cloud.positions,
										 colors=point_cloud.colors))
								

	# Log input: plane + point cloud, output: filtered point cloud
	rr.log("filtered_point_cloud", rr.Points3D(np.asarray(filtered_point_cloud.positions),
			   colors=filtered_point_cloud.colors))


def filter_point_cloud_using_plane_splitting_example():
	"""
	Splits a point cloud by a plane, keeping one side.

	Divides a point cloud using a plane and keeps points on either the positive
	or negative side.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "mounts_3_raw.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	filtered_point_cloud = vitreous.filter_point_cloud_using_plane_splitting(
		keep_positive_side=True,
		point_cloud=point_cloud,
		plane_coefficients=[0, 0, 1, -547],
	)
	logger.success("Filtered points using plane splitting")


	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("filter_point_cloud_using_plane_splitting", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	look_target = np.array([10.928258702124491, -9.280412379284869, 493.51069810734896])
	offset = np.array([500.43928350426836, -132.2169471347786, -480.0830664379545])
	camera_eye_position = look_target + offset
	eye_up = np.array([0.03905324461567187, 0.006429765891357218, -0.9992164441178752])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=camera_eye_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)


def filter_point_cloud_using_radius_outlier_removal_example():
	"""
	Removes points with too few neighbors within a radius.

	Removes points that have fewer than a specified number of neighbors within
	a given radius.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "engine_parts_1_downsampled.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	filtered_point_cloud = vitreous.filter_point_cloud_using_radius_outlier_removal(
		num_points=75, neighborhood_radius=25, point_cloud=point_cloud
	)
	logger.success("Filtered points using radius outlier removal")


	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("filter_point_cloud_using_radius_outlier_removal", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([718.42189587, 569.90251796, 396.93629317])
	look_target = np.array([ 11.51572837,  -5.52494204, 504.87928829])
	eye_up = np.array([ 0.0407315,  -0.01940416, -0.99898169])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud",
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Filtered Point Cloud",
				origin="output_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Log the input and output point clouds
	rr.log("input_point_cloud", rr.ViewCoordinates.RDB, static=True)
	rr.log("output_point_cloud", rr.ViewCoordinates.RDB, static=True)

	# Log the input point cloud under input_point_cloud
	rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
			   colors=point_cloud.colors))
	
	# Log the output point cloud under output_point_cloud
	rr.log("output_point_cloud", rr.Points3D(positions=filtered_point_cloud.positions,
			   colors=filtered_point_cloud.colors))


def filter_point_cloud_using_statistical_outlier_removal_example():
	"""
	Removes statistical outliers based on distance distribution.

	Removes points that are farther than a threshold from their neighbors,
	where the threshold is computed from mean distance and standard deviation.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "can_vertical_6_masked.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	filtered_point_cloud = vitreous.filter_point_cloud_using_statistical_outlier_removal(
		num_neighbors=90,
		standard_deviation_ratio=0.1,
		point_cloud=point_cloud,
	)
	logger.success(f"Filtered point cloud to {len(filtered_point_cloud.positions)} points using statistical outlier removal")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("filter_point_cloud_using_statistical_outlier_removal", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([ 191.10105334, -405.66455294,  458.89275463])
	look_target = np.array([ -9.34432069, -78.6523904,  597.00921687])
	eye_up = np.array([ 0.02866881, -0.56233476, -0.82641256])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud",
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Filtered Point Cloud",
				origin="output_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	rr.log("input_point_cloud", rr.ViewCoordinates.RDB, static=True)
	rr.log("output_point_cloud", rr.ViewCoordinates.RDB, static=True)

	# Log the input point cloud under input_point_cloud
	rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
			   colors=point_cloud.colors))

	# Log the output point cloud under output_point_cloud
	rr.log("output_point_cloud", rr.Points3D(positions=filtered_point_cloud.positions,
		   colors=filtered_point_cloud.colors))


def filter_point_cloud_using_uniform_downsampling_example():
	"""
	Downsamples a point cloud by selecting every Nth point.

	Uniformly samples points by selecting every step_size-th point from the
	original cloud.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "zivid_welding_scene.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	filtered_point_cloud = vitreous.filter_point_cloud_using_uniform_downsampling(
		step_size=20, point_cloud=point_cloud
	)
	logger.success("Filtered points using uniform downsampling")


	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("filter_point_cloud_using_uniform_downsampling", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([  38.81961243, -627.17132374,  604.40133262])
	look_target = np.array([ 37.2708837,   -6.99644708,  699.19013484])
	eye_up = np.array([0., -0.2, -0.97863545])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud",
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Filtered Point Cloud",
				origin="output_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Log the input and output point clouds
	rr.log("input_point_cloud", rr.ViewCoordinates.RDB, static=True)
	rr.log("output_point_cloud", rr.ViewCoordinates.RDB, static=True)

	# Log the input point cloud under input_point_cloud
	rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions,
			   colors=point_cloud.colors))

	# Log the output point cloud under output_point_cloud
	rr.log("output_point_cloud", rr.Points3D(positions=filtered_point_cloud.positions,
			   colors=filtered_point_cloud.colors))


def filter_point_cloud_using_viewpoint_visibility_example():
	"""
	Filters points based on visibility from a camera viewpoint.

	Removes points that are occluded or outside the visibility range from
	a specified camera position.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "zivid_parcels_04_preprocessed.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	filtered_point_cloud = vitreous.filter_point_cloud_using_viewpoint_visibility(
		viewpoint=[100, -500, 250.0],
		visibility_radius=100000.0,
		point_cloud=point_cloud,
	)
	logger.success("Filtered points using viewpoint visibility")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("filter_point_cloud_using_viewpoint_visibility", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([100, -500, 250.0])
	look_target = np.array([0., 0., 0.])
	eye_up = np.array([0., 0., 1.])

	# EyeControls:
	# - overview_eye_controls for View 1 & 3 (zoomed out)
	# - camera_eye_controls for View 2 (exact viewpoint)
	overview_eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
	)

	camera_eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.0,
		speed=0.0,
	)

	# Send blueprint
	rr.send_blueprint(
		rrb.Blueprint(
			rrb.Horizontal(
				rrb.Spatial3DView(
					name="Input (Centered) – Overview (zoomed out)",
					origin="input_point_cloud",
					background=background,
					eye_controls=camera_eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information,
				),
				rrb.Spatial3DView(
					name="Camera View – What the camera sees",
					origin="input_point_cloud",
					background=background,
					eye_controls=camera_eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information,
				),
				rrb.Spatial3DView(
					name="Filtered (Centered) – Overview (zoomed out)",
					origin="output_point_cloud",
					background=background,
					eye_controls=camera_eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information,
				),
			)
		)
	)

	# Center the point clouds for better visualization
	rr.log("input_point_cloud", rr.ViewCoordinates.RDB, static=True)
	rr.log("output_point_cloud", rr.ViewCoordinates.RDB, static=True)

	# Log centered input
	rr.log(
			"input_point_cloud",
			rr.Points3D(
				positions=point_cloud.positions,
				colors=point_cloud.colors,
			),
		)

	# Log centered filtered cloud
	rr.log(
			"output_point_cloud",
			rr.Points3D(
				positions=filtered_point_cloud.positions,
				colors=filtered_point_cloud.colors,
			),
		)

	# Show the camera location as a red dot
	rr.log(
		"camera_viewpoint",
		rr.Points3D(
			positions=overview_position,
			colors=np.array([[255, 0, 0]], dtype=np.uint8),
		),
	)


def filter_point_cloud_using_voxel_downsampling_example():
	"""
	Downsamples a point cloud using voxel grid averaging.

	Divides 3D space into voxels and replaces all points within each voxel
	with their centroid.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "can_vertical_1_subtracted.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	filtered_point_cloud = vitreous.filter_point_cloud_using_voxel_downsampling(
		voxel_size=0.005, point_cloud=point_cloud
	)
	logger.success("Filtered points using voxel downsampling")


	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("filter_point_cloud_using_voxel_downsampling", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([0.01798786, -0.1938991, 0.48053457])
	look_target = np.array([0.01602358, -0.131963, 0.66461711])
	eye_up = np.array([0., 0.8, -0.6])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5, 
		speed=0.0,  
		tracking_entity=None,  

	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud", 
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Filtered Point Cloud", 
				origin="output_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	rr.log("input_point_cloud", rr.ViewCoordinates.RDB, static=True)
	rr.log("output_point_cloud", rr.ViewCoordinates.RDB, static=True)
	
	# Log the input point cloud under input_point_cloud
	rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions, colors=point_cloud.colors))

	# Log the output point cloud under output_point_cloud
	rr.log("output_point_cloud", rr.Points3D(positions=filtered_point_cloud.positions, colors=filtered_point_cloud.colors))

# Operation examples
	
def add_point_clouds_example():
	"""
	Merges two point clouds into a single cloud.

	Combines all points from both clouds into one unified point cloud.
	"""
	# ===================== Operation ==========================================

	# Load point clouds
	filepath1 = str(DATA_DIR / "point_clouds" / "can_vertical_3_clustered.ply")
	filepath2 = str(DATA_DIR / "point_clouds" / "can_vertical_3_segmented_plane.ply")
	point_cloud1 = io.load_point_cloud(filepath=filepath1)
	point_cloud2 = io.load_point_cloud(filepath=filepath2)
	logger.success(f"Loaded point cloud 1 with {len(point_cloud1.positions)} points")
	logger.success(f"Loaded point cloud 2 with {len(point_cloud2.positions)} points")

	# Execute operation
	added_point_cloud = vitreous.add_point_clouds(
		point_cloud1=point_cloud1, point_cloud2=point_cloud2
	)
	logger.success(
		f"Added point clouds: {len(point_cloud1.positions)} + {len(point_cloud2.positions)} points"
	)

	# #=============================== Visualization (Optional) ===============================
	
	# Setup camera view
	overview_position = np.array([ 141.36766041,  230.084104,   -188.49101382])
	look_target = np.array([ 17.26467918, -10.168208,   676.98202765])
	eye_up = np.array([0., 0., -1.])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,  # Speed of camera rotation/spin
		speed=0.0,  # Translation speed of camera movement
		tracking_entity=None,  # Entity to track (None = no tracking)

	)

	# Initialize Rerun
	rr.init("add_point_clouds", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Vertical(
				rrb.Spatial3DView(
					name="Point Cloud 1",
					origin="point_cloud_1",
					background=background,
					eye_controls=eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information
				),
				rrb.Spatial3DView(
					name="Point Cloud 2",
					origin="point_cloud_2",
					background=background,
					eye_controls=eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information
				),
			),
			rrb.Spatial3DView(
				name="Added Point Cloud",
				origin="added_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information
			),)
	))

	# Log the first point cloud under point_cloud_1
	rr.log("point_cloud_1", rr.Points3D(positions=point_cloud1.positions,
			   colors=point_cloud1.colors))

	# Log the second point cloud under point_cloud_2
	rr.log("point_cloud_2", rr.Points3D(positions=point_cloud2.positions,
		   colors=point_cloud2.colors))

	# Log the added point cloud under added_point_cloud
	rr.log("added_point_cloud", rr.Points3D(positions=added_point_cloud.positions,
		   colors=added_point_cloud.colors))


def subtract_point_clouds_example():
	"""
	Removes points from one cloud that are near points in another cloud.

	Subtracts point_cloud2 from point_cloud1 by removing any point in cloud1
	that is within distance_threshold of any point in cloud2.
	"""
	# ===================== Operation ==========================================

	# Load point clouds
	filepath1 = str(DATA_DIR / "point_clouds" / "zivid_mixed_grocery_pallet_centered.ply")
	filepath2 = str(DATA_DIR / "point_clouds" / "zivid_mixed_grocery_pallet_box_filtered.ply")
	point_cloud1 = io.load_point_cloud(filepath=filepath1)
	point_cloud2 = io.load_point_cloud(filepath=filepath2)
	logger.success(f"Loaded point cloud 1 with {len(point_cloud1.positions)} points")
	logger.success(f"Loaded point cloud 2 with {len(point_cloud2.positions)} points")

	# Execute operation
	subtracted_point_cloud = vitreous.subtract_point_clouds(
		distance_threshold=0.1,
		point_cloud1=point_cloud1,
		point_cloud2=point_cloud2,
	)
	logger.success("Subtracted point clouds")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("subtract_point_clouds", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	look_target = np.array([100, 0, 0])
	offset = np.array([0.0, 150.0, 200.0])
	camera_eye_position = look_target + offset
	eye_up = np.array([0.0, 0.0, 1.0])
	look_target = [100, 0, 0]
	zoom_out_factor = 10

	vec = camera_eye_position - look_target
	dir_vec = vec / np.linalg.norm(vec)
	overview_position = look_target + dir_vec * (np.linalg.norm(vec) * zoom_out_factor)

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,

	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Vertical(
				rrb.Spatial3DView(
					name="Point Cloud 1",
					origin="point_cloud_1",
					background=background,
					eye_controls=eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information
				),
				rrb.Spatial3DView(
					name="Point Cloud 2",
					origin="point_cloud_2",
					background=background,
					eye_controls=eye_controls,
					line_grid=line_grid,
					spatial_information=spatial_information
				),
			),
			rrb.Spatial3DView(
				name="Subtracted Point Cloud",
				origin="subtracted_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information
			),)
	))

	# Log the first point cloud under point_cloud_1
	rr.log("point_cloud_1", rr.Points3D(positions=point_cloud1.positions,
			   colors=point_cloud1.colors))

	# Log the second point cloud under point_cloud_2
	rr.log("point_cloud_2", rr.Points3D(positions=point_cloud2.positions,
		   colors=point_cloud2.colors))

	# Log the subtracted point cloud under subtracted_point_cloud
	rr.log("subtracted_point_cloud", rr.Points3D(positions=subtracted_point_cloud.positions,
		   colors=subtracted_point_cloud.colors))


def scale_point_cloud_example():
	"""
	Scales a point cloud uniformly about a center point.

	Multiplies all point coordinates by a scale factor relative to a center.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "relay_2_raw.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	scaled_point_cloud = vitreous.scale_point_cloud(
		point_cloud=point_cloud,
		center_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
		scale_factor=0.3,
		modify_inplace=False)
	logger.success(f"Scaled point cloud to {len(scaled_point_cloud.positions)} points")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("scale_point_cloud", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([117.44420607, -90.56381865, 110.3344537])
	look_target = np.array([18.521805, 4.61124328, 282.22516171])
	eye_up = np.array([-0.07754533, 0.34059355, -0.93700734])

	eye_controls_original = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	look_target = np.array([5.556541501027933, 1.383372985065839, 84.66754851275184])
	offset = np.array([80.57584366, -106.54735255, -171.92510329])
	camera_eye_position = look_target + offset
	eye_up = np.array([-0.0774216, 0.34095774, -0.93688511])

	eye_controls_scaled= rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=camera_eye_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud", 
				origin="original_point_cloud",
				background=background,
				eye_controls=eye_controls_original,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Scaled Point Cloud", 
				origin="scaled_point_cloud",
				background=background,
				eye_controls=eye_controls_scaled,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Visualize original point cloud
	rr.log("original_point_cloud", rr.Points3D(positions=point_cloud.positions,
		   colors=point_cloud.colors))

	# Visualize scaled point cloud
	if scaled_point_cloud is not None:
		rr.log("scaled_point_cloud", rr.Points3D(positions=scaled_point_cloud.positions,
			   colors=scaled_point_cloud.colors))
	else:
		logger.error("Scaling failed: No scaled point cloud to log.")


def apply_transform_to_point_cloud_example():
	"""
	Applies a 6-DOF rigid transformation (rotation + translation) to a point cloud.

	Transforms points using a 4x4 homogeneous transformation matrix.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "plastic_centered.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	transformed_point_cloud = vitreous.apply_transform_to_point_cloud(
		point_cloud=point_cloud, 
		transformation_matrix= [[1, 0, 0, 15], [0, 1, 0, 15], [0, 0, 1, 5], [0, 0, 0, 1]],
		modify_inplace=False
	)
	logger.success(f"Applied transform to {len(transformed_point_cloud.positions)} points")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("apply_transform_to_point_cloud", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([ 51.87143098,   2.90578544, -47.97485367])
	look_target = np.array([0, 0, 0])
	eye_up = np.array([ 0.03973926,  0.00298701, -0.99920562])

	eye_controls_original = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Source Point Cloud", 
				origin="source_point_cloud",
				background=background,
				eye_controls=eye_controls_original,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Transformed Point Cloud", 
				origin="transformed_point_cloud",
				background=background,
				eye_controls=eye_controls_original,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))
	
	# Visualize original point cloud
	rr.log("source_point_cloud", rr.Points3D(
		positions=point_cloud.positions,
		colors=point_cloud.colors)
	   )

	# Draw origin frame for source point cloud
	axis_length = 10  # Adjust based on point cloud scale
	rr.log("source_point_cloud/origin_frame", rr.Arrows3D(
		origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
		vectors=[[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
		colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for X, Y, Z
	))

	# Visualize transformed point cloud
	if transformed_point_cloud is not None:
		rr.log("transformed_point_cloud", rr.Points3D(
			positions=transformed_point_cloud.positions,
			colors=transformed_point_cloud.colors)
		   )

		# Draw origin frame for transformed point cloud
		rr.log("transformed_point_cloud/origin_frame", rr.Arrows3D(
			origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
			vectors=[[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
			colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for X, Y, Z
		))
	else:
		logger.error("Transformation failed: No transformed point cloud to log.")

# Projection examples

def project_point_cloud_to_plane_example():
	"""
	Projects all points orthogonally onto a plane.

	Moves each point to its closest point on the specified plane. Flattens
	the cloud onto a 2D surface in 3D space.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "engine_parts_0.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	projected_point_cloud = vitreous.project_point_cloud_to_plane(
		add_white_noise=False,
		white_noise_standard_deviation=1e-6,
		point_cloud=point_cloud,
		plane_coefficients=[0.0, 0.0, 1.0, 0.0],
	)
	logger.success(f"Projected {len(projected_point_cloud.positions)} points to plane")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("project_point_cloud_to_plane", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup camera view
	overview_position = np.array([502.75708451, -134.34381185, -519.97747961])
	look_target = np.array([0, 0, 0])
	eye_up = np.array([0.04082638, -0.00847461, -0.99913032])

	# Add EyeControls3D with all parameters for camera movement tuning
	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,  # Camera control type: Orbital or FirstPerson
		position=overview_position,  # Initial camera position (None = auto)
		look_target=look_target,  # Point the camera looks at (None = auto)
		eye_up=eye_up,  # Up direction vector (None = auto)
		spin_speed=0.5,  # Speed of camera rotation/spin
		speed=0.0,  # Translation speed of camera movement
		tracking_entity=None,  # Entity to track (None = no tracking)
	)
	
	line_grid = rrb.LineGrid3D(
		visible=False,  # The grid is enabled by default, but you can hide it with this property.
	)
	
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud", 
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Projected Point Cloud", 
				origin="projected_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))
	# Visualize original point cloud
	rr.log("input_point_cloud/points", rr.Points3D(
		positions=point_cloud.positions,
		colors=point_cloud.colors,
		radii=point_cloud.radii,
	))


	# Visualize the projection plane
	plane_coeffs = np.array([0, 0, 1, 0])
	a, b, c, d = plane_coeffs
	normal = np.array([a, b, c])
	normal = normal / np.linalg.norm(normal)  # Normalize
	point_on_plane = -d / (a**2 + b**2 + c**2) * np.array([a, b, c])

	# Calculate the extent based on the point cloud bounding box
	points_np = np.asarray(point_cloud.positions)
	extent = np.linalg.norm(np.ptp(points_np, axis=0))
	rect_size = 0.3 * extent if extent > 0 else 0.2

	# Create plane rectangle
	helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])

	v = np.cross(normal, helper)
	v /= np.linalg.norm(v)

	u = np.cross(normal, v)
	u /= np.linalg.norm(u)

	half = rect_size / 2.0
	rect_corners = np.stack(
		[
			point_on_plane + half * u + half * v,
			point_on_plane - half * u + half * v,
			point_on_plane - half * u - half * v,
			point_on_plane + half * u - half * v,
		],
		axis=0,
	)

	rect_lines = np.stack([
		np.stack([rect_corners[0], rect_corners[1]]),
		np.stack([rect_corners[1], rect_corners[2]]),
		np.stack([rect_corners[2], rect_corners[3]]),
		np.stack([rect_corners[3], rect_corners[0]]),
	], axis=0)

	rr.log("input_point_cloud/plane",
		   rr.LineStrips3D(rect_lines,
						   radii=rr.Radius.ui_points(3.0),
						   colors=np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (4, 1))))

	rr.log("input_point_cloud/plane_normal",
		   rr.Arrows3D(origins=np.array([point_on_plane]),
					   vectors=np.array([normal * extent * 0.2]),
					   colors=np.array([[255, 0, 0]]),
					   radii=2.0))

	# Visualize projected point cloud
	if projected_point_cloud is not None:
		rr.log("projected_point_cloud/points", rr.Points3D(
			positions=projected_point_cloud.positions,
			colors=projected_point_cloud.colors,
			radii=projected_point_cloud.radii,
		))


def project_point_cloud_to_plane_defined_by_point_normal_example():
	"""
	Projects points onto a plane defined by a point and normal (alternative parameterization).

	Same as plane projection but using point+normal instead of coefficients.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "engine_parts_0.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	plane_point = np.array([0.0, 0.0, 0.0])
	plane_normal = np.array([0.0, 0.0, 1.0])

	projected_point_cloud = vitreous.project_point_cloud_to_plane_defined_by_point_normal(
		add_white_noise=False,
		white_noise_standard_deviation=1e-6,
		point_cloud=point_cloud,
		point=plane_point,
		plane_normal=plane_normal,
	)
	logger.success(
		f"Projected {len(projected_point_cloud.positions)} points to plane (point+normal)"
	)

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("project_point_cloud_to_plane_defined_by_point_normal", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([502.75708451, -134.34381185, -519.97747961])
	look_target = np.array([0, 0, 0])
	eye_up = np.array([0.04082638, -0.00847461, -0.99913032])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud", 
				origin="input_point_cloud_and_plane",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Projected Point Cloud", 
				origin="projected_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))
	# Log input point cloud 
	rr.log("input_point_cloud_and_plane/points", rr.Points3D(
		positions=point_cloud.positions,
		colors=point_cloud.colors,
		radii=point_cloud.radii,
	))

	# Log plane
	points_np = np.asarray(point_cloud.positions)
	extent = np.linalg.norm(np.ptp(points_np, axis=0))
	rect_size = 0.3 * extent if extent > 0 else 0.2


	# Create plane rectangle
	helper = np.array([0.0, 0.0, 1.0]) if abs(plane_normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])

	v = np.cross(plane_normal, helper)
	v /= np.linalg.norm(v)

	u = np.cross(plane_normal, v)
	u /= np.linalg.norm(u)

	half = rect_size / 2.0
	rect_corners = np.stack(
		[
			plane_point + half * u + half * v,
			plane_point - half * u + half * v,
			plane_point - half * u - half * v,
			plane_point + half * u - half * v,
		],
		axis=0,
	)
	rect_lines = np.stack(
		[
			np.stack([rect_corners[0], rect_corners[1]]),
			np.stack([rect_corners[1], rect_corners[2]]),
			np.stack([rect_corners[2], rect_corners[3]]),
			np.stack([rect_corners[3], rect_corners[0]]),
		],
		axis=0,
	)

	rr.log(
		"input_point_cloud_and_plane/plane",
		rr.LineStrips3D(
			rect_lines,
			radii=rr.Radius.ui_points(3.0),  # constant screen-space thickness
			colors=np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (4, 1)),
		),
	)

	# Log plane normal vector
	normal_normalized = plane_normal / np.linalg.norm(plane_normal)
	rr.log("input_point_cloud_and_plane/plane_normal",
		   rr.Arrows3D(origins=np.array([plane_point]),
					   vectors=np.array([normal_normalized * extent * 0.2]),
					   colors=np.array([[255, 0, 0]]),
					   radii=2.0))

	# Log projected point cloud
	rr.log("projected_point_cloud/points", rr.Points3D(
		positions=projected_point_cloud.positions,
		colors=projected_point_cloud.colors,
		radii=projected_point_cloud.radii,
	))

# Reconstruction examples

def reconstruct_mesh_using_convex_hull_example():
	"""
	Computes the convex hull mesh enclosing a point cloud.

	Creates the smallest convex shape that contains all points.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "beer_can_corrupted_normals.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	result_mesh = vitreous.reconstruct_mesh_using_convex_hull(
		joggle_inputs=False, 
		point_cloud=point_cloud
	)
	logger.success(
		f"Reconstructed convex hull mesh from {len(point_cloud.positions)} points"
	)

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("reconstruct_mesh_using_convex_hull", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	look_target = np.array([-0.0007480896814801301, -0.00017694868037873947, 0.06533939420168008])
	offset = np.array([0.1431982364518114, -0.007435637521927663, 0.15607460400219597])
	camera_eye_position = look_target + offset
	eye_up = np.array([0.5181035277056806, -0.8545437583115796, -0.036382684200725636])
	zoom_out_factor = 2

	vec = camera_eye_position - look_target
	dir_vec = vec / np.linalg.norm(vec)
	overview_position = look_target + dir_vec * (np.linalg.norm(vec) * zoom_out_factor)

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud",
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Convex Hull Reconstructed Mesh",
				origin="convex_hull_mesh",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Log the input point cloud
	rr.log("input_point_cloud", rr.Points3D(
		positions=point_cloud.positions,
		colors=point_cloud.colors if point_cloud.colors is not None else None,
	))

	# Log the output mesh
	rr.log("convex_hull_mesh", rr.Mesh3D(
		vertex_positions=result_mesh.vertex_positions,
		triangle_indices=result_mesh.triangle_indices,
		vertex_normals=result_mesh.vertex_normals if hasattr(result_mesh, 'vertex_normals') else None,
		albedo_factor=[0.8, 0.8, 0.8, 1.0],
	))


def reconstruct_mesh_using_poisson_example():
	"""
	Reconstructs a watertight mesh from an oriented point cloud using Poisson surface reconstruction.

	Solves a Poisson equation to fit a smooth surface through points with normals.
	Produces closed, manifold meshes. Requires point cloud normals.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "industrial_part_7_normals.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	reconstructed_mesh = vitreous.reconstruct_mesh_using_poisson(
		octree_depth=7, octree_width=0, scale_factor=1.1,
		point_cloud=point_cloud,
	)
	logger.success(
		f"Reconstructed mesh from {len(point_cloud.positions)} points using Poisson"
	)

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("reconstruct_mesh_using_poisson", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	look_target = np.array([-0.18742625171499017, 0.00011380453432235013, 8.55789070127426e-05])
	offset = np.array([0.3001245906351528, -0.2984173793020396, -0.07186921049564646])
	camera_eye_position = look_target + offset
	eye_up = np.array([0.0006837715295567754, -0.9723082452673518, -0.2337011096285566])
	zoom_out_factor = 2

	vec = camera_eye_position - look_target
	dir_vec = vec / np.linalg.norm(vec)
	overview_position = look_target + dir_vec * (np.linalg.norm(vec) * zoom_out_factor)

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud",
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Poisson Reconstructed Mesh",
				origin="poisson_mesh",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Log the input point cloud
	rr.log("input_point_cloud", rr.Points3D(
		positions=point_cloud.positions,
		colors=point_cloud.colors if point_cloud.colors is not None else None,
	))

	# Log the output mesh
	rr.log("poisson_mesh", rr.Mesh3D(
		vertex_positions=reconstructed_mesh.vertex_positions,
		triangle_indices=reconstructed_mesh.triangle_indices,
		vertex_normals=reconstructed_mesh.vertex_normals if hasattr(reconstructed_mesh, 'vertex_normals') else None,
		albedo_factor=[0.8, 0.8, 0.8, 1.0],
	))

# Registration examples

def register_point_clouds_using_centroid_translation_example():
	"""
	Aligns point clouds by matching their centroids (coarse alignment).

	Computes a translation that moves the source cloud's center to the target cloud's
	center. Fast initial alignment step before fine registration.
	"""
	# ===================== Operation ==========================================

	# Load point clouds
	filepath1 = str(DATA_DIR / "point_clouds" / "zivid_manufacturing_workpieces.ply")
	filepath2 = str(DATA_DIR / "point_clouds" / "zivid_manufacturing_workpieces_centered.ply")
	source_point_cloud = io.load_point_cloud(filepath=filepath1)
	target_point_cloud = io.load_point_cloud(filepath=filepath2)
	logger.success(f"Loaded source point cloud with {len(source_point_cloud.positions)} points")
	logger.success(f"Loaded target point cloud with {len(target_point_cloud.positions)} points")

	# Execute operation
	transformation_matrix = vitreous.register_point_clouds_using_centroid_translation(
		source_point_cloud=source_point_cloud,
		target_point_cloud=target_point_cloud,
		initial_transformation_matrix=np.eye(4),
	)
	logger.success(f"Registered point clouds using centroid translation, transformation_matrix: {transformation_matrix.matrix}")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("register_point_clouds_using_centroid_translation", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(name="Source Point Cloud", origin="source_point_cloud"),
			rrb.Spatial3DView(name="Target Point Cloud", origin="target_point_cloud")
		))
	)
	# Log input point clouds
	rr.log(
		"source_point_cloud",
		rr.Points3D(
			positions=source_point_cloud.positions,
			colors=source_point_cloud.colors
		),
	)
	rr.log(
		"target_point_cloud",
		rr.Points3D(
			positions=target_point_cloud.positions,
			colors=target_point_cloud.colors
		),
	)


def register_point_clouds_using_cuboid_translation_sampler_icp_example():
	"""
	Finds best alignment by sampling translations in a 3D grid (cuboid) with ICP.

	Tries translations on a regular 3D grid within specified x/y/z ranges, runs ICP
	for each, and keeps best result.
	"""
	# ===================== Operation ==========================================

	# Load point clouds
	source_filepath = str(DATA_DIR / "point_clouds" / "weld_clamp_model_shifted.ply")
	target_filepath = str(DATA_DIR / "point_clouds" / "weld_clamp_cluster_0_centroid_registered.ply")
	source_point_cloud = io.load_point_cloud(filepath=source_filepath)
	target_point_cloud = io.load_point_cloud(filepath=target_filepath)
	logger.success(f"Loaded source point cloud with {len(source_point_cloud.positions)} points")
	logger.success(f"Loaded target point cloud with {len(target_point_cloud.positions)} points")

	# Execute operation
	transformation_matrix = (
		vitreous.register_point_clouds_using_cuboid_translation_sampler_icp(
			step_size=2,
			x_min=-20,
			x_max=20,
			y_min=-20,    
			y_max=20,
			z_min=-20,
			z_max=20,
			early_stop_fitness_score=0.3,
			min_fitness_score=0.48,
			max_iterations=50,
			max_correspondence_distance=2,
			estimate_scaling=False,
			source_point_cloud=source_point_cloud,
			target_point_cloud=target_point_cloud,
			initial_transformation_matrix=np.eye(4),
		)
	)
	logger.success(f"Registered point clouds using cuboid translation sampler ICP: transformation_matrix: {transformation_matrix.matrix}")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("register_point_clouds_using_cuboid_translation_sampler_icp", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([182.91228535, 149.88950409, 796.55904478])
	look_target = np.array([-60.82915742, -93.85193868, 552.81760201])
	eye_up = np.array([0., 0., 1.])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Before Registration",
				origin="before_registration",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="After Registration",
				origin="after_registration",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Create aligned source point cloud
	aligned_source = vitreous.apply_transform_to_point_cloud(
		point_cloud=source_point_cloud,
		transformation_matrix=transformation_matrix.matrix,
		modify_inplace=False,
	)

	# Before: Show source (red) and target (green) misaligned
	rr.log("before_registration/source", rr.Points3D(
		positions=source_point_cloud.positions,
		colors=[[255, 0, 0]] * len(source_point_cloud.positions),
	))
	rr.log("before_registration/target", rr.Points3D(
		positions=target_point_cloud.positions,
		colors=[[0, 255, 0]] * len(target_point_cloud.positions),
	))

	# After: Show aligned source (red) and target (green) overlapping
	rr.log("after_registration/source_aligned", rr.Points3D(
		positions=aligned_source.positions,
		colors=[[255, 0, 0]] * len(aligned_source.positions),
	))
	rr.log("after_registration/target", rr.Points3D(
		positions=target_point_cloud.positions,
		colors=[[0, 255, 0]] * len(target_point_cloud.positions),
	))


def register_point_clouds_using_fast_global_registration_example():
	"""
	Aligns point clouds using Fast Global Registration (FGR).

	Feature-based registration that's faster than RANSAC. Uses graduated
	non-convexity optimization.
	"""
	# ===================== Operation ==========================================

	# Load point clouds
	source_filepath = str(DATA_DIR / "point_clouds" / "gusset_model_voxelized.ply")
	target_filepath = str(DATA_DIR / "point_clouds" / "gusset_0_preprocessed_voxelized.ply")
	source_point_cloud = io.load_point_cloud(filepath=source_filepath)
	target_point_cloud = io.load_point_cloud(filepath=target_filepath)
	logger.success(f"Loaded source point cloud with {len(source_point_cloud.positions)} points")
	logger.success(f"Loaded target point cloud with {len(target_point_cloud.positions)} points")

	# Execute operation
	transformation_matrix = vitreous.register_point_clouds_using_fast_global_registration(
		normal_radius=0.02,
		normal_max_neighbors=20,
		feature_radius=0.05,
		feature_max_neighbors=30,
		max_correspondence_distance=0.015,
		source_point_cloud=source_point_cloud,
		target_point_cloud=target_point_cloud,
		initial_transformation_matrix=np.eye(4),
	)
	logger.success(f"Registered point clouds using fast global registration, transformation_matrix: {transformation_matrix.matrix}")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("register_point_clouds_using_fast_global_registration", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	look_target = np.array([0.00032920947519606545, -0.000529476746455524, 0.06306978755275303])
	offset = np.array([0.1538893433972759, 0.1538893433972759, 0.1538893433972759])
	camera_eye_position = look_target + offset
	eye_up = np.array([0.0, 1.0, 0.0])
	zoom_out_factor = 2

	vec = camera_eye_position - look_target
	dir_vec = vec / np.linalg.norm(vec)
	overview_position = look_target + dir_vec * (np.linalg.norm(vec) * zoom_out_factor)

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Before Registration",
				origin="before_registration",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="After Registration",
				origin="after_registration",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Create aligned source point cloud
	aligned_source = vitreous.apply_transform_to_point_cloud(
		point_cloud=source_point_cloud,
		transformation_matrix=transformation_matrix.matrix,
		modify_inplace=False,
	)

	# Before: Show source (red) and target (green) misaligned
	rr.log("before_registration/source", rr.Points3D(
		positions=source_point_cloud.positions,
		colors=[[255, 0, 0]] * len(source_point_cloud.positions),
	))
	rr.log("before_registration/target", rr.Points3D(
		positions=target_point_cloud.positions,
		colors=[[0, 255, 0]] * len(target_point_cloud.positions),
	))

	# After: Show aligned source (red) and target (green) overlapping
	rr.log("after_registration/source_aligned", rr.Points3D(
		positions=aligned_source.positions,
		colors=[[255, 0, 0]] * len(aligned_source.positions),
	))
	rr.log("after_registration/target", rr.Points3D(
		positions=target_point_cloud.positions,
		colors=[[0, 255, 0]] * len(target_point_cloud.positions),
	))


def register_point_clouds_using_point_to_plane_icp_example():
	"""
	Aligns point clouds using Point-to-Plane ICP.

	Minimizes point-to-tangent-plane distances instead of point-to-point. More
	accurate than point-to-point ICP, especially for planar surfaces.
	"""
	# ===================== Operation ==========================================

	# Load point clouds
	source_filepath = str(DATA_DIR / "point_clouds" / "weld_clamp_cluster_0_centroid_registered.ply")
	target_filepath = str(DATA_DIR / "point_clouds" / "weld_clamp_model_centered.ply")
	source_point_cloud = io.load_point_cloud(filepath=source_filepath)
	target_point_cloud = io.load_point_cloud(filepath=target_filepath)
	logger.success(f"Loaded source point cloud with {len(source_point_cloud.positions)} points")
	logger.success(f"Loaded target point cloud with {len(target_point_cloud.positions)} points")

	# Execute operation
	transformation_matrix = vitreous.register_point_clouds_using_point_to_plane_icp(
		max_iterations=50,
		max_correspondence_distance=0.05,
		normal_max_neighbors=30,
		normal_search_radius=0.001,
		use_robust_kernel=False,
		loss_type="tukey_loss",
		noise_standard_deviation=0.001,
		source_point_cloud=source_point_cloud,
		target_point_cloud=target_point_cloud,
		initial_transformation_matrix=np.eye(4),
	)
	logger.success("Registered point clouds using point-to-plane ICP")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("register_point_clouds_using_point_to_plane_icp", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	look_target = np.array([-59.93859144312686, -93.42425677943251, 552.6664551913788])
	offset = np.array([243.71321614552414, 243.71321614552414, 243.71321614552414])
	camera_eye_position = look_target + offset
	eye_up = np.array([0.0, 1.0, 0.0])
	zoom_out_factor = 2

	vec = camera_eye_position - look_target
	dir_vec = vec / np.linalg.norm(vec)
	overview_position = look_target + dir_vec * (np.linalg.norm(vec) * zoom_out_factor)

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Before Registration",
				origin="before_registration",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="After Registration",
				origin="after_registration",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Create aligned source point cloud
	aligned_source = vitreous.apply_transform_to_point_cloud(
		point_cloud=source_point_cloud,
		transformation_matrix=transformation_matrix.matrix,
		modify_inplace=False,
	)

	# Before: Show source (red) and target (green) misaligned
	rr.log("before_registration/source", rr.Points3D(
		positions=source_point_cloud.positions,
		colors=[[255, 0, 0]] * len(source_point_cloud.positions),
	))
	rr.log("before_registration/target", rr.Points3D(
		positions=target_point_cloud.positions,
		colors=[[0, 255, 0]] * len(target_point_cloud.positions),
	))

	# After: Show aligned source (red) and target (green) overlapping
	rr.log("after_registration/source_aligned", rr.Points3D(
		positions=aligned_source.positions,
		colors=[[255, 0, 0]] * len(aligned_source.positions),
	))
	rr.log("after_registration/target", rr.Points3D(
		positions=target_point_cloud.positions,
		colors=[[0, 255, 0]] * len(target_point_cloud.positions),
	))


def register_point_clouds_using_point_to_point_icp_example():
	"""
	Aligns point clouds using Point-to-Point Iterative Closest Point (ICP).

	Iteratively refines alignment by minimizing point-to-point distances.
	Requires good initial alignment.
	"""
	# ===================== Operation ==========================================

	# Load point clouds
	source_filepath = str(DATA_DIR / "point_clouds" / "gusset_0_preprocessed.ply")
	target_filepath = str(DATA_DIR / "point_clouds" / "gusset_0_icp_alignment.ply")
	source_point_cloud = io.load_point_cloud(filepath=source_filepath)
	target_point_cloud = io.load_point_cloud(filepath=target_filepath)
	logger.success(f"Loaded source point cloud with {len(source_point_cloud.positions)} points")
	logger.success(f"Loaded target point cloud with {len(target_point_cloud.positions)} points")

	# Execute operation
	transformation_matrix = vitreous.register_point_clouds_using_point_to_point_icp(
		max_iterations=500,
		max_correspondence_distance=10,
		estimate_scaling=False,
		source_point_cloud=source_point_cloud,
		target_point_cloud=target_point_cloud,
		initial_transformation_matrix=np.eye(4),
	)
	logger.success("Registered point clouds using point-to-point ICP")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("register_point_clouds_using_point_to_point_icp", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	overview_position = np.array([ 141.36766041,  230.084104,   -188.49101382])
	look_target = np.array([ 17.26467918, -10.168208,   676.98202765])
	eye_up = np.array([ 0.,  0., -1.])

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Before Registration",
				origin="before_registration",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="After Registration",
				origin="after_registration",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Create aligned source point cloud
	aligned_source = vitreous.apply_transform_to_point_cloud(
		point_cloud=source_point_cloud,
		transformation_matrix=transformation_matrix.matrix,
		modify_inplace=False,
	)

	# Before: Show source (red) and target (green) misaligned
	rr.log("before_registration/source", rr.Points3D(
		positions=source_point_cloud.positions,
		colors=[[255, 0, 0]] * len(source_point_cloud.positions),
	))
	rr.log("before_registration/target", rr.Points3D(
		positions=target_point_cloud.positions,
		colors=[[0, 255, 0]] * len(target_point_cloud.positions),
	))

	# After: Show aligned result (red) and target (green) overlapping
	rr.log("after_registration/source_aligned", rr.Points3D(
		positions=aligned_source.positions,
		colors=[[255, 0, 0]] * len(aligned_source.positions),
	))
	rr.log("after_registration/target", rr.Points3D(
		positions=target_point_cloud.positions,
		colors=[[0, 255, 0]] * len(target_point_cloud.positions),
	))


def register_point_clouds_using_rotation_sampler_icp_example():
	"""
	Finds best alignment by trying multiple rotations with ICP refinement.

	Samples rotations in Euler angle space, runs ICP for each, and keeps the best.
	"""
	# ===================== Operation ==========================================

	# Load point clouds
	source_filepath = str(DATA_DIR / "point_clouds" / "zivid_bottle_segmented.ply")
	target_filepath = str(DATA_DIR / "point_clouds" / "zivid_bottle_cylinder_centered.ply")
	source_point_cloud = io.load_point_cloud(filepath=source_filepath)
	target_point_cloud = io.load_point_cloud(filepath=target_filepath)
	logger.success(f"Loaded source point cloud with {len(source_point_cloud.positions)} points")
	logger.success(f"Loaded target point cloud with {len(target_point_cloud.positions)} points")

	# Execute operation
	transformation_matrix = vitreous.register_point_clouds_using_rotation_sampler_icp(
		x_step_size_deg=20,
		y_step_size_deg=20,
		z_step_size_deg=20,
		x_min_deg=0,
		x_max_deg=180,
		y_min_deg=0,
		y_max_deg=180,
		z_min_deg=0,
		z_max_deg=180,
		early_stop_fitness_score=0.5,
		min_fitness_score=0.9,
		max_iterations=50,
		max_correspondence_distance=0.02,
		estimate_scaling=False,
		source_point_cloud=source_point_cloud,
		target_point_cloud=target_point_cloud,
		initial_transformation_matrix=np.eye(4),
	)
	logger.success("Registered point clouds using rotation sampler ICP")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("register_point_clouds_using_rotation_sampler_icp", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	look_target = np.array([-169.68149308056488, -57.38953201221218, 37.960415331431584])
	offset = np.array([263.0040045164911, 263.0040045164911, 263.0040045164911])
	camera_eye_position = look_target + offset
	eye_up = np.array([0.0, 0.0, 1.0])
	zoom_out_factor = 2

	vec = camera_eye_position - look_target
	dir_vec = vec / np.linalg.norm(vec)
	overview_position = look_target + dir_vec * (np.linalg.norm(vec) * zoom_out_factor)

	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=overview_position,
		look_target=look_target,
		eye_up=eye_up,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Before Registration",
				origin="before_registration",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="After Registration",
				origin="after_registration",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))

	# Create aligned source point cloud
	aligned_source = vitreous.apply_transform_to_point_cloud(
		point_cloud=source_point_cloud,
		transformation_matrix=transformation_matrix.matrix,
		modify_inplace=False,
	)

	# Before: Show source (red) and target (green) misaligned
	rr.log("before_registration/source", rr.Points3D(
		positions=source_point_cloud.positions,
		colors=[[255, 0, 0]] * len(source_point_cloud.positions),
	))
	rr.log("before_registration/target", rr.Points3D(
		positions=target_point_cloud.positions,
		colors=[[0, 255, 0]] * len(target_point_cloud.positions),
	))

	# After: Show aligned result (red) and target (green) overlapping
	rr.log("after_registration/source_aligned", rr.Points3D(
		positions=aligned_source.positions,
		colors=[[255, 0, 0]] * len(aligned_source.positions),
	))
	rr.log("after_registration/target", rr.Points3D(
		positions=target_point_cloud.positions,
		colors=[[0, 255, 0]] * len(target_point_cloud.positions),
	))

# Segmentation examples

def segment_point_cloud_using_color_example():
	"""
	Segments points by color similarity to a target color.

	Keeps points whose RGB color is within a distance threshold (Euclidean in
	RGB space) of a target color.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "engine_parts_0.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	segmented_point_cloud = vitreous.segment_point_cloud_using_color(
		target_color=[50, 75, 200],
		color_distance_threshold=60.0,
		point_cloud=point_cloud,
	)
	logger.success(f"Segmented {len(segmented_point_cloud.positions)} points using color")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("segment_point_cloud_using_color", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup camera view
	look_target = np.array([-27.917760217865144, 8.154586928673055, 529.1368178181901])
	offset = np.array([490.3580603260833, -175.19232461052903, -520.3408869973264])
	position = look_target + offset
	eye_up = np.array([0.04159355788852604, -0.009328899120838747, -0.9990910607063638])


	# Add EyeControls3D with all parameters for camera movement tuning
	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,  # Camera control type: Orbital or FirstPerson
		position=position,  # Initial camera position (None = auto)
		look_target=look_target,  # Point the camera looks at (None = auto)
		eye_up=eye_up,  # Up direction vector (None = auto)
		spin_speed=0.5,  # Speed of camera rotation/spin
		speed=0.0,  # Translation speed of camera movement
		tracking_entity=None,  # Entity to track (None = no tracking)
	)
	
	line_grid = rrb.LineGrid3D(
		visible=False,  # The grid is enabled by default, but you can hide it with this property.
	)
	
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud", 
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Segmented Point Cloud", 
				origin="segmented_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))
	# Visualize input point cloud
	rr.log("input_point_cloud", rr.Points3D(
		positions=point_cloud.positions,
		colors=point_cloud.colors
	))

	# Visualize segmented point cloud
	rr.log("segmented_point_cloud", rr.Points3D(
		positions=segmented_point_cloud.positions,
		colors=segmented_point_cloud.colors
	))


def segment_point_cloud_using_plane_example():
	"""
	Segments the dominant plane from a point cloud using RANSAC.

	Finds the largest planar surface in the cloud using random sample consensus.
	Returns inlier points and plane equation.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "can_vertical_3_downsampled.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	segmented_point_cloud = vitreous.segment_point_cloud_using_plane(
		distance_threshold=1.0,
		num_initial_points=3,
		max_iterations=1000,
		keep_outliers=False,
		point_cloud=point_cloud,
	)
	logger.success(f"Segmented {len(segmented_point_cloud.positions)} points using plane")

	# ===================== Visualization  (Optional) ======================

	# Initialize Rerun
	rr.init("segment_point_cloud_using_plane", spawn=False)
	try:
		rr.connect()
	except Exception:
		rr.spawn()

	# Setup additional rerun settings
	line_grid = rrb.LineGrid3D(visible=False)
	spatial_information = rrb.SpatialInformation(
		show_axes=False,
		show_bounding_box=False,
	)
	background = rrb.Background(color=(255, 255, 255))

	# Setup camera view
	look_target = np.array([17.017827726026972, -12.119958718903193, 679.1044351589703])
	offset = np.array([482.9731371585466, -460.7716744655184, -394.649557512302])
	position = look_target + offset
	plane_normal = np.array([0.02831009584389102, -0.5750797310951596, -0.817607388271919])
	eye_controls = rrb.EyeControls3D(
		kind=rrb.Eye3DKind.Orbital,
		position=position,
		look_target=look_target,
		eye_up=plane_normal,
		spin_speed=0.5,
		speed=0.0,
		tracking_entity=None,
	)

	# Send blueprint
	rr.send_blueprint(rrb.Blueprint(
		rrb.Horizontal(
			rrb.Spatial3DView(
				name="Input Point Cloud", 
				origin="input_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
			rrb.Spatial3DView(
				name="Segmented Point Cloud", 
				origin="segmented_point_cloud",
				background=background,
				eye_controls=eye_controls,
				line_grid=line_grid,
				spatial_information=spatial_information,
			),
		)
	))
	# Visualize input point cloud
	rr.log("input_point_cloud", rr.Points3D(positions=point_cloud.positions, colors=point_cloud.colors))
	rr.log("segmented_point_cloud", rr.Points3D(positions=segmented_point_cloud.positions, colors=segmented_point_cloud.colors))


def segment_point_cloud_using_vector_proximity_example():
	"""
	Segments points near a line defined by a point and direction vector.

	Keeps points within a distance threshold of an infinite line through a
	reference point along a direction.
	"""
	# ===================== Operation ==========================================

	# Load point cloud
	filepath = str(DATA_DIR / "point_clouds" / "can_vertical_3_downsampled.ply")
	point_cloud = io.load_point_cloud(filepath=filepath)
	logger.success(f"Loaded point cloud with {len(point_cloud.positions)} points")

	# Execute operation
	result_point_cloud = vitreous.segment_point_cloud_using_vector_proximity(
		distance_threshold=0.1,
		keep_outliers=False,
		point_cloud=point_cloud,
		reference_point=[0.0, 0.0, 0.0],
		reference_vector=[0.0, 0.0, 1.0],
	)
	logger.success(
		f"Segmented {len(result_point_cloud.positions)} points using vector proximity"
	)

	# ===================== Visualization  (Optional) ======================

def get_example_dict():
	"""Returns a dictionary mapping example names (without _example suffix) to their functions."""
	return {		
		# Calculate
		"calculate_axis_aligned_bounding_box": calculate_axis_aligned_bounding_box_example,   
		"calculate_oriented_bounding_box": calculate_oriented_bounding_box_example,
		"calculate_point_cloud_centroid": calculate_point_cloud_centroid_example, 
		"calculate_points_in_point_cloud": calculate_points_in_point_cloud_example,
		"calculate_plane_normal": calculate_plane_normal_example,
		
		# Cluster
		"cluster_point_cloud_using_dbscan": cluster_point_cloud_using_dbscan_example,  
		"cluster_point_cloud_based_on_density_jump": cluster_point_cloud_based_on_density_jump_example,
		
		# Convert
		"convert_mesh_to_point_cloud": convert_mesh_to_point_cloud_example,
		
		# Create
		"create_cylinder_mesh": create_cylinder_mesh_example, #TODO: Camera eye position needs to be adjusted
		"create_plane_mesh": create_plane_mesh_example,
		"create_sphere_mesh": create_sphere_mesh_example, #TODO: Camera eye position needs to be adjusted
		"create_torus_mesh": create_torus_mesh_example, #TODO: Camera eye position needs to be adjusted
		
		# Filters and estimators
		"estimate_principal_axis_within_radius": estimate_principal_axis_within_radius_example,
		"estimate_principal_axes": estimate_principal_axes_example,  #TODO: Missing point cloud file
		"filter_point_cloud_using_pass_through_filter": filter_point_cloud_using_pass_through_filter_example,
		"filter_point_cloud_using_bounding_box": filter_point_cloud_using_bounding_box_example,
		"filter_point_cloud_using_cylinder_base_removal": filter_point_cloud_using_cylinder_base_removal_example, 
		"filter_point_cloud_using_mask": filter_point_cloud_using_mask_example,  
		"filter_point_cloud_using_oriented_bounding_box": filter_point_cloud_using_oriented_bounding_box_example,  
		"filter_point_cloud_using_plane_proximity": filter_point_cloud_using_plane_proximity_example,
		"filter_point_cloud_using_plane_defined_by_point_normal_proximity": filter_point_cloud_using_plane_defined_by_point_normal_proximity_example,
		"filter_point_cloud_using_plane_splitting": filter_point_cloud_using_plane_splitting_example, 
		"filter_point_cloud_using_radius_outlier_removal": filter_point_cloud_using_radius_outlier_removal_example, 
		"filter_point_cloud_using_statistical_outlier_removal": filter_point_cloud_using_statistical_outlier_removal_example, 
		"filter_point_cloud_using_uniform_downsampling": filter_point_cloud_using_uniform_downsampling_example,
		"filter_point_cloud_using_viewpoint_visibility": filter_point_cloud_using_viewpoint_visibility_example, 
		"filter_point_cloud_using_voxel_downsampling": filter_point_cloud_using_voxel_downsampling_example,
		
		# Operations
		"add_point_clouds": add_point_clouds_example,
		"subtract_point_clouds": subtract_point_clouds_example,
		"scale_point_cloud": scale_point_cloud_example,
		"apply_transform_to_point_cloud": apply_transform_to_point_cloud_example,
		
		# Project
		"project_point_cloud_to_plane": project_point_cloud_to_plane_example,
		"project_point_cloud_to_plane_defined_by_point_normal": project_point_cloud_to_plane_defined_by_point_normal_example,
		
		# Reconstruct
		"reconstruct_mesh_using_convex_hull": reconstruct_mesh_using_convex_hull_example,
		"reconstruct_mesh_using_poisson": reconstruct_mesh_using_poisson_example,  #TODO, Arjun, not working because pcd doesnt have normals. compute normals inside vitreous?
		
		# Register
		"register_point_clouds_using_centroid_translation": register_point_clouds_using_centroid_translation_example,  
		"register_point_clouds_using_cuboid_translation_sampler_icp": register_point_clouds_using_cuboid_translation_sampler_icp_example, 
		"register_point_clouds_using_fast_global_registration": register_point_clouds_using_fast_global_registration_example,
		"register_point_clouds_using_point_to_plane_icp": register_point_clouds_using_point_to_plane_icp_example, #TODO: Arjun, add better visualization
		"register_point_clouds_using_point_to_point_icp": register_point_clouds_using_point_to_point_icp_example, #TODO: Arjun, add better visualization
		"register_point_clouds_using_rotation_sampler_icp": register_point_clouds_using_rotation_sampler_icp_example,
	   
		# Segment 
		"segment_point_cloud_using_color": segment_point_cloud_using_color_example, 
		"segment_point_cloud_using_plane": segment_point_cloud_using_plane_example,
		"segment_point_cloud_using_vector_proximity": segment_point_cloud_using_vector_proximity_example, #TODO for Haris
	}


def parse_args():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description="Run vitreous examples")
	parser.add_argument(
		"--example",
		type=str,
		help="Name of the example to run (without _example suffix) or use --list to see all available examples",
	)
	parser.add_argument(
		"--list", action="store_true", help="List all available examples"
	)
	parser.add_argument(
        "--all", action="store_true", help="Run all available examples"
    )
	parser.add_argument(
		"--pause", action="store_true", help="Pause between examples when running all"
	)
	return parser.parse_args()


def main():
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
	start_time = time.perf_counter()
	example_dict[args.example]()
	elapsed = time.perf_counter() - start_time
	logger.success(
    f"{args.example} example completed in {elapsed:.3f} seconds."
    )
	logger.success(f"{args.example} example completed.")


if __name__ == "__main__":
	main()
