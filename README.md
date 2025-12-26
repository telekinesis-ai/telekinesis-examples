<div align="center">
  <p>
    <a align="center" href="https://github.com/telekinesis-ai" target="_blank">
      <img
        width="100%"
        src="assets/telekinesis_banner.png"
      >
    </a>
  </p>

  <br>

  <a href="https://github.com/telekinesis-ai/telekinesis-examples">Telekinesis Examples</a> |
  <a href="https://gitlab.com/telekinesis/telekinesis-data">Telekinesis Data</a>
  <br>

</div>

# Telekinesis Examples

This directory contains **example scripts** demonstrating how to use the **Telekinesis SDK (Python)** for robotics and computer vision workflows.

These examples focus on **practical usage** of Telekinesis APIs, including:

* point clouds and meshes
* 3D geometry processing
* perception utilities
* optional 3D visualization

---

## Getting Started

### Requirements

To run these examples, you’ll need:
- Python 3.11
- A Telekinesis account
- A valid Telekinesis API key

1. Create an API Key

Create a Telekinesis account and generate an API key: [Create a API Key](https://platform.telekinesis.ai/api-keys)

2. Configure the API Key

Export your API key as an environment variable:
```bash
# macOS / Linux
export TELEKINESIS_API_KEY="your_api_key"
```

```shell
# Windows
setx TELEKINESIS_API_KEY "your_api_key"
```
> After running setx on Windows, restart your terminal for the change to take effect.

3. Install the Telekinesis SDK
```bash
pip install telekinesis-ai
```
---

## Repository Setup

### Clone the repository

```bash
git clone --recurse-submodules https://github.com/telekinesis-ai/telekinesis-examples.git
cd telekinesis-examples
```
> This will also download the `telekinesis-data` repository, which contains sample point clouds, meshes, and images used by the examples.

If you already cloned the repository without submodules, you can fetch them with:
```bash
git submodule update --init --recursive
```

All point cloud processing examples are contained within `vitreous_examples.py`.

### Install example-only dependencies

These dependencies are **only required for running examples**, not for the core SDK.

```bash
pip install numpy scipy opencv-python rerun-sdk==0.27.3 loguru
```

Once installed, you’re ready to run any example in this repository.

---

## Running Examples

### Quick start example

Run a simple example to verify everything is working:

```bash
python examples/vitreous_examples.py --example filter_point_cloud_using_voxel_downsampling
```

Terminal output:
```bash
2025-12-17 17:59:58.437 | SUCCESS  | __main__:main:4783 - Running filter_point_cloud_using_voxel_downsampling example...
2025-12-17 17:59:58.852 | SUCCESS  | __main__:filter_point_cloud_using_voxel_downsampling_example:2998 - Filtered points using voxel downsampling
```

This will also open a `rerun` visualization window showing the input point cloud and the filtered output point cloud.

<div style="display: flex; flex-direction: column; gap: 20px; margin: 20px 0;">
  <div style="flex: 1;">
    <h4>Visualization Output:</h4>
    <img src="assets/voxel_downsample_input_output.png" width="400">
  </div>
</div>

### List available examples
```bash
python examples/vitreous_examples.py --list
```

This will display all available examples, such as:
- `calculate_axis_aligned_bounding_box`
- `calculate_oriented_bounding_box`
- `calculate_point_cloud_centroid`
- `cluster_point_cloud_using_dbscan`
- `convert_mesh_to_point_cloud`
- `filter_point_cloud_using_voxel_downsampling`
- ... and many more

### Run a specific example
```bash
python examples/vitreous_examples.py --example calculate_axis_aligned_bounding_box
```

The example name should be provided **without** the `_example` suffix.

### Run a specific example with your own data
The data path is configured in `examples/vitreous_examples.py` on line 16. 
By default, it points to `telekinesis-data/` in the repository root. 

To use a custom data location, modify the `DATA_DIR` variable.

```bash
python examples\vitreous_examples.py --example filter_point_cloud_using_statistical_outlier_removal
```

---

## Example Categories

Examples are organized by functionality and common workflows:

* **Point Cloud Calculations**
  * Axis-aligned and oriented bounding boxes
  * Centroids and point counting
  * Plane normal estimation
  * Principal axes estimation

* **Point Cloud Filtering**
  * Pass-through, bounding box, and mask-based filtering
  * Outlier removal (statistical and radius-based)
  * Downsampling (uniform and voxel-based)
  * Plane-based filtering and splitting
  * Cylinder base removal
  * Viewpoint visibility filtering

* **Point Cloud Clustering & Segmentation**
  * DBSCAN clustering
  * Density-based clustering
  * Color-based segmentation
  * Plane-based segmentation
  * Vector proximity segmentation

* **Point Cloud Transformations**
  * Adding and subtracting point clouds
  * Scaling and applying transforms
  * Projecting to planes

* **Point Cloud Registration**
  * Centroid-based translation
  * ICP registration (point-to-point, point-to-plane)
  * Fast global registration
  * Rotation and cuboid translation samplers

* **Mesh Operations**
  * Creating primitive meshes (cylinder, plane, sphere, torus)
  * Converting meshes to point clouds
  * Reconstructing meshes from point clouds (convex hull, Poisson)

* **Visualization (Optional)**
  * Rerun-based 3D visualization
  * Camera controls and overlays

All examples include optional visualization sections that can be removed if only numerical output is needed.

---

## Directory Structure

```text
telekinesis-examples/
├── examples/
│   └── datatypes_examples.py   # Datatypes examples
│   └── vitreous_examples.py    # Main vitreous examples script with all example functions
├── telekinesis-data/           # Git submodule containing example data files
│   ├── point_clouds/           # PLY point cloud files
│   ├── meshes/                 # GLB mesh files
│   └── images/                 # Image files
├── README.md                   # This file
├── LICENSE.txt                 # License file
└── .gitmodules                 # Git submodule configuration
```

---

## Documentation
Full SDK documentation is available at: [Telekinesis Docs](https://docs.telekinesis.ai/)

## Support

For issues and questions:
- Create a GitHub issue
- Contact the Telekinesis development team

<p align="center">
  <a href="https://github.com/telekinesis-ai">GitHub</a>
  &nbsp;•&nbsp;
  <a href="https://www.linkedin.com/company/telekinesis-ai/">LinkedIn</a>
  &nbsp;•&nbsp;
  <a href="https://x.com/telekinesis_ai">X</a>
  &nbsp;•&nbsp;
  <a href="https://discord.gg/7NnQ3bQHqm">Discord</a>
</p>
