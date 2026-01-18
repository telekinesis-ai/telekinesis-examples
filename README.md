<div align="center">
  <p>
    <a href="https://github.com/telekinesis-ai" target="_blank">
      <img width="100%" src="assets/telekinesis_banner.png" />
    </a>
  </p>

  <p align="center">
    <a href="https://pypi.org/project/telekinesis-ai/">
      <img src="https://img.shields.io/pypi/v/telekinesis-ai" />
    </a>
    <a href="https://pypi.org/project/telekinesis-ai/">
      <img src="https://img.shields.io/pypi/pyversions/telekinesis-ai" />
    </a>
    <a href="https://pypi.org/project/telekinesis-ai/">
      <img src="https://img.shields.io/pypi/l/telekinesis-ai" />
    </a>
    <a href="https://docs.telekinesis.ai">
      <img src="https://img.shields.io/badge/docs-telekinesis.ai-blue" />
    </a>
  </p>

  <p>
    <a href="https://github.com/telekinesis-ai/telekinesis-examples">Telekinesis Examples</a>
    &nbsp;•&nbsp;
    <a href="https://gitlab.com/telekinesis/telekinesis-data">Telekinesis Data</a>
  </p>
</div>

# Telekinesis SDK Examples – Robotics & Computer Vision in Python

This repository provides **Python examples for the Telekinesis SDK**, demonstrating **Telekinesis Skills - atomic robotics control and computer vision operations** - using Telekinesis APIs.

Each example in this repository:
- Demonstrates a **single Telekinesis SDK function call**
- Focuses on one **well-defined perception or robotics operation**
- Is designed to be **standalone, readable, and composable**

These examples are intended as **indiviudal Skills - building blocks** that can be combined to create larger **industrial robotics and computer vision pipelines**.

> Note: Use examples as **reference implementations** to integrate into your own projects.

## Table of Contents

- [Telekinesis SDK Modules](#telekinesis-sdk-modules)
- [Getting Started](#getting-started)
- [Repository Setup](#repository-setup)
- [Running Examples](#running-examples)
- [How to Use These Examples](#how-to-use-these-examples)
- [Example Categories](#example-categories)
- [Directory Structure](#directory-structure)
- [Who Is This For?](#who-is-this-for)
- [Next Steps](#next-steps)
- [Support](#support)


## Telekinesis SDK Modules

### Available Today

1. **Vitreous** – 3D point cloud and mesh processing  
2. **Pupil** – 2D image processing

### In Active Development (Not in the examples)

The SDK is modular and continuously expanding. Upcoming modules include:

- **Retina** – 2D object detection & visual foundation models  
- **Cornea** – 2D image segmentation 
- **Illusion** – Synthetic data generation 
- **Iris** – Model training & fine-tuning

And many more...

For a full overview of the Telekinesis SDK ecosystem, see the [Official Documentation](https://docs.telekinesis.ai).


## Getting Started

Follow these steps to run **Telekinesis SDK examples** on your local machine.

These instructions are self-contained and let you run examples directly from this repository.

> This setup is also available in the official documentation:
> [Quickstart Guide](https://docs.telekinesis.ai/getting-started/quickstart.html)

### Requirements

To run these examples, you’ll need:
- Python 3.11 or 3.12
- A Telekinesis account
- A valid Telekinesis API key

1. Create an API Key

Create a Telekinesis account, generate an API key and **save it**: [Create an API Key](https://platform.telekinesis.ai/api-keys)

2. Configure the API Key

Run the below command to export your API key as an environment variable:
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


## Repository Setup

### Clone the repository

```bash
git clone --recurse-submodules https://github.com/telekinesis-ai/telekinesis-examples.git
cd telekinesis-examples
```
> This will also download the [telekinesis-data repository](https://gitlab.com/telekinesis/telekinesis-data), which contains sample point clouds, meshes, and images used by the examples.

If you already cloned the repository without submodules, you can fetch them with:
```bash
git submodule update --init --recursive
```

### Install example-only dependencies

These dependencies are **required for running examples**:

```bash
pip install numpy scipy opencv-python rerun-sdk==0.27.3 loguru
```

Once installed, you’re ready to run any example in this repository.


## Running Examples

### Quick start example

Run a simple example to verify your setup:

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

The following commands list all available **Skills**.

**`Vitreous` module (point cloud and mesh processing):**
```bash
python examples/vitreous_examples.py --list
```

This will display all available examples, such as:
- `calculate_axis_aligned_bounding_box`
- `calculate_oriented_bounding_box`
- `calculate_point_cloud_centroid`
- ... [and many more](https://docs.telekinesis.ai/vitreous_sdk/vitreous_overview.html#overview-of-skills)

**`Pupil` module (image processing):**
```bash
python examples/pupil_examples.py --list
```

This will display all available examples, such as:
- `enhance_image_using_auto_gamma_correction`
- `enhance_image_using_clahe`
- ... [and many more](https://docs.telekinesis.ai/pupil_sdk/pupil_overview.html#overview-of-skills)

### Run a specific example

```bash
python examples/vitreous_examples.py --example calculate_axis_aligned_bounding_box        # Vitreous
python examples/pupil_examples.py --example filter_image_using_morphological_gradient     # Pupil
```

### Run a specific example with your own data

By default, examples load data from the bundled `telekinesis-data` submodule.

To use your own data, 

1. Create `your_data_directory` with a structure:
```text
<your_data_directory>/        
  ├── images/                 # Image files
  ├── point_clouds/           # PLY point cloud files
  └── meshes/                 # GLB mesh files
```

2. Update the `DATA_DIR` variable with `your_data_directory` location in the example scripts:

- [Vitreous](https://github.com/telekinesis-ai/telekinesis-examples/blob/137792e9eddc33ed666c1a139c8ac660b59d4973/examples/vitreous_examples.py#L18)

- [Pupil](https://github.com/telekinesis-ai/telekinesis-examples/blob/137792e9eddc33ed666c1a139c8ac660b59d4973/examples/pupil_examples.py#L13)

```bash
python examples\vitreous_examples.py --example filter_point_cloud_using_statistical_outlier_removal
```


## How to Use In Your Project

Each example is:

- **Standalone** – run independently
- **Modifiable** – easy to adapt to your own data

**Steps to use in project:**

1. Find an example matching your use case
2. Run it using the provided command
3. Inspect numerical output or visualization
4. Copy the relevant SDK calls into your own project

**Typical use cases:**

By composing multiple **Skills** you can build below **robotics and computer vision workflows** such as:

- **3D and 6D pose estimation**
- **ICP-based point cloud registration and alignment**
- **Point cloud preprocessing for grasping and manipulation**
- **Industrial computer vision and inspection workflows**
- **Image processing for detection, segmentation, and measurement**
- **Robotics control pipelines**


## Example Categories

Examples are grouped into modules for - **vision, robotics and agentic(coming soon) workflows**, making it easy to explore specific capabilities of the Telekinesis SDK.

**Visualization**

All examples include **optional visualization using Rerun**, enabling:

- Interactive inspection of point clouds and images
- Debugging pipelines visually

Visualization can be disabled if only numerical output is required.

<details>
<summary><strong>Vitreous</strong></summary>

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

</details>

<details>
<summary><strong>Pupil</strong></summary>

* **Image Filtering**
  * **Morphological filters**  
    Erode, Dilate, Open, Close, Gradient, Top-hat, Black-hat    

  * **Ridge & structure enhancement**  
    Frangi, Hessian, Sato, Meijering    

  * **Edge & sharpening filters**  
    Laplacian, Sobel, Scharr, Gabor   

  * **Smoothing & denoising**  
    Gaussian blur, Median blur, Box filter, Bilateral filter, Blur

* **Image Enhancement**
  * CLAHE (contrast enhancement)
  * Auto gamma correction
  * White balance

* **Image Transformation**
  * Pyramid downsampling & upsampling
  * Binary mask thinning (blob thinning)

</details>    


## Directory Structure

The repository is organized to separate **example scripts** from **sample data**, enabling easy customization and extension.

```text
telekinesis-examples/
├── examples/
│   └── datatypes_examples.py   # Datatypes examples script with all example functions
│   └── vitreous_examples.py    # vitreous examples script with all example functions
│   └── pupil_examples.py       # Pupil examples script with all example functions
├── telekinesis-data/           # Git submodule containing example data files
│   ├── images/                 # Image files
│   ├── point_clouds/           # PLY point cloud files
│   └── meshes/                 # GLB mesh files
├── README.md                   # This file
├── LICENSE.txt                 # License file
├── .gitignore                  # License file
└── .gitmodules                 # Git submodule configuration
```


## Who Is This For?

This repository is intended for:

- Robotics engineers
- Computer Vision engineers
- Teams building Physical AI and robotic perception systems


## Next Steps

- Explore the full SDK capabilities at [Telekinesis Docs](https://docs.telekinesis.ai).
- Integrate Telekinesis into your own robotics or vision pipelines.
- Join the [Discord community](https://discord.gg/7NnQ3bQHqm) to ask questions and share feedback.


## Support

For issues and questions:
- Create a GitHub [issue](https://github.com/telekinesis-ai/telekinesis-examples/issues)
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
