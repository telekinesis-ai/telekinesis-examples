  <div align="center">
    <p>
      <a align="center" href="https://github.com/telekinesis-ai" target="_blank">
        <img
          width="100%"
          src="https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/Telekinesis+Banner.png"
        >
      </a>
    </p>

    <br>

    <a href="https://github.com/telekinesis-ai/telekinesis-examples">Telekinesis Examples</a> |
    <a href="https://github.com/telekinesis-ai/telekinesis-data">Telekinesis Data</a>
    <br>

    <a href="https://pypi.org/project/telekinesis-ai/">
      <img src="https://img.shields.io/pypi/v/telekinesis-ai" />
    </a>
    <img src="https://img.shields.io/pypi/l/telekinesis-ai" />
    <a href="https://pypi.org/project/telekinesis-ai/">
      <img src="https://img.shields.io/pypi/pyversions/telekinesis-ai" />
    </a>
  </div>

  # Telekinesis Examples

  This directory contains **example scripts** demonstrating how to use the **Telekinesis Python SDK** for robotics and computer vision workflows.

  These examples focus on **practical usage** of Telekinesis APIs, including:

  * point clouds and meshes
  * 3D geometry processing
  * perception utilities
  * optional 3D visualization

  ---

  ## Setup

  ### Create an API Key 

  Create a Telekinesis account and generate an API key:

  [Create a Telekinesis account!](https://main.d21qivo5yi7kdj.amplifyapp.com/api-keys)

  Store the key in a safe location, like a `.zshrc` file or another text file on your computer.


  ### Configure the API Key 

  Export the API key as an environment variable:

  #### macOS / Linux

  ```bash
  export TELEKINESIS_API_KEY="your_api_key_here"
  ```

  #### Windows (PowerShell)

  ```powershell
  setx TELEKINESIS_API_KEY "your_api_key_here"
  ```

  Restart your terminal after setting the variable.

  Verify that the key is set:

  ```bash
  echo $TELEKINESIS_API_KEY       # macOS / Linux
  echo $Env:TELEKINESIS_API_KEY   # Windows
  ```

  Telekinesis SDK is configured to automatically read your API key from the system environment.

  ### Install the Telekinesis SDK

  ```bash
  pip install telekinesis-ai
  ```

  ### Clone repository along with data(submodule)

  If you're setting up for the first time, clone with submodules:

  ```bash
  git clone --recurse-submodules https://github.com/telekinesis-ai/telekinesis-examples.git
  ```

  If you cloned this repository, initialize and update the submodule:

  ```bash
  git submodule update --init --recursive
  ```

  **Note:** The examples require data files from the [telekinesis-data](https://gitlab.com/telekinesis-ai/telekinesis-data) repository, which is included as a git submodule.

  ### Install example-only dependencies

  ```bash
  pip install numpy scipy opencv-python rerun-sdk==0.27.3
  ```
  These dependencies are **only required for running examples**, not for the core SDK.

  **Note:** The data path is configured in `examples/vitreous_examples.py` on line 16. If you need to use a different data location, modify the `DATA_DIR` variable there.

  ---

  ## Running Examples

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

  ### Run all examples

  ```bash
  python examples/vitreous_examples.py --all
  ```

  To pause between examples:

  ```bash
  python examples/vitreous_examples.py --all --pause
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
  │   └── vitreous_examples.py    # Main examples script with all example functions
  ├── telekinesis-data/           # Git submodule containing example data files
  │   ├── point_clouds/           # PLY point cloud files
  │   ├── meshes/                 # GLB mesh files
  │   └── images/                 # Image files
  ├── README.md                   # This file
  ├── LICENSE.txt                 # License file
  └── .gitmodules                 # Git submodule configuration
  ```

  All examples are contained within `vitreous_examples.py`. Use the `--list` flag to see all available examples.

  The data path is configured in `examples/vitreous_examples.py` on line 16. By default, it points to `telekinesis-data/` in the repository root. To use a custom data location, modify the `DATA_DIR` variable.

  ---

  ## Documentation

  Full SDK documentation is available at:
  [https://docs.telekinesis.ai/getting-started/quickstart.html](https://docs.telekinesis.ai/getting-started/quickstart.html)


  ## License

  See LICENSE.txt for details.

  ## Support

  For issues and questions:
  - Create an issue
  - Contact the Telekinesis development team


  <p align="center">
    <a href="https://github.com/telekinesis-ai">
      <img src="https://cdn.jsdelivr.net/npm/simple-icons@v11/icons/github.svg" width="18" alt="GitHub"/>
    </a>
    &nbsp;
    <a href="https://www.linkedin.com/company/telekinesis-ai/">
      <img src="https://cdn.jsdelivr.net/npm/simple-icons@v11/icons/linkedin.svg" width="18" alt="LinkedIn"/>
    </a>
    &nbsp;
    <a href="https://x.com/telekinesis_ai">
      <img src="https://cdn.jsdelivr.net/npm/simple-icons@v11/icons/x.svg" width="18" alt="X"/>
    </a>
    &nbsp;
    <a href="https://discord.gg/7NnQ3bQHqm">
      <img src="https://cdn.jsdelivr.net/npm/simple-icons@v11/icons/discord.svg" width="18" alt="Discord"/>
    </a>
  </p>
