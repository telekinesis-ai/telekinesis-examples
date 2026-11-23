# Third-party example data

Some example inputs loaded by the scripts in this repository (from `https://assets.telekinesis.ai/examples/v1/`) are derived from public research datasets. They are redistributed under the licences below, with the changes noted. These licences apply to the listed data files only, not to this repository's code (see `LICENSE.txt`).

| Dataset | Licence | Files |
|---|---|---|
| [XYZ Industrial Bin-Picking Dataset (XYZ-IBD)](#xyz-ibd) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | 37 |
| [ARMBench](#armbench) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 8 |
| [Visual Anomaly (VisA)](#visa) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 11 |
| [LOCO: Logistics Objects in Context](#loco) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | 4 |
| [FabriD: Fabric Defect Detection Dataset](#fabrid) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 2 |
| [MATWI: A Multimodal Automatic Tool Wear Inspection dataset](#matwi) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 1 |
| [T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects](#t-less) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 4 |

## XYZ-IBD

- **Dataset:** [XYZ Industrial Bin-Picking Dataset (XYZ-IBD)](https://xyz-ibd.github.io/)
- **Authors:** Junwen Huang, Jiaqi Hu, Peter KT Yu, Slobodan Ilic, Martin Sundermeyer, Benjamin Busam (TUM, MCML, Google, XYZ Robotics)
- **Citation:** J. Huang et al., "XYZ-IBD: Benchmarking Robust 6D Object Pose Estimation under Real-World Industrial Complexity", arXiv:2506.00599, 2025.
- **Licence:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Note:** Used with the written permission of the authors (J. Huang, P. Yu) to demonstrate use cases in Telekinesis documentation, with credit and citation.
- **Changes made by Telekinesis:** Cropped and/or resized from Photoneo/RealSense frames; some derived: depth maps back-projected to point clouds, ground-truth masks merged or cropped, 16-bit depth holes inpainted, single channels extracted. `xyzibd_flat_bars_noisy.png` contains simulated salt-and-pepper noise; `xyzibd_angle_brackets_bin_noisy.png` contains simulated low-light sensor noise (Poisson shot noise at 12 e- full well plus Gaussian read noise, sigma 10).

| File | Used by |
|---|---|
| `depth_images/xyzibd_flat_bar_pose_depth.png` | `estimate_planar_pose` |
| `depth_images/xyzibd_flat_bars_depth_16bit.png` | `normalize_image_intensity` |
| `images/xyzibd_angle_brackets_bin_noisy.png` | `filter_image_using_blur`, `filter_image_using_box` |
| `images/xyzibd_brackets_yellow_bin_rgb.jpg` | `crop_image_using_polygon` |
| `images/xyzibd_flanges_bin_gray.jpg` | `segment_image_using_sam3` |
| `images/xyzibd_flat_bar_mask_a.png` | `bitwise_or_images`, `calculate_mask_pca` |
| `images/xyzibd_flat_bar_mask_b.png` | `bitwise_or_images` |
| `images/xyzibd_flat_bar_pose_mask.png` | `estimate_planar_pose` |
| `images/xyzibd_flat_bars_bin_gray.png` | `segment_image_using_flood_fill`, `segment_image_using_otsu_threshold` |
| `images/xyzibd_flat_bars_denoised.png` | `filter_image_using_laplacian` |
| `images/xyzibd_flat_bars_gray.png` | `filter_image_using_morphological_blackhat` |
| `images/xyzibd_flat_bars_noisy.png` | `filter_image_using_median_blur` |
| `images/xyzibd_flat_bars_scene36_gray.jpg` | `filter_image_using_morphological_erode`, `segment_image_foreground_using_birefnet` |
| `images/xyzibd_flat_bars_scene36_gray_channel.png` | `merge_image_from_channels` |
| `images/xyzibd_flat_bars_scene36_height_channel.png` | `merge_image_from_channels` |
| `images/xyzibd_flat_bars_scene36_mask_channel.png` | `merge_image_from_channels` |
| `images/xyzibd_gear_bin_instances.png` | `overlay_images_using_weighted_overlay` |
| `images/xyzibd_gear_bin_photoneo.jpg` | `overlay_images_using_weighted_overlay` |
| `images/xyzibd_gears_bin_mask.png` | `filter_point_cloud_using_mask` |
| `images/xyzibd_lever_mask.png` | `calculate_mask_centroid` |
| `images/xyzibd_rods_bin_gray.png` | `segment_image_using_threshold`, `segment_image_using_yen_threshold` |
| `images/xyzibd_rods_bin_s56_gray.png` | `filter_image_using_frangi`, `filter_image_using_hessian`, `filter_image_using_sato` |
| `images/xyzibd_rods_instance_mask.png` | `transform_mask_using_blob_thinning` |
| `images/xyzibd_tie_rods_bin_gray.jpg` | `filter_image_using_morphological_close`, `filter_image_using_morphological_dilate` |
| `point_clouds/xyzibd_brackets_bin_photoneo_downsampled.ply` | `filter_point_cloud_using_radius_outlier_removal` |
| `point_clouds/xyzibd_brackets_bin_realsense.ply` | `filter_point_cloud_using_statistical_outlier_removal` |
| `point_clouds/xyzibd_flat_bar_segmented.ply` | `calculate_oriented_bounding_box` |
| `point_clouds/xyzibd_flat_bars_bin_downsampled.ply` | `calculate_axis_aligned_bounding_box`, `filter_point_cloud_using_plane_defined_by_point_normal_proximity`, `filter_point_cloud_using_plane_proximity`, `segment_point_cloud_using_plane`, `segment_point_cloud_using_vector_proximity` |
| `point_clouds/xyzibd_flat_bars_bin_floor_plane.ply` | `add_point_clouds` |
| `point_clouds/xyzibd_flat_bars_bin_without_floor.ply` | `add_point_clouds` |
| `point_clouds/xyzibd_flat_bars_scene_downsampled.ply` | `filter_point_cloud_using_oriented_bounding_box` |
| `point_clouds/xyzibd_gears_bin_camera_frame.ply` | `apply_transform_to_point_cloud` |
| `point_clouds/xyzibd_gears_bin_dense.ply` | `filter_point_cloud_using_voxel_downsampling` |
| `point_clouds/xyzibd_gears_bin_organized.ply` | `filter_point_cloud_using_mask` |
| `point_clouds/xyzibd_gears_bin_realsense_rgb.ply` | `segment_point_cloud_using_color` |
| `point_clouds/xyzibd_gears_bin_workcell_frame.ply` | `filter_point_cloud_using_bounding_box`, `filter_point_cloud_using_pass_through_filter`, `filter_point_cloud_using_plane_splitting`, `project_point_cloud_to_plane`, `project_point_cloud_to_plane_defined_by_point_normal` |
| `point_clouds/xyzibd_rods_bin_downsampled.ply` | `estimate_principal_axis_within_radius` |

## ARMBench

- **Dataset:** [ARMBench](https://www.armbench.com/)
- **Authors:** Amazon Robotics (C. Mitash, F. Wang, S. Lu, V. Terhuja, T. Garaas, F. Polido, M. Nambi)
- **Citation:** C. Mitash et al., "ARMBench: An Object-centric Benchmark Dataset for Robotic Manipulation", ICRA 2023, arXiv:2303.16382.
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Changes made by Telekinesis:** Cropped, resized and in one case rotated; one channel-swapped copy for the channel-merge example.

| File | Used by |
|---|---|
| `images/armbench_blue_tote_air_filter_cartridge.jpg` | `segment_image_using_grab_cut` |
| `images/armbench_blue_tote_kraft_parcels.jpg` | `segment_image_using_hsv`, `segment_image_using_rgb` |
| `images/armbench_gripper_fiducial_markers.jpg` | `segment_image_using_local_threshold` |
| `images/armbench_parcel_shipping_labels.jpg` | `segment_image_using_adaptive_threshold` |
| `images/armbench_tote_two_cartons.jpg` | `segment_image_using_lab`, `segment_image_using_slic_superpixel` |
| `images/armbench_yellow_tote_carton_polybag.jpg` | `rotate_image`, `split_image_into_channels`, `translate_image` |
| `images/armbench_yellow_tote_mailers.jpg` | `convert_image_color_space` |
| `images/armbench_yellow_tote_white_cartons.jpg` | `filter_segments_by_mask` |

## VisA

- **Dataset:** [Visual Anomaly (VisA)](https://registry.opendata.aws/visa/)
- **Authors:** Amazon (Y. Zou, J. Jeong, L. Pemula, D. Zhang, O. Dabeer)
- **Citation:** Y. Zou et al., "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation", ECCV 2022, arXiv:2207.14315.
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Changes made by Telekinesis:** Resized and/or cropped; one downsampled copy for the pyramid-upsampling example. `visa_pcb_ir_sensor_*` (pcb3): an anomalous board (missing IR receiver) registered to a normal board by homography and both cropped. `visa_pcb_driver_board_*` (pcb2): rotated 180°, cropped, upscaled 1.25× and binarised (bright silkscreen, pads and leads = white); the anomalous board (two ICs removed) was registered to a normal board and only its defect region (VisA ground-truth mask, dilated) was pasted into the normal board's binary to make `visa_pcb_driver_board_missing_ics_binary.png`; `visa_pcb_driver_board_layout_dark.png` is the inverted reference binary.

| File | Used by |
|---|---|
| `images/visa_pcb_charger_board.jpg` | `filter_image_using_scharr`, `filter_image_using_sobel` |
| `images/visa_pcb_charger_module.jpg` | `crop_image_center`, `crop_image_using_bounding_boxes` |
| `images/visa_pcb_driver_board_layout_dark.png` | `filter_image_using_morphological_hitmiss` |
| `images/visa_pcb_driver_board_missing_ics_binary.png` | `bitwise_xor_images` |
| `images/visa_pcb_driver_board_reference_binary.png` | `bitwise_not_image`, `bitwise_xor_images` |
| `images/visa_pcb_ir_sensor_missing_receiver.png` | `bitwise_difference_images` |
| `images/visa_pcb_ir_sensor_reference.png` | `bitwise_difference_images` |
| `images/visa_pcb_sensor_board.jpg` | `filter_image_using_morphological_tophat` |
| `images/visa_pcb_ultrasonic_board.png` | `filter_image_using_morphological_gradient` |
| `images/visa_pcb_ultrasonic_sensor.jpg` | `resize_image`, `resize_image_with_aspect_fit`, `transform_image_using_pyramid_downsampling` |
| `images/visa_pcb_ultrasonic_sensor_small.png` | `transform_image_using_pyramid_upsampling` |

## LOCO

- **Dataset:** [LOCO: Logistics Objects in Context](https://github.com/tum-fml/loco)
- **Authors:** C. Mayershofer, D.-M. Holm, B. Molter, J. Fottner (TUM fml)
- **Citation:** C. Mayershofer et al., "LOCO: Logistics Objects in Context", ICMLA 2020.
- **Licence:** [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
- **Note:** Public-domain dedication; credit given as a courtesy.
- **Changes made by Telekinesis:** Resized; square crop for the Qwen example; an unblurred face in `loco_crossdock_workers.jpg` was pixelated by Telekinesis.

| File | Used by |
|---|---|
| `images/loco_crossdock_workers.jpg` | `detect_objects_using_rfdetr`, `detect_objects_using_yolox`, `segment_image_using_ycrcb` |
| `images/loco_dark_warehouse_racks.jpg` | `enhance_image_using_auto_gamma_correction`, `enhance_image_using_clahe` |
| `images/loco_forklift_loading_dock.jpg` | `detect_objects_using_grounding_dino` |
| `images/loco_pallet_truck_stillages_square.jpg` | `detect_objects_using_qwen` |

## FabriD

- **Dataset:** [FabriD: Fabric Defect Detection Dataset](https://data.mendeley.com/datasets/vncdwps24r/3)
- **Authors:** Sabbir Ahmed, Ismam Ahmed, Nishat Mahzabin, Tahmina Islam (University of Information Technology and Sciences, Dhaka)
- **Citation:** S. Ahmed et al., "FabriD: Fabric Defect Detection Dataset", Mendeley Data, V3, doi:10.17632/vncdwps24r.3.
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Changes made by Telekinesis:** Cropped; a threshold mask derived for the morphological-open example.

| File | Used by |
|---|---|
| `images/fabrid_black_knit_lint_threshold_mask.png` | `filter_image_using_morphological_open` |
| `images/fabrid_black_knit_missing_yarn.jpg` | `filter_image_using_gabor`, `filter_image_using_meijering` |

## MATWI

- **Dataset:** [MATWI: A Multimodal Automatic Tool Wear Inspection dataset](https://doi.org/10.48804/GK6LHH)
- **Authors:** Lars De Pauw, Tom Jacobs, Toon Goedemé (KU Leuven)
- **Citation:** L. De Pauw, T. Jacobs, T. Goedemé, "Replication Data for: MATWI", KU Leuven RDR, 2025, doi:10.48804/GK6LHH.
- **Licence:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **Note:** ShareAlike: the modified image(s) below are themselves licensed under CC BY-SA 4.0.
- **Changes made by Telekinesis:** Cropped and resized.

| File | Used by |
|---|---|
| `images/matwi_milling_insert_wear_macro.jpg` | `segment_image_using_focus_region` |

## T-LESS

- **Dataset:** [T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects](https://cmp.felk.cvut.cz/t-less/)
- **Authors:** T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis, X. Zabulis (CTU Prague)
- **Citation:** T. Hodaň et al., "T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects", WACV 2017.
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Changes made by Telekinesis:** Depth back-projected to point clouds (cropped to the table/part); CAD meshes sliced and converted to GLB; one RGB frame cropped and upscaled 2.75x (bicubic).

| File | Used by |
|---|---|
| `images/tless_lamp_socket_blocks_touching.jpg` | `segment_image_using_watershed` |
| `meshes/tless_lamp_socket_sleeve_obj14.glb` | `filter_mesh_using_cylinder_base_removal` |
| `point_clouds/tless_parts_on_table_scene11.ply` | `cluster_point_cloud_based_on_density_jump` |
| `point_clouds/tless_socket_block_scan_obj08.ply` | `reconstruct_mesh_using_convex_hull` |
