"""Demonstrates merging single-channel images into a multi-channel image."""

from loguru import logger
import rerun as rr

from telekinesis import pupil, datatypes


def merge_image_from_channels_example():
    """Merges three single-channel images into a false-colour composite."""
    # ===================== Load Images =========================================
    # Three aligned single-channel views of the same bin of flat bars:
    # gray = appearance, height = depth, mask = part mask.
    # Merging them gives a false-colour composite, e.g. as a multi-channel input
    # for a part-detection network in a bin-picking cell.
    # Example data: XYZ-IBD (CC BY-NC-SA 4.0), modified. Credits and licences: THIRD_PARTY_DATA.md
    image_urls = [
        "https://assets.telekinesis.ai/examples/v1/images/xyzibd_flat_bars_scene36_gray_channel.png",
        "https://assets.telekinesis.ai/examples/v1/images/xyzibd_flat_bars_scene36_height_channel.png",
        "https://assets.telekinesis.ai/examples/v1/images/xyzibd_flat_bars_scene36_mask_channel.png",
    ]
    channel_names = ["gray", "height", "mask"]
    image_channels = [datatypes.Image.from_url(url).to_grayscale() for url in image_urls]

    # ===================== Run Skill ==========================================
    filtered_image = pupil.merge_image_from_channels(channels=image_channels)

    # ===================== Log ================================================
    logger.success(f"Merged {len(image_channels)} channels")
    logger.success(f"Result: {filtered_image}")

    # ===================== Visualization  (Optional) ======================
    rr.init("merge_image_from_channels_example", spawn=True)
    for i, channel_image in enumerate(image_channels):
        datatypes.visualize(channel_image, entity_path=f"{i + 1}-{channel_names[i]}")
    datatypes.visualize(filtered_image, entity_path="4-merged")

if __name__ == "__main__":
    merge_image_from_channels_example()
