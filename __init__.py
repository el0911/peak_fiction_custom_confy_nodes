from .nodes.custom_nodes import Extract_mask_with_scrible_map
from .nodes.custom_nodes import Cube_map_From_Panorama

NODE_CLASS_MAPPINGS = {
    "Extract_mask_with_scrible_map": Extract_mask_with_scrible_map,
    "PanoramaToCubemap": Cube_map_From_Panorama
    # "ImageBlender": ImageBlender
}

# Optional: Add display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Extract_mask_with_scrible_map": "Extract mask with scrible map",
    "PanoramaToCubemap": "Panorama to Cubemap Converter",
    # "ImageBlender": "Image Blender"
}

