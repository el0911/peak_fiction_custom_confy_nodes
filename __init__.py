from .nodes.custom_nodes import Extract_mask_with_scrible_map


NODE_CLASS_MAPPINGS = {
    "Extract_mask_with_scrible_map": Extract_mask_with_scrible_map,
    # "ImageBlender": ImageBlender
}

# Optional: Add display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Extract_mask_with_scrible_map": "Extract mask with scrible map",
    # "ImageBlender": "Image Blender"
}