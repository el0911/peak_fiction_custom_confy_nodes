from .nodes.custom_nodes import Extract_mask_with_scrible_map
from .nodes.custom_nodes import Cube_map_From_Panorama, Load_model_from_memory
from .nodes.custom_nodes import Generate_point_cloud_or_glb, Load_model_from_memory

NODE_CLASS_MAPPINGS = {
    "Extract_mask_with_scrible_map": Extract_mask_with_scrible_map,
    "PanoramaToCubemap": Cube_map_From_Panorama,
    "Load_model_from_memory":Load_model_from_memory,
    "Gen_glb_scene":Generate_point_cloud_or_glb
    # "ImageBlender": ImageBlender
}

# Optional: Add display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Extract_mask_with_scrible_map": "Extract mask with scrible map",
    "PanoramaToCubemap": "Panorama to Cubemap Converter",
    "Load_model_from_memory":"Load model vggt model from memory",
    "Gen_glb_scene":"Make a scene from images"
    # "ImageBlender": "Image Blender"
}

