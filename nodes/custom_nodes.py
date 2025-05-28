from ..utils.extract_mask_with_scrible_map import ExtractMaskFromScribbleMap
from ..utils.panorama_to_cube_map import PanoramaToCubemap
from ..utils.visual_utils import predictions_to_glb

import numpy as np
import cv2
import logging
import sys
from PIL import Image
import os
from ..vggt.models.vggt import VGGT
from ..vggt.utils.load_fn import load_and_preprocess_images
from ..vggt.utils.pose_enc import pose_encoding_to_extri_intri
from ..vggt.utils.geometry import unproject_depth_map_to_point_map


import torch  # If using PyTorch
# import tensorflow as tf  # Uncomment if using TensorFlow

def tensor_to_numpy(tensor):
    # Convert the tensor to a numpy array
    numpy_image = tensor.numpy()

    # Convert the numpy array to a cv2 image
    cv2_image = np.transpose(numpy_image, (1, 2, 0))
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return cv2_image


class Extract_mask_with_scrible_map:
    """
    A node that extracts a mask from a scribble map using the original image as reference.
    Returns a MASK type that can be used with other mask-compatible nodes.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "scribble_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_map"
    CATEGORY = "peakfiction/custom"

   # Set up logging configuration
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    
    def get_map(self, original_image, scribble_image):
        try:
            # If input is already a NumPy array, skip conversion
            if not isinstance(original_image, np.ndarray):
                original_image = np.clip(
                    255.0 * original_image.cpu().numpy().squeeze(), 0, 255
                ).astype(np.uint8)

            if not isinstance(scribble_image, np.ndarray):
                scribble_image = np.clip(
                    255.0 * scribble_image.cpu().numpy().squeeze(), 0, 255
                ).astype(np.uint8)

            # Debugging: Check the converted array shapes
            logging.debug(f"Original Image Shape: {original_image.shape}")
            logging.debug(f"Scribble Image Shape: {scribble_image.shape}")

            # Your further processing logic here...
            # For example: compute some map or apply transformations

            # Ensure images are in BGR format for OpenCV
            if original_image.shape[-1] == 4:  # RGBA
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
                logging.debug("Converted original image from RGBA to BGR.")
            elif original_image.shape[-1] == 3:  # RGB
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                logging.debug("Converted original image from RGB to BGR.")

            if scribble_image.shape[-1] == 4:  # RGBA
                scribble_image = cv2.cvtColor(scribble_image, cv2.COLOR_RGBA2BGR)
                logging.debug("Converted scribble image from RGBA to BGR.")
            elif scribble_image.shape[-1] == 3:  # RGB
                scribble_image = cv2.cvtColor(scribble_image, cv2.COLOR_RGB2BGR)
                logging.debug("Converted scribble image from RGB to BGR.")

            # Optionally use ExtractMaskFromScribbleMap
            mask = ExtractMaskFromScribbleMap.get_map(original_image, scribble_image)
            
            # print(type(original_image_))
            print(type(mask))
            print(type(mask))
            print(type(mask))
            print(type(mask))
            print(mask)
            
            # Assuming 'mask' is the NumPy array you got from the ExtractMaskFromScribbleMap method
            # # Convert the NumPy array to a tensor
            # mask_tensor = torch.from_numpy(mask)

            # # Reshape to add batch and channel dimensions (if needed)
            # mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

            # # Optionally, convert to float type for image processing or neural networks
            # mask_tensor = mask_tensor.float()
 


            # Check if mask is returned
            # if mask is None:
            #     logging.warning("No mask found, returning original image.")
            #     return original_image  # Return original image if no mask is found
            
            print("Mask successfully created.")
            return (mask,)
        
        except Exception as e:
            # Log the exception details
            logging.error(f"An error occurred while processing images: {str(e)}")
            # In case of an error, return the original image
            return (original_image,)


# node to turn a panoram into a cubemap
class Cube_map_From_Panorama:
    """
    A node that converts a panorama image into a cubemap.
    Returns a list of 6 images representing the cubemap faces.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    FUNCTION = "convert_to_cubemap"
    CATEGORY = "peakfiction/custom"

    def convert_to_cubemap(self, panorama_image):
        # Placeholder for actual cubemap conversion logic
        # For now, just returning the input image six times
        #  make sure input is NDArray: Numpy array with shape [H, W, C].
        
        if not isinstance(panorama_image, np.ndarray):
            panorama_image = np.clip(
                255.0 * panorama_image.cpu().numpy().squeeze(), 0, 255
            ).astype(np.uint8)

        
        panorama_images = PanoramaToCubemap.panorama_to_cubemap(panorama_image, cube_size=512)
        print(f" length of images: {len(panorama_images)}")
        return (
            torch.from_numpy(panorama_images[0]).float().unsqueeze(0) / 255.0,
            torch.from_numpy(panorama_images[1]).float().unsqueeze(0) / 255.0,  
            torch.from_numpy(panorama_images[2]).float().unsqueeze(0) / 255.0,
            torch.from_numpy(panorama_images[3]).float().unsqueeze(0) / 255.0,
            torch.from_numpy(panorama_images[4]).float().unsqueeze(0) / 255.0,
            torch.from_numpy(panorama_images[5]).float().unsqueeze(0) / 255.0,
        )

    
# class to load a model from memory must be in the models/vggt folder loads all models in specified folder load a list of files from a folder and show it as options to the user its a select option as input not string
class Load_model_from_memory:
    """
    A node that loads a model from memory.
    Returns the loaded model.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # load all files in the models/vggt folder and show them as options
        model_folder = os.path.join(os.path.dirname(__file__), "../../models/vggt")
        # Create the directory if it does not exist
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_files = [f for f in os.listdir(model_folder) if f.endswith('.pt') or f.endswith('.pth')]
        return {
            "required": {
                "model_name": (model_files,),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "peakfiction/custom"

    def load_model(self, model_name):
        # Placeholder for actual model loading logic
        # For now, just returning a dummy model
        try:
            model_path = model_folder = os.path.join(os.path.dirname(__file__), "../../models/vggt")
            model_path = model_folder = os.path.join(model_path,model_name)
            state_dict = torch.load(model_path)

            device = "cuda" if torch.cuda.is_available() else "cpu"

            model = VGGT()
            model.load_state_dict(state_dict)
            model.eval()
            model = model.to(device)

            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available. Check your environment.")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")




        print(f"Loading model: {model_name}")
        return (model,)  # Replace with actual model loading logic
class Generate_point_cloud_or_glb:
            """
            A node that uses a loaded model to process a batch of images and generates a point cloud or GLB file.
            """
            def __init__(self):
                pass

            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "model": ("MODEL",),
                        "images": ("IMAGE",),
                        "show_cam":  ("BOOL", {"options": [True, False]}),
                        "mask_sky":  ("BOOL", {"options": [True, False]}),
                        "conf_thres": ("FLOAT",),
                        "prediction_mode": ("STRING", {"options": ["Depthmap", "Pointmap Regression"]}),
                        "model_name_test": (["Depthmap", "Pointmap Regression"],),
                    },
                }

            RETURN_TYPES = ("POINT_CLOUD_DATA", "GLB_FILE")
            FUNCTION = "generate_output"
            CATEGORY = "peakfiction/custom"

            def generate_output(self, model, images, show_cam, mask_sky, conf_thres, prediction_mode,model_name_test):
                try:
                    # Ensure images are in the correct format
                    if not isinstance(images, torch.Tensor):
                        images = torch.from_numpy(images).float().unsqueeze(0) / 255.0

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = model.to(device)

                    # Ensure images are provided
                    if len(images) == 0:
                        raise ValueError('Error: No images provided for processing.')

                    # Convert images to PIL format
                    images = [Image.fromarray(image.astype(np.uint8)) if isinstance(image, np.ndarray) else image for image in images]

                    # Preprocess images for the model
                    images_formatted = load_and_preprocess_images(images)
                    print(f"Preprocessed images shape: {images_formatted.shape}")

                    # Run inference
                    print("Running inference...")
                    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=dtype):
                            predictions = model(images)

                    # Convert pose encoding to extrinsic and intrinsic matrices
                    print("Converting pose encoding to extrinsic and intrinsic matrices...")
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
                    predictions["extrinsic"] = extrinsic
                    predictions["intrinsic"] = intrinsic

                    # Convert tensors to numpy
                    for key in predictions.keys():
                        if isinstance(predictions[key], torch.Tensor):
                            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

                    # Generate world points from depth map
                    print("Computing world points from depth map...")
                    depth_map = predictions["depth"]  # (S, H, W, 1)
                    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
                    predictions["world_points_from_depth"] = world_points

                    # Clean space
                    torch.cuda.empty_cache()

                    # Generate point cloud
                    point_cloud_data = self._generate_point_cloud(predictions, show_cam, mask_sky, conf_thres, prediction_mode)

                    # Create a GLB file
                    folder_path = os.path.join(os.path.dirname(__file__), "../../temp/glbs")
                    # Create the directory if it does not exist
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    
                    # Generate a random file name
                    file_name = f"output_{np.random.randint(1000, 9999)}.glb"
                    file_path = os.path.join(folder_path, file_name)
                    
                    # Placeholder logic for saving GLB file
                    try:
                        # Assuming `predictions` contains the necessary data for GLB generation
                        glb_data = self._generate_glb_file(predictions)
                        
                        # Save the GLB data to the file
                        with open(file_path, "wb") as glb_file:
                            glb_file.write(glb_data)
                        
                        print(f"GLB file successfully saved at: {file_path}")
                        return point_cloud_data, file_path
                    except Exception as e:
                        logging.error(f"Failed to save GLB file: {str(e)}")
                        raise RuntimeError("Error occurred while saving GLB file.")

                except Exception as e:
                    logging.error(f"An error occurred while generating output: {str(e)}")
                    raise RuntimeError("Failed to generate point cloud or GLB file.")

            def _generate_point_cloud(self, predictions, show_cam, mask_sky, conf_thres, prediction_mode):
                # Convert predictions to GLB
                # prediction mode is either Depthmap or Pointmap Regression
                target_dir = os.path.join(os.path.dirname(__file__), "../../temp/point_clouds")
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                glbscene = predictions_to_glb(
                    predictions,
                    conf_thres=conf_thres,
                    filter_by_frames="All",
                    mask_black_bg=False,
                    mask_white_bg=False,
                    show_cam=show_cam,
                    mask_sky=mask_sky,
                    target_dir=target_dir,
                    prediction_mode=prediction_mode,
                )
                glbfile = os.path.join(target_dir, f"point_cloud_scene{np.random.randint(0,10000000)}.glb")
                glbscene.export(file_obj=glbfile)
                return "POINT_CLOUD_DATA"

            def _generate_glb_file(self, output):
                # Placeholder for GLB file generation logic
                # Replace with actual implementation
                return "GLB_FILE_DATA"
