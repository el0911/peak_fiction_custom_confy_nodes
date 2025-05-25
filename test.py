# test.py
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from nodes.custom_nodes import Extract_mask_with_scrible_map
from nodes.custom_nodes import Cube_map_From_Panorama

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'test_outputs/test_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class NodeTester:
    def __init__(self):
        self.test_assets_dir = Path("test_assets")
        self.test_outputs_dir = Path("test_outputs")
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        self.test_outputs_dir.mkdir(exist_ok=True)
        logger.info(f"Ensuring output directory exists: {self.test_outputs_dir}")
        
    def load_image(self, image_path):
        """Load an image and handle potential errors."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

    def save_output(self, mask, original_name, suffix=""):
        """Save the output mask with timestamp."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"{original_name}_output_{suffix}_{timestamp}.png"
            output_path = self.test_outputs_dir / output_name
            cv2.imwrite(str(output_path), mask)
            logger.info(f"Saved output to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving output: {str(e)}")

    def save_mask(self,mask_tensor, save_path):
        from PIL import Image

        # Make sure tensor is on CPU and detached from computation graph
        mask_np = mask_tensor.detach().cpu().numpy()
        
        # If it's a 3D tensor (C,H,W), convert appropriately
        if len(mask_np.shape) == 3:
            if mask_np.shape[0] == 1:  # Single channel
                mask_np = mask_np[0]  # Take first channel
            else:  # Multi-channel - assuming it's in CHW format
                mask_np = np.transpose(mask_np, (1, 2, 0))  # Convert to HWC
        
        # Normalize to 0-255 range if needed
        if mask_np.max() <= 1.0:
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = mask_np.astype(np.uint8)
        
        # Create image from numpy array
        mask_img = Image.fromarray(mask_np)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{save_path}_output__{timestamp}.png"
        
        output_path = self.test_outputs_dir / output_name
           
        print(output_path)
        # Create directory if it doesn't exist
        # os.makedirs( output_path , exist_ok=True)
        
        # Save the image
        mask_img.save(output_path)
        print(f"Mask saved to {output_path}")
    def validate_mask(self, mask):
        """Validate that the output is a proper mask."""
        try:
            if mask is None:
                raise ValueError("Mask is None")
            
            if not isinstance(mask, np.ndarray):
                raise TypeError(f"Expected numpy array, got {type(mask)}")
            
            if mask.dtype != np.uint8:
                raise TypeError(f"Expected uint8 array, got {mask.dtype}")
            
            if len(mask.shape) != 2:
                raise ValueError(f"Expected 2D array, got shape {mask.shape}")
            
            if not np.all((mask == 0) | (mask == 255)):
                logger.warning("Mask contains values other than 0 and 255")
            
            return True
        except Exception as e:
            logger.error(f"Mask validation failed: {str(e)}")
            return False

    def run_tests(self):
        """Run tests on all image pairs in the test assets directory."""
        # try:
        #     # Get all image files
        #     image_files = list(self.test_assets_dir.glob("*.png")) + \
        #                  list(self.test_assets_dir.glob("*.jpg")) + \
        #                  list(self.test_assets_dir.glob("*.jpeg"))
            
        #     logger.info(f"Found {len(image_files)} images in test assets directory")
            
        #     # Group images by base name (assuming pairs are named similarly)
        #     image_pairs = {}
        #     for img_path in image_files:
        #         base_name = img_path.stem.split('_')[0]  # Assuming names like "base_original.png" and "base_scribble.png"
        #         if base_name not in image_pairs:
        #             image_pairs[base_name] = []
        #         image_pairs[base_name].append(img_path)
            
        #     # Create node instance
        #     node = Extract_mask_with_scrible_map()
            
        #     # Process each pair
        #     for base_name, paths in image_pairs.items():
        #         if len(paths) != 2:
        #             logger.warning(f"Skipping {base_name}: Expected 2 images, found {len(paths)}")
        #             continue
                
        #         logger.info(f"\nTesting image pair: {base_name}")
                
        #         try:
        #             # Load images
        #             original_img = self.load_image(paths[0])
        #             scribble_img = self.load_image(paths[1])
                    
        #             # Process through node
        #             logger.info("Processing images through node...")
        #             mask_tuple = node.get_map(original_img, scribble_img)
                    
        #             if not isinstance(mask_tuple, tuple):
        #                 raise ValueError("Node output should be a tuple")
                    
        #             mask = mask_tuple[0]
                    
        #             # Validate output
        #             if True:
        #                 logger.info("Mask validation passed")
        #                 self.save_mask(mask, base_name)
        #             else:
        #                 logger.error("Mask validation failed")
                    
        #         except Exception as e:
        #             logger.error(f"Error processing {base_name}: {str(e)}")
        #             continue
                
        # except Exception as e:
        #     logger.error(f"Test execution failed: {str(e)}")

        # run cubemap conversion test   
        try:
            panorama_path = self.test_assets_dir / "panorama_equirectangular.png"  # Example panorama image
            if not panorama_path.exists():
                logger.error(f"Panorama image not found: {panorama_path}")
                return
            
            logger.info(f"\nTesting panorama to cubemap conversion with {panorama_path.name}")
            
            # Load panorama image
            panorama_img = self.load_image(panorama_path)
            
            # Create node instance for cubemap conversion
            cubemap_node = Cube_map_From_Panorama()
            
            # Process through node
            logger.info("Converting panorama to cubemap...")
            cubemap_faces = cubemap_node.convert_to_cubemap(panorama_img)
            # convert rurple to a list
            new_cubemap_faces = []
           
            # if not isinstance(cubemap_faces, list) or len(cubemap_faces) != 6:
            #     raise ValueError("Cubemap conversion did not return 6 faces")
            
            # array of pytorvh tesnors
            # Save each face of the cubemap
            for i, face in enumerate(list(cubemap_faces)):
                face_name = f"cubemap_face_{i+1}"
                # save pytorch tensor
                print(face)
                print(type(face))
                # type is a tensor
                # save tensor image
                self.save_mask(face, face_name)

                # self.save_mask(face, face_name)
                
        except Exception as e:
            logger.error(f"Error during cubemap conversion test: {str(e)}")

def main():
    """Main function to run tests."""
    logger.info("Starting node tests...")
    tester = NodeTester()
    tester.run_tests()
    logger.info("Testing completed!")

if __name__ == "__main__":
    main()