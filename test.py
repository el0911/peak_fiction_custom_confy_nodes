# test.py
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from utils.extract_mask_with_scrible_map import ExtractMaskFromScribbleMap
from nodes.custom_nodes import Extract_mask_with_scrible_map

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
        try:
            # Get all image files
            image_files = list(self.test_assets_dir.glob("*.png")) + \
                         list(self.test_assets_dir.glob("*.jpg")) + \
                         list(self.test_assets_dir.glob("*.jpeg"))
            
            logger.info(f"Found {len(image_files)} images in test assets directory")
            
            # Group images by base name (assuming pairs are named similarly)
            image_pairs = {}
            for img_path in image_files:
                base_name = img_path.stem.split('_')[0]  # Assuming names like "base_original.png" and "base_scribble.png"
                if base_name not in image_pairs:
                    image_pairs[base_name] = []
                image_pairs[base_name].append(img_path)
            
            # Create node instance
            node = Extract_mask_with_scrible_map()
            
            # Process each pair
            for base_name, paths in image_pairs.items():
                if len(paths) != 2:
                    logger.warning(f"Skipping {base_name}: Expected 2 images, found {len(paths)}")
                    continue
                
                logger.info(f"\nTesting image pair: {base_name}")
                
                try:
                    # Load images
                    original_img = self.load_image(paths[0])
                    scribble_img = self.load_image(paths[1])
                    
                    # Process through node
                    logger.info("Processing images through node...")
                    mask_tuple = node.get_map(original_img, scribble_img)
                    
                    if not isinstance(mask_tuple, tuple):
                        raise ValueError("Node output should be a tuple")
                    
                    mask = mask_tuple[0]
                    
                    # Validate output
                    if self.validate_mask(mask):
                        logger.info("Mask validation passed")
                        self.save_output(mask, base_name)
                    else:
                        logger.error("Mask validation failed")
                    
                except Exception as e:
                    logger.error(f"Error processing {base_name}: {str(e)}")
                    continue
                
        except Exception as e:
            logger.error(f"Test execution failed: {str(e)}")

def main():
    """Main function to run tests."""
    logger.info("Starting node tests...")
    tester = NodeTester()
    tester.run_tests()
    logger.info("Testing completed!")

if __name__ == "__main__":
    main()