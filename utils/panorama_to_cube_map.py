import py360convert
import cv2

# class to convert panorama to cubemap
class PanoramaToCubemap:
    @staticmethod
    def panorama_to_cubemap(panorama, cube_size=512):
        """
        Converts a panorama image to a cubemap format.
        Args:
            panorama: Input panorama image.
            cube_size: Size of each face of the cubemap.
        Returns:
            A list of images representing the six faces of the cubemap.
        """
        # Ensure the panorama is in RGB format
        if len(panorama.shape) == 2 or panorama.shape[2] == 1:
            panorama = cv2.cvtColor(panorama, cv2.COLOR_GRAY2BGR)

        # Convert panorama to cubemap using py360convert
        cubemap = py360convert.e2c(panorama,face_w=cube_size, mode='bilinear',cube_format='dice')

        # return cubemap as a list of images
        
        return cubemap
