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
        # # Ensure the panorama is in RGB format
        # if len(panorama.shape) == 2 or panorama.shape[2] == 1:
        #     panorama = cv2.cvtColor(panorama, cv2.COLOR_GRAY2BGR)

        # Convert panorama to cubemap using py360convert
        overlap = 0.1  # Overlap percentage for cubemap faces
        res_w, res_h = 1920, 1080
        fov_h_deg = 90.0 + (overlap * 90.0)  # Add overlap to horizontal FOV
        fov_v_deg = 90.0 + (overlap * 90.0)  # Add overlap to vertical FOV
        out_hw = (res_h, res_w)  # Output image size for each face
        cubemap = []
        angles = [
            (0, 0),  # Front
            (90, 0),  # Right
            (180, 0),  # Back
            (-90, 0),  # Left
            (0, 90),  # Top
            (0, -90)  # Bottom
        ]
       
        for u_deg, v_deg in angles:
            face = py360convert.e2p(
                panorama, 
                fov_deg=(fov_h_deg, fov_v_deg), 
                u_deg=u_deg, 
                v_deg=v_deg, 
                out_hw=out_hw, 
                mode='bilinear'
            )
            cubemap.append(face)
        
        # Return cubemap as a list of images
        return cubemap
