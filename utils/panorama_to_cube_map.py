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
        aspect = res_w / res_h
        fov_h_deg = 90.0
        fov_v_deg = 2 * math.degrees(
            math.atan(math.tan(math.radians(fov_h_deg) / 2) / aspect)
        )
        out_hw = (res_h, res_w)  # Use provided resolution for output image size
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
            face = py360convert.e2p(panorama, fov_h_deg=fov_h_deg, fov_v_deg=fov_v_deg, u_deg=u_deg, v_deg=v_deg, out_hw=out_hw, mode='bilinear')
            cubemap.append(face)
        
        # Return cubemap as a list of images
        return cubemap
