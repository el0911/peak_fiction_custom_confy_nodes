import py360convert
import cv2

# class to convert panorama to cubemap
class PanoramaToCubemap:
    @staticmethod
    def panorama_to_cubemap(panorama, cube_size=512, overlap=0.5):
    h, w, _ = panorama.shape  # Get the dimensions of the input panorama
    aspect_ratio = 16 / 9  # Maintain a 16:9 aspect ratio for up and down shots
    res_w, res_h = 1920, 1080  # Set resolution for up and down shots
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
        # Adjust resolution for top and bottom faces
        if v_deg in [90, -90]:  # Top or Bottom
            out_hw = (res_h, res_w)
        else:
            out_hw = (cube_size, cube_size)
        
        face = py360convert.e2p(
            panorama, 
            fov_deg=(fov_h_deg, fov_v_deg), 
            u_deg=u_deg, 
            v_deg=v_deg, 
            out_hw=out_hw, 
            mode='bilinear'
        )
        cubemap.append(face)
        # save face as an image
    
    # Return cubemap as a list of images
    return cubemap
