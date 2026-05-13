from pytorch3d.renderer import (MeshRasterizer, MeshRenderer,SoftSilhouetteShader,
                                RasterizationSettings, HardFlatShader, HardPhongShader, SoftPhongShader,
                                PerspectiveCameras, PointLights, AmbientLights)
from pose_reconstruction.src.utils.data_loader import conversion_opencv_pytorch
import torch
import pytorch3d.utils as py3dutils

import numpy as np



class SingleViewRenderer():

    def __init__(self, image_size, device):


        # Define Render settings
        self.faces_per_pixel = 90
        self.sigma = 1e-4

        self.device = device

        # Tuple with width and height
        self.image_size = image_size
        self.image_size_as_tuple = (int(self.image_size[0]), int(self.image_size[1]))
        #
        self.lights = AmbientLights(device=self.device)



    def get_camera(self, focal_lengths):

        self.K[0, 0, 0] = focal_lengths[0, 0]
        self.K[0, 1, 1] = focal_lengths[0, 1]


        # Retype the image_size correctly and flip to width, height.
        image_size_wh = self.image_size_tensor.flip(dims=(1,))

        # Screen to NDC conversion:
        # For non square images, we scale the points such that smallest side
        # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
        # This convention is consistent with the PyTorch3D renderer, as well as
        # the transformation function `get_ndc_to_screen_transform`.
        scale = image_size_wh.to(self.device).min(dim=1, keepdim=True)[0] / 2.0
        scale = scale.expand(-1, 2)
        #c0 = image_size_wh / 2.0

        # Get the PyTorch3D focal length and principal point.
        focal_pytorch3d = focal_lengths / scale
        #p0_pytorch3d = -(principal_point - c0) / scale

        self.camera.focal_length = focal_pytorch3d

        return self.camera

    def get_camera(self, img_size, focal_lengths):

        self.K[0, 0, 0] = focal_lengths[0, 0]
        self.K[0, 1, 1] = focal_lengths[0, 1]

        # image size is not necessary will be taken from the camera itself
        # Ndarray (2,) (width, height)
        #self.image_size_tensor = image_size
        # Tuple with width and height
        #self.image_size_as_tuple = image_size

        # R (3, 3) ndarray
        # K (3, 3) ndarray
        # t (3, 1) ndarray


        ### Create parameters to create the camera
        K = np.eye(3)

        K[0, 0] = focal_lengths[0, 0]
        K[1, 1] = focal_lengths[0, 1]

        # Add the principal point
        K[0, 2] = 0
        K[1, 2] = 0

        R = np.eye(3)

        # Do not put 0 as this will lead to a camera that probably does not get world coordinates back.
        # No Rendering possible., t needs to be >0?
        t = np.array([0, 0, 10])
        d = np.zeros(5)


        self.camera = conversion_opencv_pytorch(R=R, t=t, K=K,
                                                img_size=img_size, device=self.device)

        return self.camera

    def get_renderers(self, camera):

        sil_renderer = self.create_sil_renderer(camera)
        disp_renderer = self.create_display_renderer(camera)

        return sil_renderer, disp_renderer

    def create_sil_renderer(self, camera):

        raster_settings_silhouette = RasterizationSettings(
            image_size=self.image_size_as_tuple,
            blur_radius=np.log(1. / 1e-4 - 1.) * self.sigma,
            faces_per_pixel=self.faces_per_pixel,
            perspective_correct=True,
            bin_size=0
        )

        # Create a mesh rasterizer
        mesh_rasterizer = MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings_silhouette
        )

        # Create a shader
        shader = SoftSilhouetteShader()

        # Silhouette renderer
        renderer_silhouette = MeshRenderer(
            rasterizer=mesh_rasterizer,
            shader=shader
        )

        return renderer_silhouette

    def create_display_renderer(self, camera, lights=None):
        # ------------- Displaying Shader

        raster_settings_soft = RasterizationSettings(
            image_size=self.image_size_as_tuple,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            bin_size=0  # This changed drastically the mesh silhouette
        )

        # Create a mesh rasterizer
        mesh_rasterizer_soft = MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings_soft
        )

        if lights is None:
            lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])

        display_shader = SoftPhongShader(
            device=self.device,
            cameras=camera,
            lights=lights
        )

        renderer_mesh_display = MeshRenderer(rasterizer=mesh_rasterizer_soft, shader=display_shader)

        return renderer_mesh_display

def create_silhouette_renderer(camera_example, lights, device, image_size = (2056, 2464), sigma=1e-4, faces_per_pixel=40,
                               bin_size=None, max_faces_per_bin=100000, display_img_size=(2056, 2464), body_part_seg=False):
    """
    Renderer for silhouettes in the optimization
    :param camera_example:
    :param image_size:
    :param sigma:
    :param faces_per_pixel:
    :param bin_size:
    :param max_faces_per_bin:
    :return:
    """


    # Bin size = 0, increases time for optimization drastically
    # more faces per pixel increases gpu memory

    # Rasterization settings for silhouette rendering
    raster_settings_silhouette = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=faces_per_pixel,
        perspective_correct=True,
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin
    )

    raster_settings_silhouette = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=faces_per_pixel,
        perspective_correct=True,
        bin_size=0
    )

    # Create a mesh rasterizer
    mesh_rasterizer = MeshRasterizer(
        cameras=camera_example,
        raster_settings=raster_settings_silhouette
    )

    # Create a shader
    shader = SoftSilhouetteShader()

    # Silhouette renderer
    renderer_silhouette = MeshRenderer(
        rasterizer=mesh_rasterizer,
        shader=shader
    )

    if body_part_seg:
        raster_settings_silhouette = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=faces_per_pixel,
            perspective_correct=True,
            bin_size=0
        )

        # Create a mesh rasterizer
        mesh_rasterizer = MeshRasterizer(
            cameras=camera_example,
            raster_settings=raster_settings_silhouette
        )

        # # Create a shader
        # shader = HardPhongShader(device=device,
        #                          cameras=camera_example,
        #                          lights=lights
        #                          )
        shader = SoftPhongShader(device=device,
                                 cameras=camera_example,
                                 lights=lights
                                 )

        # Silhouette renderer
        renderer_silhouette = MeshRenderer(
            rasterizer=mesh_rasterizer,
            shader=shader
        )

    #
    renderer_mesh_display = get_rgb_renderer(camera_example, lights, display_img_size, faces_per_pixel=1)

    return renderer_silhouette, renderer_mesh_display


def get_rgb_renderer(cameras, lights, display_img_size, blur_radius=0, faces_per_pixel=1):
    """
    Create a renderer in pytorch3d for RGBA images
    :param cameras:
    :param lights:
    :param display_img_size:
    :param blur_radius:
    :param faces_per_pixel:
    :return:
    """

    raster_settings_soft = RasterizationSettings(
        image_size=display_img_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        perspective_correct=True,
        bin_size=0  # This changed drastically the mesh silhouette
    )

    # Create a mesh rasterizer
    mesh_rasterizer_soft = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings_soft
    )

    display_shader = SoftPhongShader(
        device=cameras.device,
        cameras=cameras,
        lights=lights
    )

    renderer_mesh_display = MeshRenderer(rasterizer=mesh_rasterizer_soft, shader=display_shader)


    return renderer_mesh_display


def sil_renderer_opt_loop(camera_example, image_size = (2056, 2464), sigma=1e-4, faces_per_pixel=90):
    """
    Renderer for silhouettes in the optimization
    :param camera_example:
    :param image_size:
    :param sigma:
    :param faces_per_pixel:
    :param bin_size:
    :param max_faces_per_bin:
    :return:
    """


    # Bin size = 0, increases time for optimization drastically
    # more faces per pixel increases gpu memory


    raster_settings_silhouette = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=faces_per_pixel,
        perspective_correct=True,
        bin_size=0
    )

    # Create a mesh rasterizer
    mesh_rasterizer = MeshRasterizer(
        cameras=camera_example,
        raster_settings=raster_settings_silhouette
    )

    # Create a shader
    shader = SoftSilhouetteShader()

    # Silhouette renderer
    renderer_silhouette = MeshRenderer(
        rasterizer=mesh_rasterizer,
        shader=shader
    )

    return renderer_silhouette


