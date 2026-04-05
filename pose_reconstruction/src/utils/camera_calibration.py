import pickle

import scipy.io as sio
import os
import pose_reconstruction.src.utils.utils_camera as utils_camera
import matplotlib.pyplot as plt
import cv2
from pytorch3d.renderer import PerspectiveCameras

import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pytorch3d.common.datatypes import Device
import toml
import numpy as np
from pytorch3d.renderer.cameras import PerspectiveCameras




def change_focal_length_pytorch3d(cam: PerspectiveCameras, focal_lengths):
    """
    Change the focal length of a pytorch3d camera from usual opencv pixels to noramlized focal_lengths.
    :param cam:
    :param focal_lengths:
    :return:
    """

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = cam.image_size.flip(dims=(1,))

    # Screen to NDC conversion:
    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer, as well as
    # the transformation function `get_ndc_to_screen_transform`.
    scale = image_size_wh.min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    # c0 = image_size_wh / 2.0

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = focal_lengths / scale
    # p0_pytorch3d = -(principal_point - c0) / scale

    cam.focal_length = focal_pytorch3d


    return cam


class Cameras:

    def __init__(self, calib_path=None):
        
        # Cameras dictionary
        self.cam_dict = {}
        # Calibration directory
        self.calib_path = calib_path

    def read_cam_params(self, alter_path=None, extension="sys_calib"):
        """
        Reads back camera parameters that were exported to matlab via
        save_camera_parameters_to_mat()
        :param base_path:
        :return:
        """
    
        if alter_path is not None:
            obj_array = sio.loadmat(os.path.join(alter_path, extension + '.mat'))
        else:
            try:
                obj_array = sio.loadmat(os.path.join(self.calib_path, extension + '.mat'))
            except OSError:
                print("There was no system calibration file in ", self.calib_path)
                try:
                    print("Try loading a recalibration file")
                    obj_array = sio.loadmat(os.path.join(self.calib_path, 'recalibration.mat'))
                except OSError:
                    raise ValueError("No calibration files found in the specified path")
    
        obj_array = obj_array["Cameras"][0]
    
        for i in range(obj_array.__len__()):
            cam = Camera()
            name, cam.im_size, cam.K, cam.d, cam.R, cam.t, cam.calib_error = obj_array[i][0]
            cam.d = cam.d.squeeze()
            cam.im_size = cam.im_size[0]
            self.cam_dict[name[0]] = cam
    
        # Sort alphabetically
        self.cam_dict = dict(sorted(self.cam_dict.items()))
        print("Parameters loaded.")

    def read_camera_from_pkl_mammal(self, path_to_pkl):


        with open(path_to_pkl, 'rb') as f:
            camera_pkl = pickle.load(f)


        for cam_id, camera_mammal in enumerate(camera_pkl):


            cam = Camera()

            cam.K = camera_mammal["K"]
            cam.R = camera_mammal["R"]
            cam.t = np.expand_dims(camera_mammal["T"], axis=-1)

            cam.im_size = camera_mammal['mapx'].shape
            self.cam_dict[f"{cam_id}"] = cam


        # K(3,3) ndarray
        # R the same
        # t(3, 1)

        pass



    def read_cam_params_from_toml(self, base_path):


        # Change if json Camera Parameter

        cam_file = toml.load(os.path.join(base_path, "calibration.toml"))

        for index, cam_key in enumerate(dict.keys(cam_file)):
            if cam_key != 'metadata':
                cam = Camera()
                name = cam_file[cam_key]['name']
                cam.K = np.array(cam_file[cam_key]['matrix'])
                cam.d = np.array(cam_file[cam_key]['distortions'])
                cam.r = np.array(cam_file[cam_key]['rotation'])
                cam.R = cv2.Rodrigues(cam.r)[0]
                # expand_dims so it is 3x1
                cam.t = np.expand_dims(np.array(cam_file[cam_key]['translation']), axis=1)
                cam.im_size = np.array(cam_file[cam_key]['size'])
                self.cam_dict[name] = cam

        # Sort alphabetically
        self.cam_dict = dict(sorted(self.cam_dict.items()))


    def add_camera_to_dict(self, name, K, d, R, t, im_size):
        """
        OpenCV compatible camera values
        :param name:
        :param K:
        :param d:
        :param R:
        :param t: np.array(3,1)
        :param im_size: np.array(2,)
        :return:
        """

        cam = Camera()
        cam.K = K
        cam.d = d
        cam.R = R
        cam.r = cv2.Rodrigues(cam.R)[0]
        # expand_dims so it is 3x1
        cam.t = t
        cam.im_size = im_size
        self.cam_dict[name] = cam



    def show(self, lim = 2000, dim_mm = True, cam_scale = 1, object_points=None, plot_boundaries=False):

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')


        ts = np.ones((self.cam_dict.__len__(), 3, 1))
        index = 0

        for key, cam in self.cam_dict.items():
            t, r = (None, None)

            if cam.R is not None:
                t, r = cam.t, cam.R
            else:
                t, r = cam.t, cam.get_rot_mat()[0]



            vertices_cam = utils_camera.get_camera_vertices(asp=(cam.im_size[1]/cam.im_size[0]))
            vertices_cam *= cam_scale

            if dim_mm:
                vertices_cam *= 10

            r = r.T
            t = -r@t

            ts[index, :, :] = t
            index += 1

            vertices_cam = (r@vertices_cam.T).T
            vertices_cam += t.T

            ax.plot(vertices_cam[:, 0], vertices_cam[:, 1], vertices_cam[:, 2], color='k')
            ax.text(t[0][0], t[1][0], t[2][0], f"{key}", color='red')

        ax.scatter(0, 0, 0, color = 'r')

        if object_points is not None:
            # Object points as array of (N, 3)
            ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], color = 'b')

        if plot_boundaries:

            # Find the min and max for each dimension
            min_x, min_y, min_z = np.min(ts[:, 0]), np.min(ts[:, 1]), np.min(ts[:, 2])
            max_x, max_y, max_z = np.max(ts[:, 0]), np.max(ts[:, 1]), np.max(ts[:, 2])

            vertices = np.array([
                [min_x, min_y, min_z],
                [min_x, min_y, max_z],
                [min_x, max_y, min_z],
                [min_x, max_y, max_z],
                [max_x, min_y, min_z],
                [max_x, min_y, max_z],
                [max_x, max_y, min_z],
                [max_x, max_y, max_z]
            ])

            for vertex in vertices:
                ax.scatter(vertex[0], vertex[1], vertex[2], color = 'g')
            print("Boundaries (min) (x-y-z): ", np.min(ts[:, 0]), np.min(ts[:, 1]), np.min(ts[:, 2]))
            print("Boundaries (max) (x-y-z): ", np.max(ts[:, 0]), np.max(ts[:, 1]), np.max(ts[:, 2]))

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        #plt.show()

        if plot_boundaries:
            return vertices


    def get_camera_boundaries(self):

        ts = np.ones((self.cam_dict.__len__(), 3, 1))
        index = 0

        for key, cam in self.cam_dict.items():

            t, r = (None, None)

            if cam.R is not None:
                t, r = cam.t, cam.R
            else:
                t, r = cam.t, cam.get_rot_mat()[0]

            r = r.T
            t = -r @ t

            ts[index, :, :] = t
            index += 1

        # Find the min and max for each dimension
        min_x, min_y, min_z = np.min(ts[:, 0]), np.min(ts[:, 1]), np.min(ts[:, 2])
        max_x, max_y, max_z = np.max(ts[:, 0]), np.max(ts[:, 1]), np.max(ts[:, 2])
        vertices = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z]
        ])

        return vertices

    def get_camera_in_world(self, cam_key):

        for key, cam in self.cam_dict.items():

            if key == cam_key:
                if cam.R is not None:
                    t, r = cam.t, cam.R
                else:
                    t, r = cam.t, cam.get_rot_mat()[0]

                r = r.T
                t = -r @ t

            else:
                continue

        return t


    def lift_marker(self, vis_labels, undist=True, ransac=False, orig_labels=None, min_nb_triang=2, max_nb_triang=0):
        """

        :param vis_labels:
        :param undist:
        :param ransac:
        :param orig_labels:
        :return:
        X - 3d position as ndarray (3,)

        """

        points = {}
        cam_Ms = {}


        for index, name in enumerate(vis_labels.keys()):
            coords = vis_labels[name]
            cam = self.cam_dict[name]

            # Takes already cam K into account
            if undist and ransac is False:
                coords = cv2.undistortPoints(coords.copy(), cameraMatrix=cam.K, distCoeffs=cam.d)[0, 0, :]

            points[name] = coords

            M = np.vstack((np.hstack((cam.R, cam.t)), np.array([0, 0, 0, 1])))
            cam_Ms[name] = M
            #cam_Ms.append(M)

        # Order should be preserved in both dictionaries

        if ransac:
            # Best 3d coordinates, index in triang_dic of best 3d coords, and triang dict
            X, best_index, triang_dic = self.ransac_triangulation(points, cam_Ms, orig_labels, min_nb_triang=min_nb_triang, max_nb_triang=max_nb_triang)
            # return best 3d, return its mean reprojection, return the set of the best triang, and triang itself
            return X, np.array(list(triang_dic[best_index][2].values())).mean(), triang_dic[best_index][0], best_index,\
                   triang_dic

        else:
            # approx 35us for one triangulation
            X = utils_camera.triangulate_simple(list(points.values()), list(cam_Ms.values()))

            er_list = []
            for cam_name in vis_labels.keys():
                cam = self.cam_dict[cam_name]
                rvec = cv2.Rodrigues(cam.R, None, None)[0]
                x_2d_rep = np.squeeze(cv2.projectPoints(X, rvec, cam.t, cam.K, cam.d)[0], axis=1)

                if not self.disturb:
                    e = utils_camera.rep_error(x_2d_rep, vis_labels[cam_name])
                else:
                    e = utils_camera.rep_error(x_2d_rep, orig_labels[cam_name])

                er_list.append(e)

            return X, np.array(er_list).mean(), False, False, False


    def ransac_triangulation(self, dist_points, Ms, orig_points=None, min_nb_triang=2, max_nb_triang=0):




        # Get all possible triangulation sets
        sets = utils_camera.powerset(dist_points, min_nb_triangulation=min_nb_triang, max_set=max_nb_triang)

        triang_dict = {}

        #X = -10000 * np.ones(3)

        for index, set in enumerate(sets):

            point_list = []
            M_list = []

            for cam_name in set:
                cam = self.cam_dict[cam_name]

                undist_point = cv2.undistortPoints(dist_points[cam_name].copy(),
                                                   cameraMatrix=cam.K, distCoeffs=cam.d)[0, 0, :]
                point_list.append(undist_point)
                M_list.append(Ms[cam_name])

            X = utils_camera.triangulate_simple(point_list, M_list)

            er_list = {}

            # Calculate reprojections and errors for all cameras!!!
            for cam_name in dist_points.keys():

                cam = self.cam_dict[cam_name]
                rvec = cv2.Rodrigues(cam.R, None, None)[0]
                x_2d_rep = np.squeeze(cv2.projectPoints(X, rvec, cam.t, cam.K, cam.d)[0], axis=1)


                if orig_points is not None:
                    e = utils_camera.rep_error(x_2d_rep, orig_points[cam_name])
                else:
                    e = utils_camera.rep_error(x_2d_rep, dist_points[cam_name])

                er_list[cam_name] = e

            triang_dict[index] = [set, X, er_list]

        # Get best mean reprojection error
        best_ = X
        best_index = 0
        # Default error
        prev_error = 1000

        for key, value in triang_dict.items():

            er_s = list(value[2].values())
            mean = np.array(er_s).mean()

            if mean < prev_error:
                best_ = value[1]
                best_index = key
                prev_error = mean

        return best_, best_index, triang_dict

class Camera:
    def __init__(self):

        self.name = None
        self.K = None
        self.d = None
        self.r = None
        self.t = None
        self.R = None

        self.calib_error = None
        self.im_size = None
    
    def get_rot_mat(self):
        return cv2.Rodrigues(self.r)


# Try first to create a new class, where we add the distortion function as differentiable addon on top


# An input which is a float per batch element
_BatchFloatType = Union[float, Sequence[float], torch.Tensor]

# one or two floats per batch element
_FocalLengthType = Union[
    float, Sequence[Tuple[float]], Sequence[Tuple[float, float]], torch.Tensor
]

# Default values for rotation and translation matrices.
_R = torch.eye(3)[None]  # (1, 3, 3)
_T = torch.zeros(1, 3)  # (1, 3)
_d = torch.zeros(1, 5)  # (1, 5)
_K = torch.eye(3)[None]  # (1, 3, 3)

class PerspectiveCamerasWithDistortion(PerspectiveCameras):
    """
        A class which stores a batch of parameters to generate a batch of
        transformation matrices using the multi-view geometry convention for
        perspective camera.

        Parameters for this camera are specified in NDC if `in_ndc` is set to True.
        If parameters are specified in screen space, `in_ndc` must be set to False.
        """

    # For __getitem__
    _FIELDS = (
        "K",
        "R",
        "T",
        "focal_length",
        "principal_point",
        "_in_ndc",  # arg is in_ndc but attribute set as _in_ndc
        "image_size",
        "distortion"
    )

    _SHARED_FIELDS = ("_in_ndc",)

    def __init__(
            self,
            focal_length: _FocalLengthType = 1.0,
            principal_point=((0.0, 0.0),),
            R: torch.Tensor = _R,
            T: torch.Tensor = _T,
            K: torch.Tensor = _K,
            d: torch.Tensor = _d,
            device: Device = "cpu",
            in_ndc: bool = True,
            image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    ) -> None:
        """

        Args:
            focal_length: Focal length of the camera in world units.
                A tensor of shape (N, 1) or (N, 2) for
                square and non-square pixels respectively.
            principal_point: xy coordinates of the center of
                the principal point of the camera in pixels.
                A tensor of shape (N, 2).
            in_ndc: True if camera parameters are specified in NDC.
                If camera parameters are in screen space, it must
                be set to False.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            K: (optional) A calibration matrix of shape (N, 4, 4)
                If provided, don't need focal_length, principal_point
            image_size: (height, width) of image size.
                A tensor of shape (N, 2) or a list/tuple. Required for screen cameras.
            device: torch.device or string
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        kwargs = {"image_size": image_size} if image_size is not None else {}
        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            K=K,
            _in_ndc=in_ndc,
            **kwargs,  # pyre-ignore
        )
        if image_size is not None:
            if (self.image_size < 1).any():  # pyre-ignore
                raise ValueError("Image_size provided has invalid values")
        else:
            self.image_size = None

        # When focal length is provided as one value, expand to
        # create (N, 2) shape tensor
        if self.focal_length.ndim == 1:  # (N,)
            self.focal_length = self.focal_length[:, None]  # (N, 1)
        self.focal_length = self.focal_length.expand(-1, 2)  # (N, 2)

        # Put the distortion parameters into this child
        if d is not None:
            self.d = d


    def transform_points(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """

        :param points:
        :param eps:
        :param kwargs:
        :return:
        """


        world_to_proj_transform = self.get_full_projection_transform(**kwargs)

        points_camera_3d = world_to_proj_transform.transform_points(points, eps=eps)

        # Apply differentiable distortion,
        points_camera_3d_distorted = points_camera_3d

        return points_camera_3d_distorted



class OpenCVDistortion(torch.autograd.Function):
    """
    Applies radial and tangential distortion in a differentiable way.
    """

    @staticmethod
    def forward(ctx, input, distortion_parameters):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        ctx.save_for_backward(distortion_parameters)

        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)
