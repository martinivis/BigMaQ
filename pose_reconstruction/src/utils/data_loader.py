
import pickle

import torch
from pytorch3d.renderer import PerspectiveCameras

from pose_reconstruction.src.utils.camera_calibration import Cameras
from pose_reconstruction.src.utils.utils_camera import return_nb_frames_video
import os
from os.path import join
import numpy as np
from skimage.transform import resize
import cv2
import json
from tqdm import tqdm
from tempfile import TemporaryFile
from pycocotools import mask as mask_pycoco
from itertools import groupby
from copy import deepcopy
import time
import shutil
import copy
from scipy.io import loadmat
import pose_reconstruction.src.utils.data_loader as dat_loader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import random
import pytorch3d.utils as py3dutils
from pose_reconstruction.src.utils.utils import check_create_path
import joblib
import pose_reconstruction.src.utils.utils as utils
from pathlib import Path



def conversion_opencv_pytorch(R, t, K, img_size, device):


    RR = torch.from_numpy(R).unsqueeze(0).to(device)  # dim = (1, 3, 3)
    tt = torch.from_numpy(t.squeeze()).unsqueeze(0).to(device)
    KK = torch.from_numpy(K).unsqueeze(0).to(device)
    img_size_torch = torch.from_numpy(img_size).unsqueeze(0).to(device)

    return py3dutils.cameras_from_opencv_projection(R=RR.float(), tvec=tt.float(), camera_matrix=KK.float(),
                                                    image_size=img_size_torch.float())


def longest_continuous_ones_indices(vector):
    """
    Find the indices of the longest continuous sequence of 1s in a vector using NumPy.

    Parameters:
        vector (list or np.ndarray): A vector containing 0s and 1s.

    Returns:
        list: Indices of the longest continuous sequence of 1s.
    """
    vector = np.array(vector)  # Ensure input is a NumPy array
    # Find where the values change (from 1 to 0 or 0 to 1)
    diff = np.diff(np.concatenate(([0], vector, [0])))
    starts = np.where(diff == 1)[0]  # Start indices of sequences of 1s
    ends = np.where(diff == -1)[0]  # End indices of sequences of 1s

    # Find the lengths of all sequences of 1s
    lengths = ends - starts
    if len(lengths) == 0:  # Handle edge case: no 1s in the vector
        return []

    # Find the longest sequence
    longest_idx = np.argmax(lengths)
    longest_start, longest_end = starts[longest_idx], ends[longest_idx]

    # Return the indices of the longest sequence
    return list(range(longest_start, longest_end))


def check_complete_rle(mask_root_path):

    masks_npy = np.load(mask_root_path, allow_pickle=True)[()]

    check_ok = True

    for cam, rle_dict in masks_npy.items():


        if type(rle_dict[0]) is np.ndarray:

        #if not isinstance(rle_dict[0], dict):
        #if not 'counts' in rle_dict[0][0].keys():


            rle_list = []

            for frame in range(rle_dict[0].shape[0]):
                rle_list.append(convert_mask_to_rle(rle_dict[0][frame, :, :]))

        #convert_mask_to_rle
        #    masks_npy[cam] =

        #    for frame in

            masks_npy[cam] = {0: rle_list}

            print(f"---------------------- EMPTY MASK CONVERSION for camera {cam} ----------------------")
            ## Convert the array into RLE format
    return masks_npy

def check_vertex_information(vertices_information, nb_vertices):

    flattened_pairs = [x for xs in vertices_information["pairs"] for x in xs]
    flattened_pairs.extend(vertices_information["no_partner"])

    # 3625 LowPoly
    # 10632 HighPoly

    for i in range(nb_vertices):
        if str(i) not in flattened_pairs:
            raise ValueError("Not all vertices are included")

    #print("Vertex information seems complete")


def load_paths_dict(file_name):
    """
    Loads a dictionary of path lists from a JSON file.

    Args:
        file_name (str): Name of the JSON file to load the data from.

    Returns:
        dict: Dictionary containing lists of paths categorized by keys.
    """
    with open(file_name, 'r') as file:
        return json.load(file)


def convert_paths_windows_to_linux(paths, hard_drive_locater):

    for path_idx, path in enumerate(paths):
        paths[path_idx] = convert_windows_path_to_linux(path, hard_drive_locater)
    return paths


def convert_windows_path_to_linux(path, hard_drive_locator):
    # Replace backslashes with forward slashes
    path = path.replace('\\', '/')

    # Handle the drive letter
    if ':' in path:
        # Handling an error in saving the json
        if path[path.index(':') + 1] == '/':
            parts = path.split(':')
            rest_of_path = parts[1]
            path = f'{hard_drive_locator}{rest_of_path}'
        else:
            parts = path.split(':')
            rest_of_path = parts[1]
            path = f'{hard_drive_locator}/{rest_of_path}'
    else:
        path = f'{hard_drive_locator}/{path}'

    return path


def remove_json_calib_paths(calib_paths):
    # Change the calib paths to only give the directory
    for calib_index, calib_path in enumerate(calib_paths):
        calib_paths[calib_index] = os.path.dirname(calib_path)

    return calib_paths

def load_paths_to_linux(save_loc = None):
    """
    Load action and calibration paths in appropriate format
    :param save_loc:
    :return:
    """
    #
    hard_drive_locator_change = None

    data_paths_json = load_paths_dict(save_loc)

    tr_paths = convert_paths_windows_to_linux(data_paths_json["tr_paths"], hard_drive_locator_change)
    tr_calib_paths = convert_paths_windows_to_linux(data_paths_json["tr_calib_paths"], hard_drive_locator_change)

    te_paths = convert_paths_windows_to_linux(data_paths_json["TEST_PATHS"], hard_drive_locator_change)
    te_calib_paths = convert_paths_windows_to_linux(data_paths_json["TEST_calib_paths"], hard_drive_locator_change)


    return (tr_paths, remove_json_calib_paths(tr_calib_paths), data_paths_json["tr_individuals"], te_paths,
            remove_json_calib_paths(te_calib_paths), data_paths_json["TEST_INDIVIDUALS"])

def convert_rle_to_mask(rle):
    return mask_pycoco.decode(mask_pycoco.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1]))


def convert_RLE_masks_to_array(masks):
    """

    :param masks: Dictionary over cameras, individuals, frame_indices (not actual frames in case of label data) in RLE
    :return: Masks as np binary arrays
    """
    for cam_key in masks.keys():
        for ind_key in masks[cam_key].keys():
            for frame_idx in range(len(masks[cam_key][ind_key])):

                masks[cam_key][ind_key][frame_idx] = convert_rle_to_mask(masks[cam_key][ind_key][frame_idx])

    return masks

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def convert_mask_to_rle(mask):
    return binary_mask_to_rle(np.asfortranarray(mask))


def delete_with_check(path):

    if os.path.exists(path):
        shutil.rmtree(path)



def get_cropping_info(oms_base_path, batch_id, frm):

    crop_data = loadmat(join(oms_base_path,
                        'Batch{}/crop_para.mat'.format(batch_id)))['crop']

    pt = crop_data.transpose()[0]


    u = np.unique(pt, axis=0)
    q = np.where(pt == u[frm])

    crop_data_sel = crop_data[q[0]]

    return crop_data_sel



def get_vertex_info(symm_vertices_path):
    # Get the vertex info
    if symm_vertices_path:
        ## Load the symmetry of the mesh
        json_file = os.path.join(symm_vertices_path)
        with open(json_file, 'r') as infile:
            vertices_information = json.load(infile)
        check_vertex_information(vertices_information)
    else:
        vertices_information = None

    return vertices_information

def load_json_annotations(path_to_json):
    with open(path_to_json, 'r') as infile:
        json_data = json.load(infile)
    return json_data


def create_example_cameras(image_size, device):
    """
    Creates a camera with initial focal length and passes the image dimensions, to create a camera per image
    :param image_size:
    :param device:
    :return:
    """
    ## Built the cameras
    cams_bigmac = Cameras(calib_path=None)

    cam_key = '1'

    K = np.eye(3)



    focal_length_start = max(image_size) // 2

    # Put some preliminary focal length
    K[0, 0] = focal_length_start#200
    K[1, 1] = focal_length_start#200


    # Set the optical center into the middle of the image!
    K[0, 2] = image_size[1] // 2
    K[1, 2] = image_size[0] // 2


    R = np.eye(3)

    R[0, 0] = -1
    R[1, 1] = -1

    cam_world = np.array([0, 0, -1]) # np.array([-3, -3, -4])
    t = -R.T @ cam_world
    d = np.zeros(5)

    cams_bigmac.add_camera_to_dict(name=cam_key, K=K, d=d, R=R, t=np.expand_dims(t, axis=-1)
                                   , im_size=image_size)
    cam = cams_bigmac.cam_dict[cam_key]

    cam_pytorch = {}
    cam_pytorch[cam_key] = conversion_opencv_pytorch(R, t, K, image_size, device)

    return cam_pytorch, cams_bigmac

def generate_overlapping_batches(first_frame, last_frame, batch_size, overlap):
    """
    Generate overlapping frame batches between [first_frame, last_frame),
    where last_frame is exclusive (e.g., Python slice style).

    :param first_frame: int, the first frame's index (inclusive), 0-based.
    :param last_frame: int, the last frame's index (exclusive).
    :param batch_size: int, number of frames per batch.
    :param overlap: int, number of overlapping frames between consecutive batches.
    :return: list of lists, where each sub-list is a batch of frame indices.
    """

    stride = batch_size - overlap  # how many new frames we move for each new batch
    batches = []

    start_index = first_frame
    while start_index < last_frame:
        end_index = start_index + batch_size
        # Slice out the frames for this batch
        batch = list(range(start_index, min(end_index, last_frame)))
        batches.append(batch)

        # If the end_index passes beyond last_frame, we're done
        if end_index >= last_frame:
            break

        # Move to the next start by stride
        start_index += stride

    return batches

def split_into_minibatches(list_of_lists, minibatch_size):
    # Result to store all minibatch groupings
    result = []

    for sublist in list_of_lists:
        # Split the current sublist into chunks of size minibatch_size
        minibatches = [sublist[i:i + minibatch_size] for i in range(0, len(sublist), minibatch_size)]
        result.append(minibatches)

    return result

from pose_reconstruction.src.utils.CrossViewMatching import MultiEstimator

class ActionLoader():
    """
    Returns cropped labels of actions
    """
    def __init__(self, path_to_label_split, path_to_dataset, vertex_symm_path, device='cuda', labels_only=False,
                 max_image_size=None, debug=False, hard_drive_loc=None, hard_drive_loc_fast=None, load_undist_rgb=True,
                 high_poly=False, same_size=True, force_action_reload=False, dummy=False, project_root=None):

        # Path to labeled actions
        self.label_split = path_to_label_split
        self.labeled_actions = []
        self.unique_individuals = []

        # For re-creating an already saved label file for over time prediction
        self.force_action_reload = force_action_reload

        # Entire Dataset specifics
        self.dataset_overview_path = path_to_dataset
        self.dataset = []

        self.hard_drive_loc = hard_drive_loc
        self.hard_drive_loc_fast = self.hard_drive_loc

        self.project_root = project_root

        self.unique_individuals = ['C', 'G', 'H', 'J', 'L', 'N', 'O', 'T']

        # Dictionary over individuals, list of lists, 1st (single), 2nd (double actions), 3rd (triple actions), gives index in the entire dataset
        self.action_indices_by_ind = {ind: [[], [], []] for ind in self.unique_individuals}

        # All the calib paths in linux format
        self.calib_paths_all = []


        # Debug state
        self.debug = debug

        self.load_bigmac_overview()

        #### Placeholders for current path etc
        self.action_path = []
        self.inds = []
        self.calib_path = []

        ### Placeholder for the entire data
        self.entire_labeled_data = {}

        # The maximum length of an edge of an image
        self.max_image_size = max_image_size

        # Load the images in square, for batched camera projections
        self.same_size = same_size

        self.scale_per_ind_and_frame = np.array([1, 1])


        ## Labels aligned over time (
        # dictionary (Cam_Key (str), PerspectiveCamera (pytorch))
        self.pytorch_cameras = {}
        # dict over frames (int), with values being dicts over cameras (str) providing the masks as float numpy arrays
        self.masks = {}
        # list [2d_keypoints, 3d_keypoints], same format as masks
        self.keypoints_2d = {}
        self.keypoints_3d = {}
        self.keypoints_conf = {}
        # ratings of masks, list [ratings, mapping array position to camera key], same format
        self.probs_ratings = {}
        self.rgbs = {}



        # CameraObjects from Cameras class
        self.big_cams = Cameras()
        # Vertex info, symmetries etc. as a dictionary
        self.vertex_info = None
        # First frame to work, where masks are available etc.
        self.initial_frame = None
        # dict: (cameras, number of frames), used to align the frames across views w.r.t. time (
        #                                 more frames than minimum -> increase by that amount)
        self.additive_frames_cam = None

        # Minimum number of frames for that action
        self.min_frames_action = 0

        # Which videos of the calibrated cameras were actually present, Dict of keys
        # (str as in cam system, value str of camera padded)
        self.available_videos = None

        self.crop_data_action = {}

        self.manual_labels = None
        self.manual_occs = None
        self.manual_frames = None
        self.nb_markers = 20 #16


        # Paths of the action
        self.action_index = None
        self.yolo_path = None
        self.sam_path = None

        # The frames that have tracking data, but aligned...
        self.frame_indices = {}
        self.tracking_chk_sum = None

        # Mode, eval, over time etc.
        self.labels_only = labels_only

        self.device = device


        ## Load the vertex information
        self.vertex_symm_path = vertex_symm_path
        self.vertex_info = None
        self.high_poly = high_poly

        self.load_vertex_info()



        # Action path loaded labels sufix
        self.track_data_sufix = f"tracking_data_{max_image_size}_{same_size}.pkl"


        # Load the images with/without distortion
        self.load_undist_rgb = load_undist_rgb
        self.undist_rgbs = {}

        # Get the undistort maps
        self.undistort_maps = {}


        ### Multi-view alignment code
        # TODO: put the config into config files
        self.multiview_model_cfg = {'joint_num': self.nb_markers, 'spectral': True, 'alpha_SVT': 0.5,
                 'lambda_SVT': 50, 'dual_stochastic_SVT': False, }
        self.multiview_estimator = MultiEstimator(cfg=self.multiview_model_cfg)


        ### The association matrix

        self.association_matrix = None
        self.min_cams_cross_view = 4

        ## Cluster specific
        self.disable_saving = True



    def write_trckingdata_to_pickle(self):


        if self.labels_only:
            write_path = join(self.hard_drive_loc_fast, "Training")
            check_create_path(write_path)

            write_path = join(write_path, "IndividualFits")
            check_create_path(write_path)

            write_path = join(write_path, f"labels_{self.max_image_size}.joblib")
        else:
            write_path = join(self.action_path, self.track_data_sufix)

        with open(write_path, "wb") as file:
            joblib.dump(self.entire_labeled_data, file)
    def return_individual_write_path(self):

        write_path = join(self.hard_drive_loc_fast, "Training")
        write_path = join(write_path, "IndividualFits")
        write_path = join(write_path, f"labels_{self.max_image_size}.joblib")

        return write_path
    def load_trckingdata_from_pickle(self):


        if self.labels_only:
            write_path = self.return_individual_write_path()
        else:
            write_path = join(self.action_path, self.track_data_sufix)

        with open(write_path, "rb") as file:
            self.entire_labeled_data = joblib.load(file)

        # If tracking data load the current action index
        if not self.labels_only:
            action_labels_dict = self.entire_labeled_data[self.action_index]

            # Assign the variables from the dictionary to the class attributes
            self.keypoints_2d = action_labels_dict["kp2d"]
            self.keypoints_3d = action_labels_dict["kp3d"]
            self.keypoints_conf = action_labels_dict["kpconf"]
            self.masks = action_labels_dict["masks"]
            self.rgbs = action_labels_dict["rgbs"]
            self.pytorch_cameras = action_labels_dict["pycams"]
            self.probs_ratings = action_labels_dict["prob_ratings"]
            self.big_cams = action_labels_dict["big_cams"]
            self.additive_frames_cam = action_labels_dict["additive_frames"]
            self.initial_frame = action_labels_dict["initial_frame"]
            self.tracking_chk_sum = action_labels_dict["tracking_chk_sum"]
    def load_labeled_paths(self):


        tr_paths, tr_calib_paths, tr_inds, te_paths, te_calib_paths, te_inds = (
            load_paths_to_linux(save_loc=self.label_split))

        action_paths_combined = tr_paths + te_paths
        calib_paths_combined = tr_calib_paths + te_calib_paths
        inds_combined = tr_inds + te_inds

        self.unique_individuals = sorted(list(set(element for sublist in inds_combined for element in sublist)))


        self.unique_individuals_dict = {}
        for name in self.unique_individuals:
            self.unique_individuals_dict[name[0]] = name

        self.labeled_actions = list(zip(action_paths_combined, calib_paths_combined, inds_combined))
    def fill_action_indices_by_ind(self):

        # Iterate over each unique individual
        for individual in self.unique_individuals:
            # Filter rows where the individual's letter appears in the 'individuals' column
            filtered_rows = self.dataset[self.dataset['individuals'].str.contains(individual[0])]

            # Classify filtered rows based on 'nb_individuals'
            for idx, row in filtered_rows.iterrows():
                # Used len of the letters instead, as dataframe is not complete in nb_individuals
                nb_individuals = row['n_individuals']

                assert 4 > nb_individuals > 0

                # Determine the action category (1st: single, 2nd: double, 3rd: triple)
                if 1 <= nb_individuals <= 3:  # Ensure valid range
                    self.action_indices_by_ind[individual][nb_individuals - 1].append(idx)



        with open(join(self.project_root, f"data/tmp/actions_overview_by_ind.json"), 'w') as f:
            json.dump(self.action_indices_by_ind, f, indent=4)
    def load_bigmac_overview(self):

        self.dataset = pd.read_csv(self.dataset_overview_path)#, sep=';')

        if not self.debug:
            self.fill_action_indices_by_ind()

        path = join(self.hard_drive_loc, "Calibration")
        calib_paths = [x for x in os.listdir(path) if os.path.isdir(join(path, x))]

        # Shift Session 10 to the end
        calib_paths.append(calib_paths.pop(1))

        self.calib_paths_all = [join(path, x) for x in calib_paths]

    def get_calib_path_by_action(self, action_path):
        """
        Get the corresponding calib_path to an action path
        :param action_path:
        :return:
        """

        s0 = action_path.find("Session")
        s1 = action_path[s0:].find("/")

        session = action_path[s0:s0 + s1]

        abbrev_session = "S" + session[7:]

        calib_path_list = []
        for calib_path in self.calib_paths_all:
            last_index = calib_path.rfind("/")

            exact_session = calib_path[last_index+1:]

            if exact_session == abbrev_session:
                calib_path_list.append(calib_path)

        assert calib_path_list.__len__() == 1

        return calib_path_list[0]
    def load_vertex_info(self):
        """
        Loads the vertex info
        :return:
        """
        if self.vertex_symm_path:
            ## Load the symmetry of the mesh
            json_file = os.path.join(self.vertex_symm_path)
            with open(json_file, 'r') as infile:
                vertices_information = json.load(infile)


            if self.high_poly:
                nb_vertices = 10632
            else:
                nb_vertices = 3625
            check_vertex_information(vertices_information, nb_vertices=nb_vertices)

        else:
            vertices_information = None

        self.vertex_info = vertices_information

    def load_cameras_per_action(self, vid_index):

        self.action_index = vid_index
        # Action path
        row_series = self.dataset.loc[vid_index]
        self.action_path = row_series["path"]

        # Individuals, get the list of full names as usual
        self.individuals = [self.unique_individuals_dict[x] for x in row_series["individuals"]]

        # Calib_path, get the calibration path to the session
        self.calib_path = self.get_calib_path_by_action(self.action_path)

        # Load cameras from bigmac
        self.load_big_cams()

        # Also retrieve the additive frames
        self.get_additive_frames_from_file()




    def update_action(self, vid_index):

        self.action_index = vid_index

        if self.labels_only:

            # Update paths
            self.action_path, self.calib_path, self.individuals = self.labeled_actions[vid_index]

            # Update the action labels
            self.load_action_labels()
        else:

            ## Update paths

            # Action path

            row_series = self.dataset.loc[vid_index]


            base = Path(self.hard_drive_loc)
            old_path = Path(row_series["path"])

            find_str = "BigMaQ"
            relative = old_path.parts[old_path.parts.index(find_str) + 1:]
            action_path = base.joinpath(*relative)


            self.action_path = str(action_path)


            # Individuals, get the list of full names as usual
            self.individuals = [x for x in row_series["individuals"]]

            # Calib_path, get the calibration path to the session
            #self.calib_path = join(self.action_path, "calibration")
            self.calib_path = self.get_calib_path_by_action(self.action_path)

            # Update the action data
            self.load_action_labels()

    def load_big_cams(self, scaling=True):
        # Load the cameras from our own calib data
        cams = Cameras(calib_path=self.calib_path)
        cams.read_cam_params_from_toml(base_path=self.calib_path)

        # Scale them
        if scaling:
            scale = 1 / 1000
            #scale = 1
            for key, cam in enumerate(cams.cam_dict.values()):
                cam.t = cam.t * scale

        self.big_cams = cams

        # cams.show()
        # plt.show()
        # 0
    def return_views_present_yolo(self, yolo_path):

        available_views = [x[:2] for x in os.listdir(yolo_path) if 'npy' in x]
        #available_views_as_file = [x for x in os.listdir(yolo_path) if 'npy' not in x]

        view_dict = {}
        for view_index, view in enumerate(available_views):
            view_dict[str(int(available_views[view_index]))] = view

        self.available_videos = view_dict
    def load_YOLO(self):

        # Make the yolo path
        self.yolo_path = join(self.action_path, "Tracking")
        self.sam_path = join(self.action_path, "SAM")



        # Get available video_keys
        self.return_views_present_yolo(self.yolo_path)

        # Get the data, views are basically only the strings without
        for cam, cam_file in self.available_videos.items():
            self.crop_data_action[cam] = np.load(join(self.yolo_path, f"{cam_file}_crop_extended.npy"), allow_pickle=True)[()]
    def get_additive_frames_from_file(self):

        # Retrieve the videos of the current action
        video_names = [x for x in os.listdir(self.action_path) if '.mp4' and 'CoreView' in x]


        self.additive_frames_cam = {}
        self.video_name_mapping = {}

        for video_name in video_names:

            ### Open the video
            nb_frames = return_nb_frames_video(join(self.action_path, video_name))

            ## Extract the name as well
            cam_view = video_name[video_name.rfind('_') + 1 : -4]

            self.additive_frames_cam[cam_view] = nb_frames
            self.video_name_mapping[cam_view] = video_name

        self.min_frames_action = int(min(self.additive_frames_cam.values()))
    def load_manual_labels(self):

        label_path = join(self.action_path, "labels.npy")

        if os.path.exists(label_path):

            # Get the frames that were labeled across views, format as in previous data loading functions
            self.manual_labels = np.load(label_path, allow_pickle=True)[()]

            # Returns where there is no nan in a frame
            frames = np.where(np.isnan(next(iter(self.manual_labels.values()))[0, 0, :]) == False)[0]

            self.manual_frames = [int(x) for x in frames]

            self.manual_occs = np.load(join(self.action_path, "occs.npy"), allow_pickle=True)[()]
    def load_action_labels(self):

        # Load cameras from bigmac
        self.load_big_cams()

        # Load keypoints and cropping information
        self.load_YOLO()

        # Also retrieve the additive frames
        self.get_additive_frames_from_file()

        if self.labels_only:
            # Check if there is any label information to load in eval mode
            self.load_manual_labels()
    def get_cropped_gt_2d_labels(self, frame, cam_key, ind, UL):

        # Get the camera
        cam = self.big_cams.cam_dict[cam_key]

        # Get the individual keypoints
        keypoints_ind = self.manual_labels[cam_key][ind * self.nb_markers : (ind+1) * self.nb_markers, :, frame]

        # Undistort points
        keypoints_percam_undist = cv2.undistortImagePoints(keypoints_ind, cam.K, cam.d)
        keypoints_percam_undist = np.squeeze(keypoints_percam_undist, axis=1)

        # Crop the keypoints by bounding box coordinates
        keypoints_percam_undist = keypoints_percam_undist - UL #keypoints_percam_undist * resize_factor

        return keypoints_percam_undist
    def get_3d_gt_labels(self, frame, ind_index):

        # Initiliaze 3d keypoints in number frames
        keypoint_3d_bp = np.zeros((self.nb_markers, 3))

        for marker_idx in range(self.nb_markers):
            # create the vis_m_dict
            vis_m_dict = {}
            for cam_name in self.big_cams.cam_dict.keys():
                if self.manual_occs[cam_name][marker_idx + ind_index * self.nb_markers, frame] == 0:
                    vis_m_dict[cam_name] = self.manual_labels[cam_name][marker_idx + ind_index * self.nb_markers, :,
                                           frame]
            r, rep_error, best_set, best_index, triang_dic = \
                self.big_cams.lift_marker(vis_labels=vis_m_dict, undist=True, ransac=True)
            keypoint_3d_bp[marker_idx, :] = r

        return keypoint_3d_bp
    def get_3d_preds(self, individual, keypoint_preds, keypoints_conf, threshold = 0.9):
        """
        Get a per frame 3d pose for a given individual
        :param frame: The current frame
        :param ind_index: The individual index
        :param keypoint_preds:
        :return:
        """



        # Initiliaze 3d keypoints in number frames
        keypoint_3d_bp = np.zeros((self.nb_markers, 3))

        for marker_idx in range(self.nb_markers):
            # create the vis_m_dict
            vis_m_dict = {}

            cam_list = list(self.big_cams.cam_dict.keys())
            random.shuffle(cam_list)




            # Randomized names, and after 5 cut it
            for cam_name in cam_list:

                keypoints_cpu = keypoint_preds[cam_name][individual].cpu().numpy()
                kp_conf_cpu = keypoints_conf[cam_name][individual].cpu().numpy()

                marker_2d_pos = keypoints_cpu[marker_idx, :]
                # Check if that prediction is valid then append it in vis_m_dict
                if marker_2d_pos.sum() != -2:
                    # Check that the marker confidence is large enough
                    if kp_conf_cpu[marker_idx] > threshold:
                        vis_m_dict[cam_name] = marker_2d_pos

                # Speeds it up
                if vis_m_dict.__len__() == 4:
                    break

            # Do not overwrite the markers if there are not at least two cameras available
            if vis_m_dict.__len__() < 2:
                continue

            r, rep_error, best_set, best_index, triang_dic = \
                self.big_cams.lift_marker(vis_labels=vis_m_dict, undist=True, ransac=True)

            keypoint_3d_bp[marker_idx, :] = r

        return keypoint_3d_bp
    def make_bb_square(self, undistorted_bbox, im_h, im_w):
        """
        A function to create a squared bbox within the image regions.
        :param undistorted_bbox:
        :param im_h:
        :param im_w:
        :return:
        """

        width, height = undistorted_bbox[1, :] - undistorted_bbox[0, :]

        edge_diff = width - height

        delta = abs(edge_diff)
        # If negative, expand the x coordinates
        if edge_diff <= 0:

            # Reduce the lower side first
            x_ul = undistorted_bbox[0, 0] - delta
            x_lr = undistorted_bbox[1, 0]

            if x_ul < 0:
                x_ul = undistorted_bbox[0, 0]
                x_lr = undistorted_bbox[1, 0] + delta

            y_ul = undistorted_bbox[0, 1]
            y_lr = undistorted_bbox[1, 1]

        else:

            y_ul = undistorted_bbox[0, 1] - delta
            y_lr = undistorted_bbox[1, 1]

            if y_ul < 0:
                y_ul = undistorted_bbox[0, 1]
                y_lr = undistorted_bbox[1, 1] + delta

            x_ul = undistorted_bbox[0, 0]
            x_lr = undistorted_bbox[1, 0]

        square_undist_bbox = np.array([[x_ul, y_ul], [x_lr, y_lr]])

        width, height = square_undist_bbox[1, :] - square_undist_bbox[0, :]

        assert width - height == 0

        return square_undist_bbox

    def get_undistorted_labels(self, cam_key, bbox, unaligned_frame, first_letter_ind, kps_2d_det,
                               rgb=None):
        """
        Creates the mask and undistorted rgb image, with undistorted bounding box parameters as well
        :param cam_key:
        :param bbox:
        :param unaligned_frame:
        :param first_letter_ind:
        :return:
        """

        # Load the mask, mask is from bounding boxes, but as these labels were used to train it, likely to be correct
        video_prefix = os.listdir(self.sam_path)[0]
        video_prefix = video_prefix[:video_prefix.rfind("_")+1]
        cam_folder = video_prefix + f"{cam_key}"
        cam_path = join(self.sam_path, cam_folder)

        # Need to use the unaligned frame
        mask_path = join(cam_path, f"{str(unaligned_frame).zfill(5)}_seg_{first_letter_ind}.tiff")
        mask = np.array(Image.open(mask_path))

        # Pad it and undistort it
        cam = self.big_cams.cam_dict[cam_key]

        if cam_key not in self.undistort_maps.keys():

            # Create the undistort map
            w, h = cam.im_size

            ## Exec is 15ms, compared to 40ms for cv2 undist

            # Initialize the rectify map
            map1, map2 = cv2.initUndistortRectifyMap(cam.K, cam.d, None, cam.K,
                                                     (w, h), cv2.CV_32FC1)
            self.undistort_maps[cam_key] = [map1, map2]



        # Changed to rectify
        mask_mk = cv2.remap(mask, self.undistort_maps[cam_key][0], self.undistort_maps[cam_key][1], interpolation=cv2.INTER_LINEAR)
        #mask_mk = cv2.undistort(mask, cam.K, cam.d)

        # then undistort the bbox, and then shift pixels again with image region and so that they satisfy the same length requirements, basically by flipping
        # then you have the same sides, the distorted bbox itself is only then still necessary for retrieving the keypoint positions

        ## Get improved bounding box coordinates
        # Distorted bbox (xyxy format: UL, LR)
        distorted_bbox = np.array([bbox[:2], bbox[2:]]).reshape(-1, 2)

        ## BBox coordinates --> xyxy (UL, LR)
        #undistorted_bbox = np.array([[[bbox[0], bbox[1]], [bbox[2], bbox[3]]]])

        undistorted_bbox = cv2.undistortPoints(distorted_bbox, cam.K, cam.d)

        # Convert to pixel coordinates in the undistorted image
        undistorted_bbox = np.dot(undistorted_bbox.reshape(-1, 2), cam.K[:2, :2].T) + cam.K[:2, 2]

        ### Make sure the undistorted bounding box values are within the image region range
        im_w, im_h = cam.im_size

        clipped_ys = np.clip(undistorted_bbox[:, 1], a_min=0, a_max=im_h-1).reshape(2, 1)
        clipped_xs = np.clip(undistorted_bbox[:, 0], a_min=0, a_max=im_w-1).reshape(2, 1)

        undistorted_bbox = np.hstack((clipped_xs, clipped_ys))


        # Convert to int, so pixels can be accessed
        undistorted_bbox = undistorted_bbox.astype(np.int32)
        distorted_bbox = distorted_bbox.astype(np.int32)
        # Just plot for once if the cropping is in the correct areas
        # cropped_mask = mask_mk[int(bbox[1]):int(bbox[3]),
        #                int(bbox[0]):int(bbox[2])]


        if self.same_size:
            undistorted_bbox = self.make_bb_square(undistorted_bbox, im_h, im_w)

            # Pad the image if it exceeded the limits
            mask_mk = (utils.pad_image_to_bbox(np.expand_dims(mask_mk, axis=-1), bbox=undistorted_bbox)).squeeze()

            # If it needs to be used in optimization
            if self.load_undist_rgb:
                rgb = utils.pad_image_to_bbox(rgb, bbox=undistorted_bbox)
            else:
                # Else the padding is not required. Just load the entire image, and scale the rendering.
                pass

        cropped_mask = mask_mk[undistorted_bbox[0, 1]:undistorted_bbox[1, 1],
                       undistorted_bbox[0, 0]:undistorted_bbox[1, 0]]



        ## RGB images for that frame
        # yolo_path = join(self.action_path, "YOLO_results")
        # cam_path = join(yolo_path, f"{str(cam_key).zfill(2)}")
        # image_path = join(cam_path, f"{first_letter_ind}_{str(unaligned_frame).zfill(5)}_cropped.jpg")
        # rgb = np.array(Image.open(image_path))


        if self.load_undist_rgb:
            ## Construct the entire image, undistort and crop with the undistorted bounding box
            #rgb_full = np.zeros(shape=(cam.im_size[1], cam.im_size[0], 3))
            #rgb_full[distorted_bbox[0, 1]:distorted_bbox[1, 1], distorted_bbox[0, 0]:distorted_bbox[1, 0], :] = rgb

            # Changed to remap
            rgb_full_undist = cv2.remap(rgb, self.undistort_maps[cam_key][0], self.undistort_maps[cam_key][1],
                                interpolation=cv2.INTER_LINEAR)
            #rgb_full_undist = cv2.undistort(rgb_full, cam.K, cam.d)
            rgb_full_undist_cropped = rgb_full_undist[undistorted_bbox[0, 1]:undistorted_bbox[1, 1], undistorted_bbox[0, 0]:undistorted_bbox[1, 0], :]
        else:
            rgb_full_undist_cropped = rgb



        #------------------- For checking the proper transforms
        # # Apply the mask to the cropped undistorted RGB image
        # masked_rgb = cv2.bitwise_and(rgb_full_undist_cropped, rgb_full_undist_cropped, mask=cropped_mask)
        #
        # # Plot the result
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.title("Cropped Undistorted Mask")
        # plt.imshow(cropped_mask, cmap="gray")
        # plt.axis("off")
        #
        # plt.subplot(1, 2, 2)
        # plt.title("Cropped Undistorted RGB with Mask")
        # plt.imshow(rgb_full_undist_cropped.astype(np.uint8))
        # plt.axis("off")
        #
        # plt.tight_layout()
        # plt.show()
        # ---------------




        ## Resize
        scale_factor = 1

        if self.max_image_size is not None:

            c_h, c_w = cropped_mask.shape
            scale_factor = self.max_image_size / max(c_h, c_w)

            if self.same_size:
                # The output size is known, and given by the max image size as it is square.
                s_h = self.max_image_size
                s_w = self.max_image_size
            else:
                s_h = round(c_h * scale_factor)
                s_w = round(c_w * scale_factor)

            cropped_mask = resize(cropped_mask, (s_h, s_w), anti_aliasing=True, preserve_range=False)

            rgb_full_undist_cropped = resize(rgb_full_undist_cropped, (s_h, s_w), anti_aliasing=True, preserve_range=True)


        # Make sure the range of values is between 0 and 1
        cropped_mask = cropped_mask * (1 / cropped_mask.max())


        ## Keypoints
        # Uncrop them
        uncropped_kp_2d = kps_2d_det + distorted_bbox[0, :]
        # Undistort them
        undistorted_kp_2d = cv2.undistortPoints(uncropped_kp_2d, cam.K, cam.d)

        # Convert to pixel coordinates in the undistorted image
        undistorted_kp_2d = np.dot(undistorted_kp_2d.reshape(-1, 2), cam.K[:2, :2].T) + cam.K[:2, 2]

        # Crop them again in undistorted coordinates
        undistorted_kp_2d = undistorted_kp_2d - undistorted_bbox[0, :]

        if self.same_size:
            assert cropped_mask.shape[0] == cropped_mask.shape[1] == self.max_image_size
            assert rgb_full_undist_cropped.shape[0] == rgb_full_undist_cropped.shape[1] == self.max_image_size


        return cropped_mask, scale_factor, rgb_full_undist_cropped, undistorted_bbox.flatten(), undistorted_kp_2d, uncropped_kp_2d
    def get_cropped_camera(self, cam_key, bbox, im_size, scale_factor):
        """
        Get the original camera from that viewing angle, along with mask im size (h, w), and the bbox information
        for the tracked individual. Return a camera for the cropped view point.
        :param cam_key:
        :param bbox:
        :param im_size:
        :param scale_factor: The scaling factor to preserve the max image range
        :return:
        """
        cam = self.big_cams.cam_dict[cam_key]

        UL = bbox[:2].reshape((1, 2))
        #LR = bbox[2:].reshape((1, 2))


        cropped_K = copy.deepcopy(cam.K)
        # Substract the Corner
        cropped_K[0, 2] -= int(UL[0, 0])
        cropped_K[1, 2] -= int(UL[0, 1])

        cropped_K *= scale_factor

        # (width, height)
        im_size_array = np.array([im_size[1], im_size[0]])
        im_size_array = np.array([im_size[0], im_size[1]])

        # camera = conversion_opencv_pytorch3d_extended(cam.R, cam.t, fx, fy, px, py, cam.im_size, device=device)
        cropped_camera = conversion_opencv_pytorch(cam.R, cam.t, cropped_K, im_size_array, device=self.device)

        return cropped_camera




    def get_uncropped_labels_per_frame(self, manual=False):


        if manual:
            frames_to_iter = self.manual_frames
        else:
            frames_to_iter = range(self.min_frames_action)

        # Use arrays instead of the dictionaries, but masks change their size...
        mask_p_c = {}
        kp_2d_p_c = {}
        kp_2d_conf_p_c = {}
        kp2d_uncropped_p_c = {}
        cams_p_c = {}
        prob_ratings_p_c = {}
        rgb_p_c = {}

        # Iterate the viewpoints
        for cam_key_index, cam_key in tqdm(enumerate(self.available_videos.keys())):

            # Read all frames of that video
            video_path = join(self.action_path, self.video_name_mapping[cam_key])
            image_frames = utils.read_frames_from_video(video_path=video_path, start_frame=0, num_following=1000000)


            # Iterate the viewpoints
            for frame in frames_to_iter:

                unaligned_frame = frame + int(self.additive_frames_cam[cam_key]) - self.min_frames_action

                # Initialize per-frame containers if not already
                if frame not in kp_2d_p_c:
                    kp_2d_p_c[frame] = {}
                    kp_2d_conf_p_c[frame] = {}
                    kp2d_uncropped_p_c[frame] = {}
                    mask_p_c[frame] = {}
                    cams_p_c[frame] = {}
                    prob_ratings_p_c[frame] = {}
                    rgb_p_c[frame] = {}


                mask_p_i = {}
                kp_2d_p_i = {}
                kp_2d_conf_p_i = {}
                kp2d_uncropped_p_i = {}
                cams_p_i = {}
                prob_ratings_p_i = {}
                rgb_p_i = {}

                # Iterate the individuals too
                for ind_index, ind in enumerate(self.individuals):

                    first_letter_ind = ind[0]



                    if self.crop_data_action[cam_key][unaligned_frame][first_letter_ind].__len__() == 3:

                        # Make the positions -1
                        kp_2d_p_i[ind] = -1*np.ones(shape=(self.nb_markers, 2))
                        kp2d_uncropped_p_i[ind] = -1*np.ones(shape=(self.nb_markers, 2))
                        # Put the confidences to 0
                        kp_2d_conf_p_i[ind] = np.zeros(self.nb_markers)
                        prob_ratings_p_i[ind] = 0
                        mask_p_i[ind] = np.zeros(shape=(10, 10))

                        cams_p_i[ind] = None
                        rgb_p_i[ind] = None

                        kp_2d_p_i[ind] = torch.tensor(kp_2d_p_i[ind], device=self.device)
                        kp_2d_conf_p_i[ind] = torch.tensor(kp2d_uncropped_p_i[ind], device=self.device)
                        kp2d_uncropped_p_i[ind] = torch.tensor(kp2d_uncropped_p_i[ind], device=self.device)

                    else:
                        # Get the cropping information from YOLO
                        bbox, ind_id, bbox_conf, kps_2d_det, kps_2d_conf = self.crop_data_action[cam_key][unaligned_frame][first_letter_ind]




                        ### Masks, and rgb assignment
                        mask_p_i[ind], scale_factor, rgb_p_i[ind], undist_bbox, undist_2d_kp, kp_2d_uncropped = (
                            self.get_undistorted_labels(cam_key, bbox, unaligned_frame, first_letter_ind, kps_2d_det,
                                                        rgb=image_frames[unaligned_frame]))
                        im_size = mask_p_i[ind].shape[:2]

                        ### Pytorch cameras
                        cams_p_i[ind] = self.get_cropped_camera(cam_key, undist_bbox, im_size, scale_factor)

                        UL = undist_bbox[:2].reshape((1, 2))

                        ### Keypoints
                        if manual:
                            # Frames are already aligned
                            kp_2d = self.get_cropped_gt_2d_labels(frame, cam_key, ind_index, UL)
                        else:
                            # Undistorted labels
                            kp_2d = undist_2d_kp

                        mask_p_i[ind] = torch.tensor(mask_p_i[ind], device=self.device)
                        rgb_p_i[ind] = torch.tensor(rgb_p_i[ind], device=self.device)

                        kp_2d_p_i[ind] = torch.tensor(kp_2d * scale_factor, device=self.device)
                        kp_2d_conf_p_i[ind] = torch.tensor(np.array(kps_2d_conf), device=self.device)
                        kp2d_uncropped_p_i[ind] = torch.tensor(kp_2d_uncropped, device=self.device)

                        ### Prob ratings, which are currently detection confidences...
                        prob_ratings_p_i[ind] = bbox_conf

                        # Check for Nan, if erroneous get rid of the frame
                        if torch.isnan(torch.max(mask_p_i[ind])).item():
                            # Make the positions -1
                            kp_2d_p_i[ind] = -1 * np.ones(shape=(self.nb_markers, 2))
                            kp2d_uncropped_p_i[ind] = -1 * np.ones(shape=(self.nb_markers, 2))
                            # Put the confidences to 0
                            kp_2d_conf_p_i[ind] = np.zeros(self.nb_markers)
                            prob_ratings_p_i[ind] = 0
                            mask_p_i[ind] = np.zeros(shape=(10, 10))

                            kp_2d_p_i[ind] = torch.tensor(kp_2d_p_i[ind], device=self.device)
                            kp_2d_conf_p_i[ind] = torch.tensor(kp2d_uncropped_p_i[ind], device=self.device)
                            kp2d_uncropped_p_i[ind] = torch.tensor(kp2d_uncropped_p_i[ind], device=self.device)


                            cams_p_i[ind] = None
                            rgb_p_i[ind] = None

                    # Update the checksums to check for fast camera retrieval
                    if cams_p_i[ind] is not None and not manual:
                        # Use the cam key as index
                        self.tracking_chk_sum[ind_index, frame, int(cam_key) - 1] = 1


                kp_2d_p_c[frame][cam_key] = kp_2d_p_i
                kp_2d_conf_p_c[frame][cam_key] = kp_2d_conf_p_i
                kp2d_uncropped_p_c[frame][cam_key] = kp2d_uncropped_p_i
                mask_p_c[frame][cam_key] = mask_p_i
                cams_p_c[frame][cam_key] = cams_p_i
                prob_ratings_p_c[frame][cam_key] = prob_ratings_p_i
                rgb_p_c[frame][cam_key] = rgb_p_i

        for frame in frames_to_iter:

            ## Fill the 3d gt (camera ind.)
            kp_3d_ind = {}
            for ind_index, ind in enumerate(self.individuals):
                if manual:
                    kp_3d_ind[ind] = self.get_3d_gt_labels(frame, ind_index)
                else:
                    # Problem is that the 2d labels are basically cropped and shifted, use the original ones
                    kp_3d_ind[ind] = self.get_3d_preds(ind, kp2d_uncropped_p_c[frame], kp_2d_conf_p_c[frame])

            ### Update the objects entries
            self.keypoints_2d[frame] = kp_2d_p_c[frame]

            self.keypoints_3d[frame] = kp_3d_ind
            self.keypoints_conf[frame] = kp_2d_conf_p_c[frame]
            self.masks[frame] = mask_p_c[frame]
            self.pytorch_cameras[frame] = cams_p_c[frame]
            self.probs_ratings[frame] = prob_ratings_p_c[frame]
            self.rgbs[frame] = rgb_p_c[frame]

            # add uncropped kp2d


    def update_tracking_data(self):
        """
        Returns labels over time, when queried by frame indices
        :return:
        """

        # Initialize a matrix over individuals, cameras, frames
        self.tracking_chk_sum = np.zeros(shape=(len(self.individuals), self.min_frames_action,
                                                len(self.big_cams.cam_dict)))

        print("Loading action: ", self.action_path)
        self.get_uncropped_labels_per_frame(manual=False)
    def update_manual_labels_only(self):
        """
        Updates the labels to manual labels only and corresponding frames and keypoints etc.
        :return:
        """

        self.get_uncropped_labels_per_frame(manual=True)

        self.initial_frame = self.manual_frames[0]
    def set_action_loader_by_index(self, action_index):

        # Assign the action index
        self.action_index = action_index

        self.action_path, self.calib_path, self.inds = self.labeled_actions[self.action_index]

        # Updates the cropping parameters
        #self.load_YOLO()

        self.keypoints_2d = self.entire_labeled_data[action_index]["kp2d"]
        self.keypoints_conf = self.entire_labeled_data[action_index]["kpconf"]
        self.masks = self.entire_labeled_data[action_index]["masks"]
        self.pytorch_cameras = self.entire_labeled_data[action_index]["pycams"]
        self.probs_ratings = self.entire_labeled_data[action_index]["prob_ratings"]
        self.big_cams = self.entire_labeled_data[action_index]["big_cams"]
        self.additive_frames_cam = self.entire_labeled_data[action_index]["additive_frames"]
        self.initial_frame = self.entire_labeled_data[action_index]["initial_frame"]

        self.keypoints_3d = self.entire_labeled_data[action_index]["kp3d"]

        self.rgbs = self.entire_labeled_data[action_index]["rgbs"]


    def load_entire_labeled_frames(self, reload=False):

        if (os.path.exists(self.return_individual_write_path()) and not reload):
            print("Loading the labeled data from file.")

            start_time = time.time()
            self.load_trckingdata_from_pickle()
            print(f"Time in seconds spent for loading the data: {time.time()-start_time}")
        else:
            print("Loading the entire dataset.")
            for action_index in tqdm(range(len(self.labeled_actions))):

                if self.debug and action_index > 5:
                    break

                self.load_action_into_entire_dict(action_index)
            self.write_trckingdata_to_pickle()
    def retrieve_action_frames(self):
        """
        Find the frames that can be processed.
        """

        # Find the available tracking data as frames per individual, and per camera

        for ind_index, ind in enumerate(self.individuals):

            ind_dict = {}

            for cam_key_available, crop_data in self.crop_data_action.items():

                list_of_frames = []

                for unaligned_frame, crop_d_f in crop_data.items():

                    # Continue if there is no data available
                    if crop_d_f[ind[0]].__len__() == 3:
                        continue

                    # Align the frame
                    frame = unaligned_frame - (int(self.additive_frames_cam[cam_key_available]) - self.min_frames_action)


                    # Get rid of first frames that are not shared across all cameras
                    # The alignment w
                    if frame < 0:
                        continue

                    list_of_frames.append(frame)

                ind_dict[cam_key_available] = list_of_frames

            self.frame_indices[ind] = ind_dict
    def load_action_into_entire_dict(self, action_index):

        # Otherwise overwrites itself
        self.keypoints_2d = {}
        self.keypoints_3d = {}
        self.keypoints_conf = {}
        self.masks = {}
        self.pytorch_cameras = {}
        self.probs_ratings = {}

        self.rgbs = {}

        self.update_action(action_index)

        if self.labels_only:

            self.update_manual_labels_only()

            # action_labels_dict = {"kp2d": self.keypoints_2d,
            #                       "kpconf": self.keypoints_conf,
            #                       "masks": self.masks,
            #                       "rgbs": self.rgbs,
            #                       "pycams": self.pytorch_cameras,
            #                       "prob_ratings": self.probs_ratings,
            #                       "big_cams": self.big_cams,
            #                       "additive_frames": self.additive_frames_cam,
            #                       "initial_frame": self.initial_frame}

            action_labels_dict = {"kp2d": self.keypoints_2d,
                                  "kpconf": self.keypoints_conf,
                                  "masks": self.masks,
                                  "rgbs": self.rgbs,
                                  "pycams": self.pytorch_cameras,
                                  "prob_ratings": self.probs_ratings,
                                  "big_cams": self.big_cams,
                                  "additive_frames": self.additive_frames_cam,
                                  "initial_frame": self.initial_frame,
                                  "tracking_chk_sum": self.tracking_chk_sum,
                                  "kp3d": self.keypoints_3d}

            self.entire_labeled_data[action_index] = action_labels_dict

        else:
            # Flush the labeled dataset
            self.entire_labeled_data = {}

            self.retrieve_action_frames()


            if (os.path.exists(join(self.action_path, self.track_data_sufix))) and (not self.force_action_reload):
                self.load_trckingdata_from_pickle()
            else:
                self.update_tracking_data()

                action_labels_dict = {"kp2d": self.keypoints_2d,
                                      "kpconf": self.keypoints_conf,
                                      "masks": self.masks,
                                      "rgbs": self.rgbs,
                                      "pycams": self.pytorch_cameras,
                                      "prob_ratings": self.probs_ratings,
                                      "big_cams": self.big_cams,
                                      "additive_frames": self.additive_frames_cam,
                                      "initial_frame": self.initial_frame,
                                      "tracking_chk_sum": self.tracking_chk_sum,
                                      "kp3d": self.keypoints_3d}

                self.entire_labeled_data[action_index] = action_labels_dict

                if not self.disable_saving:
                    self.write_trckingdata_to_pickle()



    def fill_with_scaled_cameras(self):


        # Iterate the min frames of the action
        for frame in range(self.min_frames_action):

            # Use arrays instead of the dictionaries, but masks change their size...
            cams_p_c = {}

            # Iterate the viewpoints
            for cam_key_index, cam_key in enumerate(self.available_videos.keys()):

                cams_p_i = {}

                # Iterate the individuals too
                for ind_index, ind in enumerate(self.individuals):

                    cam_key = str(cam_key)
                    cam = self.big_cams.cam_dict[cam_key]

                    # camera = conversion_opencv_pytorch3d_extended(cam.R, cam.t, fx, fy, px, py, cam.im_size, device=device)


                    img_size = np.array([cam.im_size[1], cam.im_size[0]])
                    # im_size in (h, w) in
                    camera = conversion_opencv_pytorch(cam.R, cam.t, cam.K, img_size, device=self.device)
                    cams_p_i[ind] = camera

                cams_p_c[cam_key] = cams_p_i

            ### Update the objects entries
            self.pytorch_cameras[frame] = cams_p_c
    def load_scaled_cameras(self, action_index):

        # Otherwise overwrites itself
        self.keypoints_2d = {}
        self.keypoints_3d = {}
        self.keypoints_conf = {}
        self.masks = {}
        self.pytorch_cameras = {}
        self.probs_ratings = {}

        self.rgbs = {}

        self.update_action(action_index)

        # Flush the labeled dataset
        self.entire_labeled_data = {}

        # Load the cameras for every frame of the video sequence.
        self.fill_with_scaled_cameras()

        action_labels_dict = {"kp2d": self.keypoints_2d,
                              "kpconf": self.keypoints_conf,
                              "masks": self.masks,
                              "rgbs": self.rgbs,
                              "pycams": self.pytorch_cameras,
                              "prob_ratings": self.probs_ratings,
                              "big_cams": self.big_cams,
                              "additive_frames": self.additive_frames_cam,
                              "initial_frame": self.initial_frame,
                              "tracking_chk_sum": self.tracking_chk_sum,
                              "kp3d": self.keypoints_3d}
        self.entire_labeled_data[action_index] = action_labels_dict

    def get_background_images_batched(self, frame, batch_range = 30):

        min_frames_action = int(min(self.additive_frames_cam.values()))

        # Get the video files in the action path folder
        video_files = [x for x in os.listdir(self.action_path) if '.mp4' and 'CoreView' in x]




        # Get the position of the second underscore
        underscore_pos = utils.find_second_underscore(video_files[0])



        frames_cam_dict = {}

        for cam_key in self.big_cams.cam_dict.keys():



            video_name = video_files[0][:underscore_pos + 1] + f"{cam_key}.mp4"

            # Open the video with opencv
            if self.additive_frames_cam[cam_key] > min_frames_action:
                add = 1
            else:
                add = 0

            unaligned_frame = frame + add
            # Load the background image
            background_images = utils.read_frames_from_video(join(self.action_path, video_name),
                                                      start_frame=unaligned_frame, num_following=batch_range)

            frames_frame_dict = {}


            for background_image_index, background_image in enumerate(background_images):

                frames_frame_dict[frame + background_image_index] = background_image


            frames_cam_dict[cam_key] = frames_frame_dict

        self.undist_rgbs = frames_cam_dict


    def export_action(self, export_path, start_frame, end_frame, cameras_to_export, export_scale_reduce=1,
                      export_vids=True, indep_cam=None, FF=False):

        self.export_path = export_path
        self.cameras_to_export = cameras_to_export


        ### Create the folders
        data_path = join(export_path, "data")

        if FF:
            action_path = join(data_path, f"big_mac_{self.action_index}_0")
        else:
            action_path = join(data_path, f"big_mac_{self.action_index}")

        self.export_explicit = action_path
        os.makedirs(action_path, exist_ok=True)

        subfolders = ["keypoints2d_undist", "simpleclick_undist", "videos_undist"]
        for subfolder in subfolders:
            os.makedirs(join(action_path, subfolder), exist_ok=True)



        self.start_frame = start_frame
        self.end_frame = end_frame

        self.export_scale_reduce = export_scale_reduce

        # Get the cameras first.
        self.export_scaled_cameras(indep_cam=indep_cam)

        # Export the keypoints
        self.export_keypoints()

        if export_vids:
            # Export the undistorted masks
            self.export_videos(mask=True)
            # Export the undistorted videos
            self.export_videos(indep_cam=indep_cam)



        print("Finished video exporting")


    def export_keypoints(self):

        # Only single individual actions for now
        assert len(self.individuals) == 1

        first_letter_ind = self.individuals[0][0]



        for cam_iter_index, cam_id in enumerate(self.cameras_to_export):
            cam_id = str(cam_id)

            unaligned_frame_start = self.start_frame + int(self.additive_frames_cam[cam_id]) - self.min_frames_action
            unaligned_frame_end = unaligned_frame_start + (self.end_frame - self.start_frame)


            keypoints_to_write = np.zeros(shape=(self.end_frame - self.start_frame, 20, 3))

            for frame_index, unaligned_frame in enumerate(range(unaligned_frame_start, unaligned_frame_end, 1)):
                bbox, ind_id, bbox_conf, kps_2d_det, kps_2d_conf = self.crop_data_action[cam_id][unaligned_frame][first_letter_ind]


                kps_2d_det = np.array(kps_2d_det)
                kps_2d_conf = np.expand_dims(np.array(kps_2d_conf), axis=1)

                UL = bbox[:2]

                keypoints_dist = UL + kps_2d_det

                big_cam = self.big_cams.cam_dict[cam_id]
                undistorted_points = cv2.undistortPoints(keypoints_dist, big_cam.K, big_cam.d).squeeze()

                # Convert to pixel coordinates in the undistorted image
                undistorted_points = np.dot(undistorted_points.reshape(-1, 2), big_cam.K[:2, :2].T) + big_cam.K[:2, 2]

                undistorted_points = undistorted_points / self.export_scale_reduce


                keypoint_info = np.hstack((undistorted_points, kps_2d_conf))

                keypoints_to_write[frame_index, :, :] = keypoint_info



            # Uncrop and undistort the keypoints

            # Save the infos accordingly
            kp_path = join(self.export_explicit, f"keypoints2d_undist/result_view_{cam_iter_index}.pkl")
            with open(kp_path, 'wb') as file:
                pickle.dump(keypoints_to_write, file)



    def export_videos(self, mask=False, indep_cam=None):


        if indep_cam is not None:
            cams_to_export = self.cameras_to_export.copy()
            cams_to_export.append(indep_cam)
        else:
            cams_to_export = self.cameras_to_export


        for cam_iter_index, cam_id in enumerate(cams_to_export):

            cam_id = str(cam_id)

            frames_as_list = []

            unaligned_frame_start = self.start_frame + int(self.additive_frames_cam[cam_id]) - self.min_frames_action
            unaligned_frame_end = unaligned_frame_start + (self.end_frame - self.start_frame)


            # Initialize the videowriter here and just push another frame to write in it

            if not mask:
                output_path = join(self.export_explicit, f"videos_undist/{cam_iter_index}.mp4")

                sam_path = join(self.action_path, "SAM")
                video_prefix = os.listdir(sam_path)[0]
                video_prefix = video_prefix[:video_prefix.rfind("_") + 1]
                vid_path = join(self.action_path, video_prefix + f"{cam_id}.mp4")

                video_loader = cv2.VideoCapture(vid_path)


            else:
                output_path = join(self.export_explicit, f"simpleclick_undist/{cam_iter_index}.mp4")

            # Use 'mp4v' codec for MP4 files
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')


            scaled_size = tuple(self.big_cams.cam_dict[cam_id].im_size // self.export_scale_reduce)
            video_writer = cv2.VideoWriter(output_path, fourcc, 40, scaled_size)


            ## Get the frames
            # end_frame non-inclusive
            for i in range(unaligned_frame_start, unaligned_frame_end, 1):

                # Get the video
                if not mask:

                    video_loader.set(cv2.CAP_PROP_POS_FRAMES, i)  # Set to the specific frame
                    ret, image = video_loader.read()

                    assert ret

                    # Load the frames of the video with open cv, then undistort them, save them as video to export path

                else:
                    sam_path = join(self.action_path, "SAM")
                    video_prefix = os.listdir(sam_path)[0]
                    video_prefix = video_prefix[:video_prefix.rfind("_") + 1]

                    cam_folder = video_prefix + f"{cam_id}"
                    cam_path = join(sam_path, cam_folder)

                    # Get the rest of the image name
                    image_postfix = os.listdir(cam_path)[0]
                    image_postfix = image_postfix[image_postfix.find("_"):]

                    mask_path = join(cam_path, f"{str(i).zfill(5)}"+image_postfix)
                    image = np.array(Image.open(mask_path))

                    # Expand dimensions to color
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


                # Undistort the image and write it
                image_undist = cv2.remap(image, self.export_cameras[cam_iter_index]['mapx_large'],
                                         self.export_cameras[cam_iter_index]['mapy_large'], interpolation=cv2.INTER_LINEAR)


                # Resize
                # Resize the image
                image_undist = cv2.resize(image_undist, scaled_size, interpolation=cv2.INTER_AREA)

                video_writer.write(image_undist)

            # Release the writer
            video_writer.release()

            if not mask:
                video_loader.release()


                # Load the individual frames from the respective folder, undistort and put into a video


            # undistortion etc.


    def export_scaled_cameras(self, indep_cam=None):


        ### big_cams existieren ja schon

        self.export_cameras = []
        self.export_cameras_small = []
        # Iterieren der Cameras nach Zahl

        # Set the range manually
        cam_keys = range(1, 17, 1)

        if indep_cam is not None:
            cams_to_export = self.cameras_to_export.copy()
            cams_to_export.append(indep_cam)
        else:
            cams_to_export = self.cameras_to_export

        for cam_key in cams_to_export:

            cam_key = str(cam_key)
            big_cam = self.big_cams.cam_dict[cam_key]

            single_cam_dict = {}

            # Dannce camera formats
            # Convert to rotation matrix
            single_cam_dict['R'] = big_cam.R#.T # (3, 3) # here the matrix is somehow transposed to opencv
            single_cam_dict['T'] = big_cam.t.squeeze() #(3,)


            #Scale the intrinsics of the camera...
            K_unscaled = copy.deepcopy(big_cam.K)

            K_unscaled[:2, :] = K_unscaled[:2, :] / self.export_scale_reduce

            single_cam_dict['K'] = K_unscaled#.T # (3, 3) # here the matrix is somehow transposed to opencv



            # Just for debugging
            single_cam_dict['id'] = cam_key


            self.export_cameras_small.append(copy.deepcopy(single_cam_dict))

            # Create the undistort map
            w, h = (big_cam.im_size) #// self.export_scale_reduce

            ## Exec is 15ms, compared to 40ms for cv2 undist

            # Initialize the rectify map
            map_x, map_y = cv2.initUndistortRectifyMap(big_cam.K, big_cam.d, None, big_cam.K,
                                                     (w, h), cv2.CV_32FC1)

            single_cam_dict['mapx_large'] = map_x
            single_cam_dict['mapy_large'] = map_y




            w, h = (big_cam.im_size) // self.export_scale_reduce

            # Initialize the rectify map
            map_x, map_y = cv2.initUndistortRectifyMap(big_cam.K, big_cam.d, None, big_cam.K,
                                                       (w, h), cv2.CV_32FC1)


            single_cam_dict['mapx'] = map_x
            single_cam_dict['mapy'] = map_y


            # image size is hardcorded...
            self.export_cameras.append(single_cam_dict)


        with open(join(self.export_explicit, "new_cam.pkl"), 'wb') as outfile:
            pickle.dump(self.export_cameras_small, outfile)



    def generate_tracking_labels(self):

        frames_to_iter = range(self.min_frames_action)

        T = []

        for cam_key_index, cam_key in tqdm(enumerate(self.available_videos.keys())):


            per_cam_results = []

            for frame in frames_to_iter:
                unaligned_frame = frame + int(self.additive_frames_cam[cam_key]) - self.min_frames_action

                per_frame_results = []

                for ind_index, ind in enumerate(self.individuals):
                    first_letter_ind = ind[0]


                    ind_results = self.crop_data_action[cam_key][unaligned_frame][
                        first_letter_ind]

                    if len(ind_results) == 3:
                        continue

                    # Get the cropping information from YOLO, ind_id is the label within the 8 individuals themselves
                    bbox, ind_id, bbox_conf, kps_2d_det, kps_2d_conf = ind_results

                    ### Only append if there is something detected

                    # bbox id is basically the ind index for the given action labels HJL f.ex.
                    per_ind_results = [ind_index]
                    per_ind_results.extend(bbox)


                    # kps2d are from cropped parameters
                    kps2d = np.array(kps_2d_det)

                    kps2d_uncropped = kps2d + bbox[:2]

                    kp2d_conf = np.array(kps_2d_conf)
                    kp2d_conf = np.expand_dims(kps_2d_conf, axis=-1)

                    kps2d_uncropped = np.hstack((kps2d_uncropped, kp2d_conf)).tolist()
                    # match them with hstack
                    # back to list

                    per_ind_results.append(kps2d_uncropped)
                    per_ind_results.append(int(ind_id))
                    per_ind_results.append(bbox_conf)

                    # Cam key cause a little easier for further processing
                    per_ind_results.append(cam_key)

                    per_frame_results.append(per_ind_results)

                per_cam_results.append(per_frame_results)

            T.append(per_cam_results)

        return T

    def proc_cross_view_matching(self):

        ### Interpreting the results of bcomb: index of value is camera in cam ID list, bbox id where no detection is -1
        self.association_matrix = np.zeros(shape=(len(self.individuals), self.min_frames_action,
                                                  len(self.big_cams.cam_dict)))




        # Make the camera parameters in the correct format of multi-view estimator? yes
        # Camera IDS
        ID = list(self.available_videos.keys())

        # Ts which is? IDK
        # List (nb_cameras) of list (nb_frames), list(


        alignment_path = join(self.action_path, "mult_align.pkl")

        if not os.path.exists(alignment_path) or self.force_action_reload:
            T = self.generate_tracking_labels()

            # filling the data works, but the cid and tt[0] are a bit hard to understand, more research necessary


                # Per camera it is frames, detected individuals


            # ID of2dtrack
            # framenumbers
            n_cam = len(ID)
            n_frame = len(T[0])


            # Iterate sparse frames
            result_keyframe = []
            bcomb_prev = []
            #for i_frame in tqdm(range(1, n_frame - 12, 12)):
            for i_frame in tqdm(range(n_frame)):
                info_dict = {}
                for i_cam in range(n_cam):

                    img = []

                    TT = T[i_cam][i_frame]

                    P = []
                    for tt in TT:
                        # tt[6] is predicted individual


                        bbox_id = [i_cam,
                                   tt[0]]  # first the camera id, tt[0] is track id, tt[6] pred label, tt[7] pred_score
                        bbox = tt[1:5]  # bbox coordinates
                        #cid = Cid[i_cam][tt[0]][i_frame]  # temporally smoothed individual ID label
                        cid = -1 #tt[0] #TEST with -1 and tt[0] for differences
                        pose2d_raw = np.array(tt[5])  # pose2d


                        cam_key = tt[8]

                        # Undistort the points
                        cam = self.big_cams.cam_dict[cam_key]

                        pts = np.ascontiguousarray(pose2d_raw[:, :2], dtype=np.float64)
                        pose2d = cv2.undistortPoints(pts, cam.K, cam.d)
                        pose2d = pose2d.squeeze()
                        #pose2d = undistort_points(config_path, i_cam, pose2d_raw[:, :2])  # pose2d undistorted

                        # 'cid' becomes -1

                        P.append({'pose2d': pose2d, 'pose2d_raw': pose2d_raw, 'bbox': bbox, 'bbox_id': bbox_id, 'cid': cid})

                    info_dict[i_cam] = {'image_data': img, 0: P}

                # Build the camera parameters

                camparam = {}

                # Just appending across ID
                K = []
                #xi = []
                D = []
                rvecs = []
                tvecs = []
                pmat = []
                camera_ids = []

                for cam_id_index, cam_id in enumerate(ID):

                    cam = self.big_cams.cam_dict[cam_id]

                    K.append(cam.K)
                    D.append(cam.d)

                    rvecs.append(cam.r)
                    tvecs.append(cam.t)

                    rmatx, _ = cv2.Rodrigues(cam.r)
                    P = np.hstack([rmatx, cam.t])

                    pmat.append(P)

                    camera_ids.append(cam_id)

                camparam['K'] = K
                camparam['D'] = D
                camparam['rvecs'] = rvecs
                camparam['tvecs'] = tvecs
                camparam['pmat'] = pmat
                camparam['camera_id'] = camera_ids


                # Per frame basically
                matched_list, pose3d, bcomb = self.multiview_estimator.predict_data(info_dict, show=False, plt_id=0,
                                                                                    camparam=camparam,
                                                                      bcomb_prev=bcomb_prev)

                bcomb_prev = bcomb

                result_keyframe.append({'frame': i_frame, 'bcomb': bcomb, 'pose3d': pose3d})

            with open(alignment_path, "wb") as f:
                pickle.dump(result_keyframe, f)

        else:
            with open(alignment_path, "rb") as f:
                result_keyframe = pickle.load(f)




        self.majority_association_matrix = np.zeros_like(self.association_matrix)

        for i_frame in tqdm(range(self.min_frames_action)):

            bcomb = result_keyframe[i_frame]['bcomb']

            # Ind index is the individual in the matrix
            for ind_index, ind in enumerate(self.individuals):


                size_ids_bcomb = []

                for m_tracklet_index, m_tracklet in enumerate(bcomb):

                    # Get the individual in the mtracklet
                    # If more than min_cams accept it

                    # cameras where this individual appears
                    cam_idx = np.where(m_tracklet == ind_index)[0]

                    size_ids_bcomb.append(cam_idx.size)

                    # check if enough cameras agree
                    if cam_idx.size >= self.min_cams_cross_view:
                        self.association_matrix[ind_index, i_frame, cam_idx] = 1

                size_ids_bcomb = np.array(size_ids_bcomb)
                max_tracklet_per_ind = np.argmax(size_ids_bcomb)
                max_tracklet = bcomb[max_tracklet_per_ind]
                cam_idx = np.where(max_tracklet == ind_index)[0]
                self.majority_association_matrix[ind_index, i_frame, cam_idx] = 1
                # Get the ids and save them





