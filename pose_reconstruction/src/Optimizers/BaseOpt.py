import copy
import os
from os.path import join

import pytorch3d.structures
import torch
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage._filters import _gaussian_kernel1d
import json

from pose_reconstruction.src.utils.losses import (silhouette_loss, scaling_loss, cage_loss,
                              calculate_keypoints3D_from_posed_mesh, keypoint_loss_2D, speed_lim_loss,
                              mean_euclidean_distance, normal_consistency_loss, compute_edge_loss,
                              return_labeled_rig_joints, compute_laplacian, silhouette_segm_loss)
import pose_reconstruction.src.utils.losses as losses
from pose_reconstruction.src.utils.rendering import create_silhouette_renderer, get_rgb_renderer
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, RasterizationSettings
from pytorch3d.renderer import PointLights, AmbientLights
import matplotlib.pyplot as plt
from pose_reconstruction.src.utils.visualize import Visualizer, LossMonitor
from pose_reconstruction.src.utils.LBS import MeshModel
from scipy.spatial.transform import Rotation as R
from pose_reconstruction.src.utils.data_loader import ActionLoader
from pose_reconstruction.src.utils import eval_utils_cropped
import pose_reconstruction.src.utils.utils as utils
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from pytorch3d.loss import mesh_laplacian_smoothing
import einops
import pose_reconstruction.src.utils.camera_calibration as cam_calib
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer import SoftSilhouetteShader, SoftPhongShader

LEARNING_RATE = 0.05


# NUM_CAMS = 16

def find_second_underscore(s):
    # Find the position of the first underscore
    first = s.find('_')

    # If there is no underscore at all, return -1
    if first == -1:
        return -1

    # Find the position of the second underscore, starting the search after the first underscore
    second = s.find('_', first + 1)

    return second


def extract_frame(video_path, frame_number):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Set the position of the next frame to be read
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame.")
        cap.release()
        return
    else:
        cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



ALL_JOINTS = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'HeadEnd', 'RightEye', 'LeftEye', 'Nose',
              'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandEnd', 'LeftHandThumb1', 'LeftHandThumb2',
              'LeftHandThumb3', 'LeftHandThumb4', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3',
              'LeftHandIndex4', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4',
              'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4', 'LeftHandPinky1',
              'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4', 'RightShoulder', 'RightArm',
              'RightForeArm', 'RightHand', 'RightHandEnd', 'RightHandThumb1', 'RightHandThumb2',
              'RightHandThumb3', 'RightHandThumb4', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3',
              'RightHandIndex4', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4',
              'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4', 'RightHandPinky1',
              'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
              'LeftToeBase', 'LeftFootThumb1', 'LeftFootThumb2', 'LeftFootThumb3', 'LeftFootThumb4',
              'LeftFootIndex1', 'LeftFootIndex2', 'LeftFootIndex3', 'LeftFootIndex4', 'LeftFootMiddle1',
              'LeftFootMiddle2', 'LeftFootMiddle3', 'LeftFootMiddle4', 'LeftFootRing1', 'LeftFootRing2',
              'LeftFootRing3', 'LeftFootRing4', 'LeftFootPinky1', 'LeftFootPinky2', 'LeftFootPinky3',
              'LeftFootPinky4', 'Tail', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'RightUpLeg',
              'RightLeg', 'RightFoot', 'RightToeBase', 'RightFootThumb1', 'RightFootThumb2',
              'RightFootThumb3', 'RightFootThumb4', 'RightFootIndex1', 'RightFootIndex2',
              'RightFootIndex3', 'RightFootIndex4', 'RightFootMiddle1', 'RightFootMiddle2',
              'RightFootMiddle3', 'RightFootMiddle4', 'RightFootRing1', 'RightFootRing2',
              'RightFootRing3', 'RightFootRing4', 'RightFootPinky1', 'RightFootPinky2',
              'RightFootPinky3', 'RightFootPinky4']

JOINTS_TO_FALL_IN_MASK = ['LeftHand', 'LeftHandEnd', 'LeftHandThumb1', 'LeftHandThumb2',
              'LeftHandThumb3', 'LeftHandThumb4', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3',
              'LeftHandIndex4', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4',
              'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4', 'LeftHandPinky1',
              'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4',
                          'RightHand','RightHandEnd', 'RightHandThumb1', 'RightHandThumb2',
                          'RightHandThumb3', 'RightHandThumb4', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3',
                          'RightHandIndex4', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3',
                          'RightHandMiddle4',
                          'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4', 'RightHandPinky1',
                          'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4',
'LeftFoot',
              'LeftToeBase', 'LeftFootThumb1', 'LeftFootThumb2', 'LeftFootThumb3', 'LeftFootThumb4',
              'LeftFootIndex1', 'LeftFootIndex2', 'LeftFootIndex3', 'LeftFootIndex4', 'LeftFootMiddle1',
              'LeftFootMiddle2', 'LeftFootMiddle3', 'LeftFootMiddle4', 'LeftFootRing1', 'LeftFootRing2',
              'LeftFootRing3', 'LeftFootRing4', 'LeftFootPinky1', 'LeftFootPinky2', 'LeftFootPinky3',
              'LeftFootPinky4', 'Tail', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5',
'RightFoot', 'RightToeBase', 'RightFootThumb1', 'RightFootThumb2',
              'RightFootThumb3', 'RightFootThumb4', 'RightFootIndex1', 'RightFootIndex2',
              'RightFootIndex3', 'RightFootIndex4', 'RightFootMiddle1', 'RightFootMiddle2',
              'RightFootMiddle3', 'RightFootMiddle4', 'RightFootRing1', 'RightFootRing2',
              'RightFootRing3', 'RightFootRing4', 'RightFootPinky1', 'RightFootPinky2',
              'RightFootPinky3', 'RightFootPinky4'
                          ]





BONE_LENGTHS_TO_OPT = ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
                       'LeftShoulder', 'LeftArm', 'LeftForeArm', 'RightShoulder',
                       'RightArm', 'RightForeArm', 'LeftUpLeg', 'LeftLeg',
                       'Tail', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5',
                       'RightUpLeg', 'RightLeg']
BONE_LENGTHS_TO_OPT = ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
                       'LeftShoulder', 'LeftArm', 'LeftForeArm', 'RightShoulder',
                       'RightArm', 'RightForeArm', 'RightHand', 'LeftHand', 'RightFoot', 'LeftFoot', 'LeftUpLeg',
                       'LeftLeg',
                       'Tail', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5',
                       'RightUpLeg', 'RightLeg']

BODY_POSE_TO_OPT = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'LeftShoulder',
                    'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder',
                    'RightArm', 'RightForeArm', 'RightHand', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
                    'Tail', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5',
                    'RightUpLeg', 'RightLeg', 'RightFoot']

Hierachy_mapping = ['RightEye', 'LeftEye', 'Nose', 'Neck',
                    'RightArm', 'RightForeArm', 'RightHand',
                    'LeftArm', 'LeftForeArm', 'LeftHand',
                    'Hips', 'RightLeg', 'RightFoot',
                    'LeftLeg', 'LeftFoot',  'Tail5']


Hierachy_mapping = ['RightEye', 'LeftEye', 'Nose', 'Neck',
                    'RightArm', 'RightForeArm', 'RightHand', 'RightHandMiddle4',
                    'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandMiddle4',
                    'Hips', 'RightLeg', 'RightFoot', 'RightFootMiddle4',
                    'LeftLeg', 'LeftFoot', 'LeftFootMiddle4', 'Tail5']


COUPLED_JOINTS = [[1, 2, 3, 4], [18, 19, 20, 21, 22, 23]]


KEYPOINT_WEIGHTING = torch.tensor([2, 2, 2,
                                   3,
                                   2, 2, 2, 2,
                                   2, 2, 2, 2,
                                   5,
                                   2, 2, 2,
                                   2, 2, 2,
                                   4], device='cuda')


## OMS Keypoints

# Joints in order Nose-Head-Neck-RShoulder-
# RHand-LShoulder-LHand-
# Hip-RKnee-RFoot-
# LKnee-LFoot-Tail

# Reduced Joints in order Nose-Neck-RShoulder-
# RHand-LShoulder-LHand-
# Hip-RKnee-RFoot-
# LKnee-LFoot-Tail

KEYPOINT_WEIGHTING_OMS = torch.Tensor([1, 1, 1,
                                       2, 1, 2,
                                       1, 1, 1,
                                       1, 1, 3])

## Keypoints OMC
# REye, LEye, Nose, Head, Neck, Right Shoulder, Right Elbow, Right Wrist, Left Sh, Left El, Left Wr, Hip, RKnee, RAnkle,
# LKnee, LAnkle, Tail
keypoints_names_omc = ["REye", "LEye", "Nose", "Head", "Neck", "RSh", "REl", "RWr",
                       "LSh", "LEl", "LWr", "Hip", "RKn", "RAn", "LKn", "LAn", "Tail"]

remap_list_omc = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # all except Head
## Keypoints OMC - Reduced
# REye, LEye, Nose, Neck, Right Shoulder, Right Elbow, Right Wrist, Left Sh, Left El, Left Wr, Hip, RKnee, RAnkle,
# LKnee, LAnkle, Tail
KEYPOINT_WEIGHTING_OMC = torch.Tensor(
    [3, 3, 2, 1,  # reduced neck, increased eyes, because neck is sometimes wrong, eyes for orientation
     2, 2, 3,
     2, 2, 3,
     5, 2, 3,
     2, 3, 4])

## Keypoints OAP - Order changes to OMC: Nose comes first, and sides are flipped
# Nose, LEye, REye, Head, Neck, Left Shoulder, Left Elbow, Left Wrist, Right Sh, Right El, Right Wr, Hip, LeftKnee, LeftAnkle,
# RKnee, RAnkle, Tail
keypoints_names_oap = ["Nose", "LEye", "REye", "Head", "Neck", "LSh", "LEl", "LWr",
                       "RSh", "REl", "RWr", "Hip", "LKn", "LAn", "RKn", "RAn", "Tail"]

# Keypoints OAP - Revised Order
# REye, LEye, Nose, Neck, Right Shoulder, Right Elbow, Right Wrist, Left Sh, Left El, Left Wr, Hip, RKnee, RAnkle,
# LKnee, LAnkle

# Position indicates new location, value old index
remap_list_oap = [2, 1, 0, 4, 8, 9, 10, 5, 6, 7, 11, 14, 15, 12, 13]
KEYPOINT_WEIGHTING_OAP = KEYPOINT_WEIGHTING_OMC[:-1]


BODY_POSE_RESTRICTED_JOINTS = ['LeftShoulder', 'RightShoulder', 'LeftArm', 'RightArm',
                               'LeftUpLeg', 'LeftLeg', 'RightUpLeg', 'RightLeg']

NO_TAIL_SPECIES = ['Barbary_macaque', 'Bonobo', 'Chimpanzee',
                   'Gibbon', 'Gorilla', 'Japanese_macaque',
                   'Mandrill', 'Orangutan',
                   'Siamang']


class BaseOptimizer():
    def __init__(self, device, mesh_model: MeshModel, action_loader: ActionLoader, cameras_to_retain, frame=0,
                 sparse=True, high_poly=False,
                 infer_from_rig=True, save_renderings=True, load_init_tensors=True, verbose=1,
                 save_iters_as_plots=False,
                 OMS=False, CROP=False, single_view_dataset=None, species=None, pose_prior=None,
                 scale_hands_feet=False, weighted_sil_loss=True, step_scheduler=True, template_stretch=True):

        self.mod_iter_show = 5

        self.action_and_keyframes = []
        self.action_frame_index = []
        self.action_frame_index_single = 0

        # Stretching the template for fitting
        self.template_stretch = template_stretch

        self.device = device
        self.mesh_model = mesh_model
        self.action_loader = action_loader
        self.cameras_to_retain = cameras_to_retain

        self.NUM_CAMS = self.cameras_to_retain.__len__()

        self.species = "Rhesus_macaque"
        self.load_init_tensors = load_init_tensors

        # OMS case
        self.OMS = OMS
        self.CROP = CROP

        self.scale_hands_feet = scale_hands_feet

        if pose_prior is not None:
            # Cast this to a tensor on device
            self.pose_prior = torch.from_numpy(pose_prior).float().to(self.device)


        ## Single Species Case
        if single_view_dataset is not None:
            self.single_view = True
            self.single_view_dataset = single_view_dataset
            self.species = species

            # Covariance, and mean of pose prior
            self.pose_prior_mean = None
            self.pose_prior_covI = None

        else:
            self.single_view = False

        self.model_params_flags = {}


        ## Environment data
        self.lights = None
        self.silhouette_renderer = None
        self.display_renderer = None

        if self.NUM_CAMS > 4:
            self.num_views = 4
        else:
            self.num_views = self.NUM_CAMS

        self.cams_bigmac = {}
        self.cameras = {}

        self.reduction_sampling_image_size = 1
        # Path to save the mesh
        self.mesh_path = None

        # The current individual
        self.cur_ind_index = 0
        self.cur_ind = None

        self.rgbs = {}

        self.model_parameters = {}

        # General
        self.high_poly = high_poly
        # change
        self.start_frame = frame
        self.frame = frame

        self.bundled_frames = []


        # Optmization visualizer
        # Visualizer
        self.vis = Visualizer(self.num_views)

        # Loss monitor
        self.loss_monitor = LossMonitor()

        self.verbose = verbose

        # Optimization parameters
        self.global_orientation = 0
        self.global_t = 0
        self.global_scale = 0
        self.body_pose = 0
        self.bone_length = 0
        self.vertex_offsets = 0

        if self.single_view:
            self.focal_lengths = 0

        # Sparsification parameters
        self.bone_length_sparse = None
        self.body_pose_sparse = None
        self.vertex_sparse = None
        self.body_pose_pre_restriction = None
        self.body_pose_sparse_P_roll = None
        self.restricted_index_in_to_opt = None

        self.sparse = sparse
        self.infer_from_rig = infer_from_rig

        # Sparsification type:
        self.vertex_normals_flag = False



        ### Optmization specific

        # 0 == global pos, 1== posing, 2==mesh, 3==time
        #self.opt_state = 0
        #self.opt_state_titles = ["Global", "Posing", "Mesh", "Time"]

        self.stage = "Global"

        self.epoch = -1

        self.iters_total = 0
        self.iter = 0
        self.steps = 0

        self.save_mesh = False


        self.alphas = {}

        # Threshold for the silhouette loss to have effect
        self.keypoint_thresh = 0
        self.momentum_window = 10

        # Previous body pose
        self.prev_body_pose = None
        self.prev_posed_3d_kps = []

        ## Intermediate visualization
        self.plot_skeleton = False
        self.plot_3d = False
        self.save_renderings = save_renderings
        self.save_iters_as_plots = save_iters_as_plots

        self.smooth = False
        self.initial_frame = frame

        self.image_preds = []
        self.gt_masks = []

        # Mesh_save_path
        self.save_mesh_template = None


        ## For mesh posing saving
        self.body_pose_LBS = self.bone_lengths_LBS = self.vertex_weights = 0

        ## Intermediate saved variables during optimization
        self.posed_verts = self.posed_mesh = self.rest_pose_shifted_V = self.joints_coordinates_posed \
            = self.posed_3d_keypoints = 0

        # Large deformations flag
        self.large_shape_deform = False

        ### Pytorch specific
        self.body_optimizer = None
        self.loss = torch.tensor([0], device=self.device)

        self.lr = LEARNING_RATE
        self.scheduler = None

        ### Evaluation specific
        self.IoUs = {}
        # Euclidean distance, with frame as key, and mean euclidean distance
        self.euclid_three_d = {}

        self.state_statistics = {}

        # For the entire action
        self.state_statistics_action = {}

        ## Render settings
        # Renderer settings
        self.faces_per_pixel = 90#10#90
        # Path for rendering
        self.render_path = None
        self.render_path_base = None

        # Det confidences
        self.use_kp_conf = False
        self.use_bbox_conf = False

        # Scheduler selectionstep_scheduler
        self.step_scheduler = step_scheduler


        # Selection scalar
        self.selection_scalar = 0.1

        # Angle weighting (for over time adjusted, because more info)
        HAND_ROT_WEIGHT = 1  # 8
        TAIL_ROT_WEIGHT = 1 / 4
        # HAND_ROT_WEIGHT = TAIL_ROT_WEIGHT

        self.ANGLE_WEIGHTING = torch.tensor([1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, HAND_ROT_WEIGHT, 1,
                                        1, 1, HAND_ROT_WEIGHT, 1, 1, HAND_ROT_WEIGHT,
                                        TAIL_ROT_WEIGHT, TAIL_ROT_WEIGHT, TAIL_ROT_WEIGHT, TAIL_ROT_WEIGHT,
                                        TAIL_ROT_WEIGHT, TAIL_ROT_WEIGHT,
                                        1, 1, HAND_ROT_WEIGHT], device=self.device)

        self.KEYPOINT_WEIGHTING = KEYPOINT_WEIGHTING

        self.undistort_maps = {}

        ## End effector indices
        end_effectors = [10, 14, 17, 26]
        end_effectors_indices = [i * 3 + j for i in end_effectors for j in range(3)]
        self.end_effector_indices = torch.tensor(end_effectors_indices,
                                            dtype=torch.long, device=self.device)



        ### Weight the color map


        tail_color_weight = 3  # 10
        hands_color_weight = 2  # 10
        extremeties_weight = 1  # 2
        head_related_weights = 1
        torso_related_weights = 1


        self.color_weights = {'Head': head_related_weights,
                              'LeftArm':extremeties_weight,
                              'LeftEar':head_related_weights,
                              'LeftFoot':hands_color_weight,
                              'LeftHand':hands_color_weight,
                              'LeftLeg':extremeties_weight,
                              'RightArm': extremeties_weight,
                              'RightEar': head_related_weights,
                              'RightFoot': hands_color_weight,
                              'RightHand': hands_color_weight,
                              'RightLeg': extremeties_weight,
                              'Tail': tail_color_weight,
                              'Torso': torso_related_weights,
                              }
        # The weighted segmentation is not available for low poly yet
        #if self.action_loader.high_poly:
        #    self.mesh_model.set_weight_texture(self.color_weights)

        # Silhouette weighted loss
        self.weighted_sil_loss = weighted_sil_loss


        # Cameras and Renderers
        self.silhouette_renderer = {}
        self.display_renderer = {}
        self.cams_bigmac = {}
        self.cameras = {}


        # Conveniently add the joints to the Optimizer, and other things to the optimizer
        self.ALL_JOINTS = ALL_JOINTS
        self.BONE_LENGTHS_TO_OPT = BONE_LENGTHS_TO_OPT
        self.BODY_POSE_TO_OPT = BODY_POSE_TO_OPT
        self.HIERARCHY_MAPPING = Hierachy_mapping

        # Render the bounding box
        self.render_bbox = True
        self.render_sil = False
        self.render_default_col = False
        self.render_raw_rgb = False
        self.render_mesh_wo_bg = False
        self.render_bright = True

        # Load the joint limits
        self._load_joint_limits()


    def _load_joint_limits(self):

        print(os.getcwd())

        with open('../data/Mesh/joint_limits.json') as f:
            joint_limits = json.load(f)['JOINT_LIMITS']

        lower, upper = [], []
        for idx, joint in enumerate(self.BODY_POSE_TO_OPT):
            if not joint in joint_limits.keys():
                continue
            lower.append(joint_limits[joint][0])
            upper.append(joint_limits[joint][1])
        self.joints_lower_bound = torch.tensor(lower, device=self.device).view(-1).unsqueeze(0) # (B, J, 3)
        self.joints_upper_bound = torch.tensor(upper, device=self.device).view(-1).unsqueeze(0) # (B, J, 3)

        return None

    def save_loss_logs(self, loss_dict):

        loss_row = f"{self.epoch}, {self.frame}, losses: "
        for alpha_key, alpha_val in self.alphas.items():
            if alpha_val != 0:
                loss_row += f"{alpha_key}: {loss_dict[alpha_key]} "

        with open(join(self.render_path_base, "logs.txt"), 'a') as f:
            f.write(loss_row)
            f.write('\n')


    def __destroy_figures__(self):

        # Close all the figures that have been created for this object.
        plt.close('all')

    def update_save_paths(self):

        self.save_opt_path = join(self.action_loader.action_path, "OPT_params")

        if not os.path.exists(self.save_opt_path):
            os.makedirs(self.save_opt_path)

        self.model_init_pose_save_path = join(self.save_opt_path,
                                              f"model_init_pose_{self.frame}_{self.cur_ind}_{self.NUM_CAMS}_{self.action_loader.labels_only}.pth")
        self.model_init_bones_save_path = join(self.save_opt_path,
                                               f"model_init_bones_{self.frame}_{self.cur_ind}_{self.NUM_CAMS}_{self.action_loader.labels_only}.pth")
        self.model_init_mesh_save_path = join(self.save_opt_path,
                                              f"model_init_mesh_{self.frame}_{self.cur_ind}_{self.NUM_CAMS}_{self.action_loader.labels_only}.pth")
        self.model_time_opt_save_path = join(self.save_opt_path,
                                             f"model_time_{self.NUM_CAMS}_{self.cur_ind}_{self.action_loader.labels_only}.pth")

    def load_rgb_labeled_images(self):

        rgb_over_cams = {}

        for cam_key in self.cameras.keys():
            # cam path

            cam_path = join(join(self.action_loader.action_path, "labeled_images"), f"cam{cam_key}")

            img_path = join(cam_path, f"{self.frame}.png")

            image = Image.open(img_path)

            # In (W, H) opposite to arrays
            cur_size = image.size

            resized_img = image.resize((int(cur_size[0] * self.reduction_sampling_image_size),
                                        int(cur_size[1] * self.reduction_sampling_image_size)))

            rgb_over_cams[cam_key] = np.array(resized_img)

        self.rgbs = rgb_over_cams



    def init_transl_rot(self):

        if self.single_view:
            self.global_t = torch.tensor([[0, 0, 0]], device=self.device)
        else:
            # A decent starting position in the cage
            self.global_t = torch.tensor([[0, 0, -0.75]], device=self.device)


        self.global_t = einops.repeat(self.global_t, 'm n -> k m n',
                                      k=len(self.action_and_keyframes)).clone().detach().requires_grad_(True)

        # Define the global orientation
        #global_orientation = R.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).as_rotvec()

        if self.single_view:
            # Rotating mesh side-view
            global_orientation = np.array([0, 1.5708, 0])

            # Facing the camera
            global_orientation = np.array([0, np.pi, 0])
        else:
            # looking at the hind, for procrustes keep the initial global rotation the identity
            global_orientation = R.from_matrix(np.eye(3)).as_rotvec()
            #global_orientation = np.array([0, np.pi, 0])

        global_orientation = np.expand_dims(global_orientation, axis=-1).T
        global_orientation = torch.from_numpy(global_orientation)
        self.global_orientation = global_orientation.to(self.device)
        self.global_orientation = einops.repeat(self.global_orientation, 'm n -> k m n',
                                                k=len(self.action_and_keyframes)).clone().detach().requires_grad_(True)

    def init_model_params(self, save=True):

        self.body_pose, self.bone_length = self.mesh_model.get_simple_pose()


        self.body_pose = einops.repeat(self.body_pose, 'm n -> k m n', k=len(self.action_and_keyframes)).clone().detach().requires_grad_(True)
        self.init_transl_rot()

        if self.single_view:
            self.global_scale = torch.tensor([1], device=self.device)
        else:
            self.global_scale = torch.tensor([1.0], device=self.device)

        # Sparsify the optimization, and also init vertex offsets etc.
        if self.sparse:
            self.init_sparsification(save)

        if self.single_view:
            #self.focal_lengths = torch.Tensor([[800, 800]]).to(self.device)
            self.focal_lengths = torch.Tensor([[200, 200]]).to(self.device)
            self.focal_lengths = einops.repeat(self.focal_lengths, pattern='m n -> k m n', k=len(self.action_and_keyframes)).clone().detach().requires_grad_(True)

        self.vertex_colors = torch.zeros(size=(1, self.mesh_model.V.shape[1], 3)).to(self.device)

    def init_sparsification(self, save=True):

        self.sparsify_bone_parameters()
        self.sparsify_body_pose()
        self.sparsify_vertex_parameters()

        if save:
            # Save the mapping
            self.save_sparsification_pose()


    def save_sparsification_pose(self):

        pose_transform_dict = {}

        pose_transform_dict["ALL_JOINTS"] = ALL_JOINTS
        pose_transform_dict["BODY_POSE_TO_OPT"] = BODY_POSE_TO_OPT
        pose_transform_dict["body_pose_sparse_mapping"] = self.body_pose_sparse.clone().tolist()

        with open(join(self.action_loader.action_path, "pose_map.json"), 'w') as f:
            json.dump(pose_transform_dict, f)


    def sparsify_bone_parameters(self):

        # Sparse mappings
        self.bone_length_sparse = torch.zeros(self.mesh_model.get_nb_bones()).to(self.device)
        self.bone_length_tail_vec = torch.ones_like(self.bone_length_sparse).to(self.device)
        self.bone_length_sparse_hf = torch.zeros_like(self.bone_length_sparse).to(self.device)

        # Bone length opt parameters
        if not self.scale_hands_feet:
            ### Bone length sparsification
            self.bone_length = 1 * torch.ones(size=(1, len(BONE_LENGTHS_TO_OPT))).to(self.device)
        else:
            self.bone_length = 1 * torch.ones(size=(1, len(BONE_LENGTHS_TO_OPT) + 1)).to(self.device)

        for joint_key, joint_index in self.mesh_model.dd['joint_to_index'].items():

            if 'Hand' in joint_key or 'Foot' in joint_key:  ## added
                self.bone_length_sparse_hf[joint_index - 1] = 1

            # Iterate over joint names
            for index_red_joint, joint_name in enumerate(BONE_LENGTHS_TO_OPT):
                if joint_name == joint_key:
                    # Get the index of that joint
                    # Row is the index of the bone length that is used in lbs, index red joint, is the corresponding optimization value
                    assert joint_index > 0
                    self.bone_length_sparse[joint_index - 1] = 1
                    # To multiply the bone lengths pre-posing
                    if self.species in NO_TAIL_SPECIES:
                        if 'Tail' in joint_name:
                            self.bone_length_tail_vec[joint_index - 1] = 0

    def sparsify_body_pose(self, rot_restriction=True):

        ### Body Pose sparsification
        if rot_restriction:

            body_pose = torch.zeros(size=(len(self.action_and_keyframes), 1, 3 * (len(BODY_POSE_TO_OPT)))).to(self.device)

            if self.template_stretch:

                ### Tail
                tail_index = 18
                axis = 0
                body_pose[:, 0, tail_index*3 + axis] = torch.pi / 2

                ### For labels only as static frames the endeffectors are kept at the initial position,
                # as the time constraints help a lot to solve the hand rotation
                if not self.action_loader.labels_only:
                    end_effectors = [10, 14, 17, 26]

                    for end_ef in end_effectors:
                        body_pose[:, 0, end_ef*3 + axis] = torch.pi / 2


            body_pose_sparse_P = torch.zeros(self.mesh_model.J.shape[1]).to(self.device)

            body_pose_sparse_P_roll = torch.zeros(self.mesh_model.J.shape[1]).to(self.device)
            restricted_index_in_to_opt = torch.zeros(len(BODY_POSE_TO_OPT)).to(self.device)

            for joint_key, joint_index in self.mesh_model.dd['joint_to_index'].items():
                # Iterate over joint names
                for index_red_joint, joint_name in enumerate(BODY_POSE_TO_OPT):
                    if joint_name == joint_key:

                        body_pose_sparse_P[joint_index] = 1

                        # Restrict the roll component
                        if joint_name in BODY_POSE_RESTRICTED_JOINTS:
                            body_pose_sparse_P_roll[joint_index] = 1

                            restricted_index_in_to_opt[index_red_joint] = 1

            self.body_pose_sparse_P_roll = body_pose_sparse_P_roll
            self.restricted_index_in_to_opt = restricted_index_in_to_opt

        else:
            body_pose = torch.zeros(size=(len(self.action_and_keyframes), 1, 3 * len(BODY_POSE_TO_OPT))).to(self.device)
            body_pose_sparse_P = torch.zeros(size=(3 * self.mesh_model.J.shape[1],
                                                   3 * len(BODY_POSE_TO_OPT))).to(self.device)

            # The rest stays zero, where there should not be a pose, so only iter over the ones to pose

            # Iterate over joint names
            for index_red_joint, joint_name in enumerate(BODY_POSE_TO_OPT):
                joint_name_index = self.mesh_model.dd['joint_to_index'][joint_name]

                # Where
                for rot_index in range(3):
                    body_pose_sparse_P[joint_name_index + rot_index, index_red_joint + rot_index] = 1

        # Assign
        self.body_pose = body_pose
        self.body_pose_sparse = body_pose_sparse_P

    def sparsify_vertex_parameters(self):

        ## Init vertex normal_weights_sparse

        if self.action_loader.vertex_info:

            """
            Example:
            extended_vertex_info = [[0, 1], [3, 4], [2]]
            # Create a sparse vector to learn weights for vertex normals
            vertex_weights_sparse = torch.zeros(len(extended_vertex_info), 1)
            # Create the mapping of the sparse_vertex_vector to all the vertex weights
            sparse_map_A = torch.zeros((5, vertex_weights_sparse.shape[0]))
            # Iterate the pairs
            for sparse_index, pair in enumerate(extended_vertex_info):
                sparse_map_A[int(pair[0]), sparse_index] = 1
                if len(pair) == 2:
                    sparse_map_A[int(pair[1]), sparse_index] = 1
            """

            if self.vertex_normals_flag:

                # Expand the info into a single list of lists
                extended_vertex_info = self.action_loader.vertex_info["pairs"]
                extended_vertex_info.extend([[x] for x in self.action_loader.vertex_info["no_partner"]])

                # Create a sparse vector to learn weights for vertex normals
                vertex_normal_weights = torch.zeros(len(extended_vertex_info), 1).to(self.device)

                # Create the mapping of the sparse_vertex_vector to all the vertex weights
                sparse_map_A = torch.zeros((self.mesh_model.V_normals.shape[0], vertex_normal_weights.shape[0])).to(
                    self.device)
                # Iterate the pairs
                for sparse_index, pair in enumerate(extended_vertex_info):
                    sparse_map_A[int(pair[0]), sparse_index] = 1
                    if len(pair) == 2:
                        sparse_map_A[int(pair[1]), sparse_index] = 1
                # Check that all rows of that sparse map have an entry
                assert torch.sum(sparse_map_A) == self.mesh_model.V_normals.shape[0]

            else:

                if 'excluded_vertices' in self.action_loader.vertex_info.keys():

                    # Reduced already there?
                    if 'reduced_excluded_vertices' not in self.action_loader.vertex_info.keys():
                        ## Create it
                        # Expand the info into a single list of lists
                        extended_vertex_info = self.action_loader.vertex_info["pairs"]
                        extended_vertex_info.extend([[x] for x in self.action_loader.vertex_info["no_partner"]])
                        extended_vertex_info = [sublist for sublist in extended_vertex_info if
                                                not any(str(element) in sublist for element in
                                                        self.action_loader.vertex_info["excluded_vertices"])]

                        self.action_loader.vertex_info["reduced_excluded_vertices"] = extended_vertex_info

                        vertex_symm_update_path = join(self.action_loader.project_root,
                                                       r"data/Mesh/vertex_info_hands_excluded.json")

                        with open(vertex_symm_update_path, 'w') as json_file:
                            json.dump(self.action_loader.vertex_info, json_file)
                    else:
                        ## Just load it
                        extended_vertex_info = self.action_loader.vertex_info["reduced_excluded_vertices"]
                else:
                    # Expand the info into a single list of lists
                    extended_vertex_info = self.action_loader.vertex_info["pairs"]
                    extended_vertex_info.extend([[x] for x in self.action_loader.vertex_info["no_partner"]])

                ## Vertex offsets
                vertex_normal_weights = torch.zeros(len(extended_vertex_info), 3).to(self.device)

                ## Can not be the same offset because then it would shift in the same direction needs to be -1

                # Create the mapping of the sparse_vertex_vector to all the vertex weights
                sparse_map_A = torch.zeros((self.mesh_model.V_normals.shape[0], vertex_normal_weights.shape[0])).to(
                    self.device)

                # Iterate the pairs
                for sparse_index, pair in enumerate(extended_vertex_info):

                    sparse_map_A[int(pair[0]), sparse_index] = 1
                    if len(pair) == 2:
                        sparse_map_A[int(pair[1]), sparse_index] = -1

                # Iterate the excluded vertices
                # Set the ith row to zero

                if 'excluded_vertices' in self.action_loader.vertex_info.keys():
                    for excluded_vertex in self.action_loader.vertex_info["excluded_vertices"]:
                        sparse_map_A[excluded_vertex, :] = 0

                # Check that all rows of that sparse map have an entry
                # assert torch.sum(torch.abs(sparse_map_A)) == self.mesh_model.V_normals.shape[0]

        else:

            if self.vertex_normals_flag:
                vertex_normal_weights = torch.zeros(self.mesh_model.V_normals.shape[0], 1).to(self.device)


            else:
                ## Vertex offsets
                # Each vertex has offsets in each dimension
                vertex_normal_weights = torch.zeros(self.mesh_model.V_normals.shape[0], 3).to(self.device)

                # Sparse Map A is just diagonal as well

            # Sparse map A is just diagonal
            sparse_map_A = torch.eye(self.mesh_model.V_normals.shape[0]).to(self.device)

        ## Assign
        self.vertex_offsets = vertex_normal_weights
        self.vertex_sparse = sparse_map_A

    def update_opt_params(self, lr=None):

        # Update optimization parameters
        self.alphas = self.params_dict["alphas"]
        #self.steps = self.params_dict["steps"]
        #self.lr = self.params_dict["LR"]


        self.action_loader.max_image_size

        # If we have a maximum image size, use it, otherwise use T unscaled
        if self.action_loader.max_image_size is not None:
            self.keypoint_thresh = self.params_dict["T"] * (self.action_loader.max_image_size / 250)#
        else:
            self.keypoint_thresh = self.params_dict["T"]


        if lr is not None:
            self.lr = lr

    def retrieve_model_params_time_copy(self):

        if self.single_view:
            model_params_copy = [self.global_orientation.clone().detach(), self.global_t.clone().detach(),
                                 self.global_scale.clone().detach(),
                                 self.body_pose.clone().detach(), self.bone_length.clone().detach(),
                                 self.vertex_offsets.clone().detach(),
                                 self.posed_3d_keypoints.clone().detach(), self.focal_lengths.clone().detach()]
        else:
            model_params_copy = [self.global_orientation.clone().detach(), self.global_t.clone().detach(),
                                 self.global_scale.clone().detach(),
                                 self.body_pose.clone().detach(), self.bone_length.clone().detach(),
                                 self.vertex_offsets.clone().detach(),
                                 self.posed_3d_keypoints.clone().detach()]

        return model_params_copy





    def create_renderers(self):
        """
        Creates renderers on the fly
        :return:
        """
        # This shouldn't be necessary with the frame cam key
        #self.silhouette_renderer = {}
        #self.display_renderer = {}
        #self.cams_bigmac = {}
        #self.cameras = {}

        boundaries_cameras = self.action_loader.big_cams.get_camera_boundaries()
        #self.lights = PointLights(device=self.device, location=torch.from_numpy(boundaries_cameras[0, :][None]))
        self.lights = AmbientLights(device=self.device)


        if self.action_loader.same_size:
            ind_frame = self.bundled_frames[0]
            ind_cam = self.cameras_to_retain[0][0]

            cam = self.action_loader.pytorch_cameras[ind_frame][ind_cam][self.cur_ind]

            big_cam = self.action_loader.big_cams.cam_dict[ind_cam]

            if self.action_loader.load_undist_rgb:
                im_h, im_w = cam.image_size.squeeze()
            else:
                im_w, im_h = big_cam.im_size // self.image_scale

            sil_render_cam, display_renderer = create_silhouette_renderer(cam, lights=self.lights,
                                                                          device=self.device,
                                                                          image_size=(
                                                                              int(im_h), int(im_w)),
                                                                          faces_per_pixel=self.faces_per_pixel,
                                                                          display_img_size=(
                                                                              int(im_h), int(im_w)),
                                                                          body_part_seg=self.weighted_sil_loss)




        for frame_index, frame in enumerate(self.bundled_frames):


            cameras_to_iter = self.action_loader.pytorch_cameras[frame].items()

            for cam_key, cam_inds in cameras_to_iter:

                if self.action_loader.labels_only:
                    # There is no frame specific camera selection process
                    cameras_to_retain = self.cameras_to_retain
                else:
                    cameras_to_retain = self.cameras_to_retain[frame_index]

                if cam_key in cameras_to_retain:
                    cam = cam_inds[self.cur_ind]

                    big_cam = self.action_loader.big_cams.cam_dict[cam_key]

                    if self.action_loader.load_undist_rgb:
                        im_h, im_w = cam.image_size.squeeze()
                    else:
                        im_w, im_h = big_cam.im_size // self.image_scale


                    if not self.action_loader.same_size:
                        # Set Up the Renderer
                        sil_render_cam, display_renderer = create_silhouette_renderer(cam, lights=self.lights,
                                                                                      device=self.device,
                                                                                      image_size=(
                                                                                          int(im_h), int(im_w)),
                                                                                      faces_per_pixel=self.faces_per_pixel,
                                                                                      display_img_size=(
                                                                                          int(im_h), int(im_w)),
                                                                                      body_part_seg=self.weighted_sil_loss)

                    frame_cam_key = f"{frame}_{cam_key}"
                    self.silhouette_renderer[frame_cam_key] = sil_render_cam
                    # Display renderer uses ambient lights for now
                    self.display_renderer[frame_cam_key] = display_renderer
                    # Add the big cam
                    self.cams_bigmac[frame_cam_key] = big_cam
                    self.cameras[frame_cam_key] = cam




    def check_frame_ind_labels(self):
        """
        Check for missing labels for a frame and individual
        :return:
        """
        skip_flag = False
        for cam_key, cam_inds in self.action_loader.pytorch_cameras[self.frame].items():
            if cam_key in self.cameras_to_retain:
                if cam_inds[self.cur_ind] is not None:
                    pass
                else:
                    skip_flag = True
        return skip_flag

    def create_env_data(self):

        ## Lights
        boundaries_cameras = self.action_loader.big_cams.get_camera_boundaries()
        self.lights = PointLights(device=self.device, location=torch.from_numpy(boundaries_cameras))

        self.create_mesh_save_path()

        ## Renderer, PyCams, and BigCams
        self.create_renderers()

    def read_json_cfg(self, opt_stage):

        with open(join(self.action_loader.project_root, fr"cfgs/{opt_stage}.json"), 'r') as f:
            params_dict = json.load(f)

        return params_dict

    def sequence_optimization_mesh(self, debug_mesh_opt=False, large_shape_deform=False,
                                   mask_increase_factor=1):
        """
        Basic function to run a sequence of mesh optimization with basic parameters
        :param mesh_optimizer:
        :return:
        """

        T = 5 * mask_increase_factor  # 10#20

        print(f"T used for optimization with mask_increase factor {(T, mask_increase_factor)}")

        ### Global optimization stage
        params_dict = self.read_json_cfg("Global")
        self.opt_pose(alphas=params_dict["alphas"], steps=params_dict["steps"], T=T,
                      param_flags=params_dict["param_flags"], lr=params_dict["LR"])

        ### Pose Alignment for a short time
        params_dict = self.read_json_cfg("Pose1")
        self.opt_pose(alphas=params_dict["alphas"], steps=params_dict["steps"], T=T,
                      param_flags=params_dict["param_flags"], lr=params_dict["LR"])

        ### Pose and Bones
        params_dict = self.read_json_cfg("Bones")
        self.opt_bones(alphas=params_dict["alphas"], steps=params_dict["steps"], T=T,
                       param_flags=params_dict["param_flags"], lr=params_dict["LR"])
        # --------------------------------------------
        print(self.bone_length)

        # Parameters for 4 cameras
        # or result of more masks simply?

        ### Mesh optimization
        if large_shape_deform:
            params_dict = self.read_json_cfg("Mesh_Large")
        else:
            params_dict = self.read_json_cfg("Mesh")

        if debug_mesh_opt:
            self.load_init_tensors = not debug_mesh_opt
            # steps = 1
            self.opt_mesh(alphas=params_dict["alphas"], steps=params_dict["steps"], T=T,
                          param_flags=params_dict["param_flags"], lr=params_dict["LR"])
            # mesh_optimizer.opt_mesh(alphas=alphas, steps=steps, T=T, param_flags=[True, True, True, True, True, True],
            #                        save=not debug_mesh_opt)
        else:
            self.opt_mesh(alphas=params_dict["alphas"], steps=params_dict["steps"], T=T,
                          param_flags=params_dict["param_flags"], lr=params_dict["LR"])
            print("Optimization mesh HERE")


    def opt_routine(self, save, save_path):

        # Run the optimization loop
        self.__opt_loop__()

        # Save parameters for over time optimization
        self.prev_body_pose = self.body_pose.detach().clone()

        self.prev_posed_3d_kps = []
        self.prev_posed_3d_kps.append(self.posed_3d_keypoints.detach().clone())

        if self.action_loader.labels_only:
            ## Calculate IoU and keypoints
            # Erase previous frames for evaluation
            self.euclid_three_d = {}
            self.IoUs = {}

            ## Keypoints error
            keypoints_3d_posed = self.posed_3d_keypoints.detach().clone().cpu().numpy()[0]
            # Get the 3d keypoints for that frame
            keypoints_3d_gt = self.action_loader.keypoints_3d[self.frame][self.cur_ind]
            euclid_over_keypoints = {}
            for keypoint_index in range(keypoints_3d_gt.shape[0]):
                euclid_dist = eval_utils_cropped.euclid_3d(keypoints_3d_posed[keypoint_index, :],
                                                           keypoints_3d_gt[keypoint_index, :])
                euclid_over_keypoints[keypoint_index] = euclid_dist
            self.euclid_three_d[self.frame] = euclid_over_keypoints

            ## IoU error
            # Iterate over the cameras
            ious_per_view = {}
            for render_index, cam_index in enumerate(self.cameras.keys()):
                cam_str = str(cam_index)
                # Get the gt mask
                iou = eval_utils_cropped.calc_IoU(self.gt_masks[render_index],
                                                  self.image_preds[render_index])
                ious_per_view[cam_index] = iou
            self.IoUs[self.frame] = ious_per_view
            self.state_statistics[self.stage] = {'kp_3d': self.euclid_three_d, 'IoUs': self.IoUs}

        if save:
            ## Save the parameters
            torch.save(self.model_parameters, save_path)

    def reduced_model_param_dict(self):
        """
        Reduce to parameters that were optimized only, so afterwards the saved parameters do not hold arbitrary values
        :return:
        """
        model_params_reduced = {}
        for key, flag in self.params_dict["param_flags"].items():
            if flag:
                model_params_reduced[key] = self.model_parameters[key]

        return model_params_reduced

    def opt_pose(self, alphas, steps=50, T=0, param_flags=[True, True, True, True, False, False], save=True,
                 lr=None):

        # Extend the optimization flags by the focal length estimation
        if self.single_view:
            assert param_flags.__len__() == 7
        #    param_flags = [True, True, True, True, False, False, True]

        # Change optimization state
        self.opt_state = 0
        self.update_opt_params(alphas=alphas, steps=steps, T=T, flags=param_flags, lr=lr)

        if os.path.exists(self.model_init_pose_save_path):
            if self.load_init_tensors:

                if self.verbose:
                    print("LOADING Pose INIT")
                self.load_parameters_from_file(self.model_init_pose_save_path)
            else:
                if self.verbose:
                    print("There is no Opt data found -- Running Opt Pose")
                self.opt_routine(save=save, save_path=self.model_init_pose_save_path)
        else:
            self.opt_routine(save=save, save_path=self.model_init_pose_save_path)

    def opt_bones(self, alphas, steps=140, T=20, param_flags=[True, True, True, True, True, False], save=True,
                  lr=None):

        # Extend the optimization flags by the focal length estimation
        if self.single_view:
            assert param_flags.__len__() == 7

        # Change optimization state
        self.opt_state = 1
        self.update_opt_params(alphas=alphas, steps=steps, T=T, flags=param_flags, lr=lr)

        if os.path.exists(self.model_init_bones_save_path):

            if self.load_init_tensors:
                print("LOADING Bones INIT")
                self.load_parameters_from_file(self.model_init_bones_save_path)
            else:
                print("There is no Opt data found -- Running Opt Bones")
                self.opt_routine(save=save, save_path=self.model_init_bones_save_path)
        else:
            self.opt_routine(save=save, save_path=self.model_init_bones_save_path)

    def opt_mesh(self, alphas, steps=140, T=20, param_flags=[False, False, True, False, True, True], save=True,
                 lr=None, large_shape_deform=False):

        # Extend the optimization flags by the focal length estimation
        if self.single_view:
            assert param_flags.__len__() == 7

        if large_shape_deform:
            self.large_shape_deform = large_shape_deform
            self.model_init_mesh_save_path = self.model_init_mesh_save_path[:-4] + f"_{large_shape_deform}.pth"

        # Change optimization state
        self.opt_state = 2
        self.update_opt_params(alphas=alphas, steps=steps, T=T, flags=param_flags, lr=lr)

        if save:
            self.save_mesh = True

        if os.path.exists(self.model_init_mesh_save_path):

            if self.load_init_tensors:
                print("LOADING MESH INIT")
                self.load_parameters_from_file(self.model_init_mesh_save_path)
            else:
                print("There is no Opt data found -- Running Opt Mesh")
                self.opt_routine(save=save, save_path=self.model_init_mesh_save_path)
        else:
            self.opt_routine(save=save, save_path=self.model_init_mesh_save_path)

        self.save_mesh = False

    def opt_time(self, alphas, steps=140, T=20, param_flags=[True, True, False, True, False, False], save=True,
                 lr=None):

        if self.single_view:
            raise ValueError("Over-time estimation, flags, smoothing is not yet implemented!")

        # Change optimization state
        self.opt_state = 3

        self.update_opt_params(alphas=alphas, steps=steps, T=T, flags=param_flags, lr=lr)

        if self.load_init_tensors:
            if os.path.exists(self.model_time_opt_save_path):
                print("LOADING TIME INIT")
                self.load_parameters_from_file(self.model_init_bones_save_path)
        else:
            print("There is no Opt data found -- Running Opt Time")
            ## Assign the current model parameters for frame in saving
            self.model_params_time[self.start_frame] = self.retrieve_model_params_time_copy()
            for frame in range(self.start_frame + 1, len(self.action_loader.masks)):
                # Change the frame
                self.frame = frame
                # Run the optimization loop
                self.__opt_loop__()
                self.prev_body_pose = self.body_pose.detach().clone()
                self.prev_posed_3d_kps.append(self.posed_3d_keypoints.detach().clone())
                self.model_params_time[frame] = self.retrieve_model_params_time_copy()
            if save:
                ## Save the parameters
                torch.save(self.model_params_time, self.model_time_opt_save_path)

    def load_individual_fits_parameters(self, path):
        """
        Only for preloading initalizations
        :param path:
        :return:
        """

        self.init_model_params()
        self.model_parameters = torch.load(path, map_location=self.device)
        self.assign_dict_to_model_parameters()
        self.__pose_mesh__()

    def load_tracking_params(self, nb_cameras):

        self.init_model_params()



        # Path to the over time fits
        self.model_parameters = torch.load(join(self.action_loader.action_path,
                                                f"action_params_{self.cur_ind}_{str(nb_cameras).zfill(2)}.pth"),
                                           map_location=self.device)

        self.assign_dict_to_model_parameters()


    def update_flags_only(self):
        # Update flags
        self.global_orientation.requires_grad = self.params_dict["param_flags"]["R"]
        self.global_t.requires_grad = self.params_dict["param_flags"]["t"]
        self.global_scale.requires_grad = self.params_dict["param_flags"]["s"]
        self.body_pose.requires_grad = self.params_dict["param_flags"]["body_pose"]
        self.bone_length.requires_grad = self.params_dict["param_flags"]["bone_lengths"]
        self.vertex_offsets.requires_grad = self.params_dict["param_flags"]["vertex_offsets"]
        self.vertex_colors.requires_grad = self.params_dict["param_flags"]["rgb"]

        if self.single_view:
            self.focal_lengths.requires_grad = self.params_dict["param_flags"]["focal"]

    def assign_model_paramaters_to_dict(self):

        self.model_parameters["R"] = self.global_orientation
        self.model_parameters["t"] = self.global_t
        self.model_parameters["s"] = self.global_scale
        self.model_parameters["body_pose"] = self.body_pose
        self.model_parameters["bone_lengths"] = self.bone_length
        self.model_parameters["vertex_offsets"] = self.vertex_offsets
        self.model_parameters["rgb"] = self.vertex_colors

        if self.single_view:
            self.model_parameters["focal"] = self.focal_lengths

    def assign_dict_to_model_parameters(self):


        self.global_orientation= self.model_parameters["R"]
        self.global_t = self.model_parameters["t"]
        self.global_scale = self.model_parameters["s"]
        self.body_pose = self.model_parameters["body_pose"]
        self.bone_length = self.model_parameters["bone_lengths"]
        self.vertex_offsets = self.model_parameters["vertex_offsets"]


        #rgb_values =

        ## Make sure these values are within 0, 255 and as uin8 --> use a sigmoid function
        self.vertex_colors = self.model_parameters["rgb"]#torch.clip(rgb_values, min=0, max=255)#.to(torch.uint8)

        if self.single_view:
            self.focal_lengths = self.model_parameters["focal"]



    def update_learnable_flags_optimizer(self):


        self.model_parameters = {}

        self.assign_model_paramaters_to_dict()
        self.update_flags_only()

        body_opt_params = [param for param_key, param in self.model_parameters.items()
                           if self.params_dict["param_flags"][param_key]]


        body_opt_params = []

        for param_key, param in self.model_parameters.items():
            if self.params_dict["param_flags"][param_key]:
                body_opt_params.append({'params': param,
                                        'lr': self.params_dict["param_lr"][param_key]})

        self.body_optimizer = torch.optim.Adam(body_opt_params, lr=self.lr, betas=(0.9, 0.999))

        if self.params_dict["Scheduler"]:
        ## Create here a new LR scheduler, new limit needs to below, 0.5% better than last
            #

            #self.scheduler = ReduceLROnPlateau(self.body_optimizer, factor=0.5, patience=15, mode='min', threshold=0.0005)
            if self.step_scheduler:
                # Similiar function as StepLR, but if wanted the milestones can be defined better
                #self.scheduler = MultiStepLR(self.body_optimizer, milestones=[100, 1400], gamma=0.25) #the previous one

                smooth_scheduler=True

                if not smooth_scheduler:
                    self.scheduler = MultiStepLR(self.body_optimizer, milestones=[300, 1400], gamma=0.25)
                else:
                    self.scheduler = MultiStepLR(self.body_optimizer, milestones=[300, 330], gamma=0.25) # this makes it more smooth
            else:
                self.scheduler = ReduceLROnPlateau(self.body_optimizer, factor=0.5, patience=15, mode='min',
                                               threshold=0.005) # not used in either vertex or time opt


        if self.params_dict["WeightedSil"]:
            self.weighted_sil_loss = True
        else:
            self.weighted_sil_loss = False


    def non_view_losses(self, loss_dict):

        scale_loss = scaling_loss(self.global_scale) * self.alphas[
            "Scale"]  # silhouette_loss(image_pred, gt_mask * 0.0039, self.device)#

        if self.single_view:
            scale_jt_loss = torch.sum(
                scaling_loss(self.bone_length, min_scale=0.5, max_scale=2.5) ** 2)

            if self.scale_hands_feet:
                ## Penalize hands and feet that are too large
                scale_jt_loss += scaling_loss(self.bone_length[0, -1], min_scale=0.2, max_scale=1) ** 2

        else:
            # Sum all the individual elements
            bones_lim = 0.85
            # 0.75 was okey, but still some shift as in drinking h1
            scale_jt_loss = torch.sum(
                scaling_loss(self.bone_length, min_scale=bones_lim, max_scale=1 / bones_lim) ** 2)

        scale_jt_loss *= self.alphas["JT_scale"]

        ## Vertex offset smooth loss
        if self.alphas["Vertex_smooth"] == 0:
            v_smooth_loss = torch.tensor([0], device=self.device)
        else:
            v_smooth_loss = losses.vertex_smooth_loss(self.vertex_weights, self.mesh_model)




        ## Laplace loss
        if self.alphas["Laplacian"] == 0:
            laplace_loss = torch.tensor([0], device=self.device)
        else:
            # laplace_loss = self.alphas["Laplacian"] * compute_laplacian(self.rest_pose_shifted_V.squeeze(),
            #                                                       self.mesh_model)

            laplace_loss = self.alphas["Laplacian"] * mesh_laplacian_smoothing(self.posed_mesh)

            # print(f"laplace_loss: {laplace_loss}")

        # print(f"Edge loss: {compute_edge_loss(self.rest_pose_shifted_V.squeeze(), self.mesh_model)}")


        if self.alphas["Prev_3dkp"] == 0:
            prev_body_3d_kp_loss = torch.tensor([0], device=self.device)
        else:
            # Apply a smooth loss over the skeletons 3d keypoints
            prev_body_3d_kp_loss = losses.sm_time_loss(self.posed_3d_keypoints, mean=False)

        #print(f"Prev keypoints loss: {prev_body_3d_kp_loss}")

        if self.alphas["T"] == 0:
            time_loss = torch.tensor([0], device=self.device)
        else:
            reduce_for_end_effectors = True
            if reduce_for_end_effectors:

                # Get selected elements
                selected = self.body_pose[:, :, self.end_effector_indices]
                # Get complement (elements NOT in indices)
                mask = torch.ones(self.body_pose.size(2), dtype=bool, device=self.device)
                mask[self.end_effector_indices] = False
                complement = self.body_pose[:, :, mask]

                hand_loss = self.selection_scalar * losses.compute_angular_velocity(selected)
                other_time_losses = (losses.compute_angular_velocity(complement) + losses.sm_time_loss(self.global_t) +
                                     losses.compute_angular_velocity(self.global_orientation))

                #small_align_loss = losses.sm_time_loss(selected) + losses.sm_time_loss(complement)

                #print(f"Hand Loss, Other Losses {hand_loss}, {other_time_losses}")
                time_loss = (hand_loss+ other_time_losses ) # the smoothing does not work for the rest of the body if there is a different global orientation
            else:
                # Get the smoothness for all the pose parameters, including R and t
                time_loss = (losses.sm_time_loss(self.body_pose) + losses.sm_time_loss(self.global_t) +
                             losses.sm_time_loss(self.global_orientation))


        time_loss *= self.alphas["T"]

        loss_dict["Scale"] = scale_loss
        #loss_dict["Cage"] = cage_pos_loss
        loss_dict["JT_scale"] = scale_jt_loss

        #loss_dict["Prev_b_p"] = prev_body_pose_loss
        loss_dict["Prev_3dkp"] = prev_body_3d_kp_loss
        #loss_dict["Kin"] = kinematic_loss
        #loss_dict["Angle_consist"] = angle_consistency_loss
        #loss_dict["Max_rot_loss"] = max_rot_loss
        loss_dict["Vertex_smooth"] = v_smooth_loss
        #loss_dict["Edge"] = edge_loss
        loss_dict["Laplacian"] = laplace_loss
        #loss_dict["Vertex_offset_norm"] = v_norm_loss
        #loss_dict["Normal_ref_loss"] = v_normals_ref_loss
        #loss_dict["Hull"] = hull_loss
        loss_dict["T"] = time_loss

        return loss_dict


    def __cam_loop__(self):

        self.loss = torch.tensor([0], device=self.device, dtype=torch.float)
        sil_loss_over_cams = torch.tensor([0], device=self.device, dtype=torch.float)

        self.image_preds = []
        self.gt_masks = []
        cps = []
        cgs = []



        if self.action_loader.same_size:

            batched_cameras = self.batched_cameras[self.flattened_idx]
            gt_masks = self.gt_masks_list[self.flattened_idx]

            # Just take the last silhouette renderer for now
            sil_renderer = list(self.silhouette_renderer.values())[0]#[frame_cam_key]

            meshes = []
            for frame_index, frame in enumerate(self.bundled_frames):
                meshes.append(self.posed_mesh[frame_index].extend(self.NUM_CAMS))

            meshes = pytorch3d.structures.join_meshes_as_batch(meshes)

            if self.alphas["Sil"] != 0:
                image_preds = sil_renderer(meshes, cameras=batched_cameras, lights=self.lights)

                # Generate the losses as well
                sil_losses = torch.mean((image_preds[:, :, :, 3] - gt_masks)**2, dim=[1, 2])

            if self.alphas["RGB"] != 0:
                disp_renderer = list(self.display_renderer.values())[0]#self.display_renderer[cam_key]
                rgb_preds = disp_renderer(meshes, cameras=batched_cameras)


        # The batching doesn't really save much, because it increases time only once for the for loop pass

        loss_dict = {}
        loss_dict = self.non_view_losses(loss_dict)

        # Frame index and frame of a list of frames to iterate
        # The association to the entire action for body pose is given by action_frame_index
        for frame_index, frame in enumerate(self.bundled_frames):

            # Set the current for frame
            self.frame = frame

            #loss_dict = {}
            cam_loss_dict = {}
            loss_dict_for_show = {}

            ### Initialize loss dict
            for alpha_key in self.alphas.keys():
                # loss_dict[alpha_key] = []
                loss_dict_for_show[alpha_key] = []

                if alpha_key in ["Keyp", "Sil", "RGB", "IoU", "rgb_lim"]:
                    cam_loss_dict[alpha_key] = []


            #camera_keys_to_retain = self.cameras_to_retain[frame_index]

            if self.action_loader.labels_only:
                # There is no frame specific camera selection process
                camera_keys_to_retain = self.cameras_to_retain
            else:
                camera_keys_to_retain = self.cameras_to_retain[frame_index]



            # Retrieve the posed 3d keypoints of the frame batch
            kps_per_frame = self.posed_3d_keypoints[frame_index]

            if self.action_loader.same_size:
                cam_sel = list(range(frame_index * self.NUM_CAMS, frame_index * self.NUM_CAMS + self.NUM_CAMS, 1))
                batched_keypoints2d = batched_cameras[cam_sel].transform_points_screen(kps_per_frame)
                # 1/5 th of the time is attributed to transforming points to screen

            # Per pose losses,
            joint_angle_loss = self.alphas["Jt_angle"] * losses.joint_angle_loss(
                self.body_pose[self.action_frame_index[frame_index], :, :], self.device,
                angle_weighting=self.ANGLE_WEIGHTING)

            # Test to have instead just the joint limitations as loss
            # Squared losses per element and the summed across all dims, using relu limits
            joint_angle_loss = self.alphas["Jt_angle"] * torch.sum( scaling_loss(scale=self.body_pose[self.action_frame_index[frame_index], :, :],
                                                                      min_scale = self.joints_lower_bound,
                                                                      max_scale = self.joints_upper_bound) ** 2 )


            # Add to the jt angle loss
            # if self.single_view and self.alphas["Mahalanobis"] != 0:
            #     # Create the variables on the fly again, otherwise can't backwardpass multiple times.
            #     # Calculate the distance to poses from the pose prior
            #     mahalanobis_loss = losses.mahalanobis_dist(self.body_pose[frame, 0, :],
            #                                                torch.tensor(self.pose_prior_mean, device=self.device),
            #                                                torch.tensor(self.pose_prior_covI, device=self.device))
            #
            # else:
            #     mahalanobis_loss = torch.Tensor([0]).to(self.device)

            loss_dict["Jt_angle"] = joint_angle_loss
            #loss_dict["Mahalanobis"] = mahalanobis_loss

            for cam_key_index, cam_key in enumerate(camera_keys_to_retain):


                cam_iter_index = frame_index * self.NUM_CAMS + cam_key_index
                frame_cam_key = f"{frame}_{cam_key}"

                cam_str = str(cam_key)
                cam = self.cameras[frame_cam_key]

                # Render the silhouette or retrieve it from a batch
                sil_renderer = self.silhouette_renderer[frame_cam_key]

                if self.alphas["Sil"] != 0:
                    if self.action_loader.same_size:
                        image_pred = image_preds[cam_iter_index, :, :, :][None]
                    else:
                        image_pred = sil_renderer(self.posed_mesh[frame_index], cameras=cam, lights=self.lights)

                # Render the rgb image or retrieve it from a batch

                disp_renderer = self.display_renderer[frame_cam_key]

                if self.alphas["RGB"] != 0:
                    if self.action_loader.same_size:
                        rgb_pred = rgb_preds[cam_iter_index, :, :, :][None]
                    else:
                        rgb_pred = disp_renderer(self.posed_mesh[frame_index], cameras=cam, lights=self.lights)

                ## Keypoint loss
                if self.action_loader.same_size:
                    # Project keypoints into camera view
                    camera_keypoints2d = batched_keypoints2d[cam_key_index, :, :]
                else:
                    camera_keypoints2d = cam.transform_points_screen(kps_per_frame)

                camera_keypoints2d *= self.reduction_sampling_image_size

                # Get the ground-truth values
                camera_keypoints2d_gt = self.action_loader.keypoints_2d[frame][cam_str]
                gt_mask = self.action_loader.masks[frame][cam_str][self.cur_ind]

                if self.alphas["RGB"] != 0:
                    rgb_gt = self.action_loader.rgbs[frame][cam_str][self.cur_ind]

                ## Debugging of camera difference between points and mask
                # camera_keypoints3d_imag = cam.transform_points_screen(self.posed_mesh.verts_list()[0][None, :, :])
                # camera_keypoints3d_imag = camera_keypoints3d_imag[0, :, :2].cpu().detach().numpy() * self.reduction_sampling_image_size
                # segm_mask_pred = image_pred.cpu().detach().numpy()[0, :, :, 3]
                # plt.imshow(segm_mask_pred)
                # plt.scatter(x=camera_keypoints3d_imag[:, 0], y=camera_keypoints3d_imag[:, 1], s=10, c='k')
                # plt.show()

                #print("Before keypoints: ", time.time() - start_time)

                #keypoint_loss = torch.Tensor([0]).to(self.device)

                if self.single_view:

                    if self.action_loader.dataset == "OMC":

                        cam_2d_gt = camera_keypoints2d_gt[self.cur_ind][remap_list_omc, :]

                        keypoint_loss = keypoint_loss_2D(camera_keypoints2d, cam_2d_gt, self.device,
                                                         keypoint_weights=KEYPOINT_WEIGHTING_OMC)
                    elif self.action_loader.dataset == "OAP":

                        indices_to_take = [i for i in range(15)]
                        camera_keypoints2d = camera_keypoints2d[:, indices_to_take, :]

                        cam_2d_gt = camera_keypoints2d_gt[self.cur_ind][remap_list_oap, :]

                        keypoint_loss = keypoint_loss_2D(camera_keypoints2d, cam_2d_gt, self.device,
                                                         keypoint_weights=KEYPOINT_WEIGHTING_OAP)

                else:

                    if self.OMS:
                        indices_to_take = [2, 3, 4, 6,
                                           7, 9, 10, 11,
                                           12, 13, 14, 15]
                        camera_keypoints2d = camera_keypoints2d[:, indices_to_take, :]
                        indices_to_take_gt = [0, 2, 3, 4,
                                              5, 6, 7, 8,
                                              9, 10, 11, 12]
                        camera_keypoints2d_gt = camera_keypoints2d_gt[indices_to_take_gt, :]

                        keypoint_loss = keypoint_loss_2D(camera_keypoints2d, camera_keypoints2d_gt, self.device,
                                                         keypoint_weights=KEYPOINT_WEIGHTING_OMS)
                    else:

                        # BIGMAC case

                        # Filter the keypoints that are really within image regions

                        keypoints_2d_gt_unfil = camera_keypoints2d_gt[self.cur_ind]

                        #H, W = gt_mask.shape[:2]

                        valid_mask = losses.keypoints_in_image(keypoints_2d_gt_unfil, gt_mask.shape)

                        # Do this for the rest as well
                        #valid_mask = np.ones(16).astype(bool)
                        # valid_mask[:5] = False

                        keypoints_2d_gt_fil = keypoints_2d_gt_unfil[valid_mask]

                        camera_keypoints2d_fil = camera_keypoints2d[valid_mask]
                        keypoint_weights = self.KEYPOINT_WEIGHTING[valid_mask]
                        keyp_det_confs = (self.action_loader.keypoints_conf[frame][cam_str][self.cur_ind])[valid_mask]

                        if self.use_kp_conf:
                            # Also give the detection confidences
                            keypoint_loss = keypoint_loss_2D(camera_keypoints2d_fil, keypoints_2d_gt_fil,
                                                             self.device,
                                                             keypoint_weights=keypoint_weights,
                                                             det_confidences=keyp_det_confs)
                        else:

                            keypoint_loss = keypoint_loss_2D(camera_keypoints2d_fil, keypoints_2d_gt_fil,
                                                             self.device,
                                                             keypoint_weights=keypoint_weights)

                        if torch.isnan(keypoint_loss):
                            keypoint_loss = torch.zeros_like(keypoint_loss)

                #image_pred = torch.zeros(size=(1, gt_mask.shape[0], gt_mask.shape[1], 4), device=self.device)
                #rgb_pred = torch.zeros(size=(1, gt_mask.shape[0], gt_mask.shape[1], 4), device=self.device)

                #print(f"Keypoint_loss: {keypoint_loss}, Threshold {self.keypoint_thresh}")

                #print("After keypoints: ", time.time() - start_time)
                #if self.epoch < 50:
                if keypoint_loss > self.keypoint_thresh:  # Per camera, so it could still be that sil loss is plotted
                    #sil_loss *= 0
                    #intersection_loss *= 0
                    #rgb_loss *= 0

                    rgb_loss = torch.tensor([0], device=self.device)
                    sil_loss = torch.tensor([0], device=self.device)

                    image_pred = torch.zeros(size=(1, gt_mask.shape[0], gt_mask.shape[1], 4), device=self.device)
                    rgb_pred = torch.zeros(size=(1, gt_mask.shape[0], gt_mask.shape[1], 4), device=self.device)

                else:

                    if self.alphas["Sil"] != 0:

                        if self.weighted_sil_loss:
                            #segm_mesh = self.mesh_model.get_segmented_posed_mesh(self.posed_mesh)
                            #image_pred = sil_renderer(segm_mesh, cameras=cam, lights=self.lights)
                            #image_pred = sil_renderer(self.posed_mesh, cameras=cam, lights=self.lights)
                            sil_loss, image_pred = silhouette_segm_loss(image_pred, gt_mask, self.device)
                            sil_loss *= self.alphas["Sil"]
                        else:

                            if self.action_loader.same_size:
                                sil_loss = sil_losses[cam_iter_index]
                                #sil_loss, image_pred = silhouette_loss(image_pred[0, :, :, 3], gt_mask, self.device)
                            else:
                                sil_loss, image_pred = silhouette_loss(image_pred[0, :, :, 3], gt_mask, self.device)
                            #sil_loss = torch.Tensor([0]).to(self.device)
                            sil_loss *= self.alphas["Sil"]
                    else:
                        sil_loss = torch.tensor([0], device=self.device)



                    # ------------ RGB
                    if self.alphas["RGB"] == 0:
                        rgb_loss = torch.tensor([0], device=self.device)
                    else:
                        # L2 loss, on masked image
                        rgb_loss = self.alphas["RGB"] * losses.rgb_loss(rgb_pred, rgb_gt, gt_mask, self.device)


                keypoint_loss *= self.alphas["Keyp"]

                #print(f"{self.frame}, {cam_str}: {keypoint_loss}, {sil_loss}, {joint_angle_loss}, {time_loss}")

                cam_loss_dict["Keyp"].append(keypoint_loss)
                cam_loss_dict["Sil"].append(sil_loss)
                cam_loss_dict["RGB"].append(rgb_loss)

                if self.verbose:

                    # Plot the joints of the entire skeleton for visualization and debugging
                    if self.plot_skeleton:
                        camera_keypoints2d = cam.transform_points_screen(self.joints_coordinates_posed)
                        camera_keypoints2d *= self.reduction_sampling_image_size

                    if camera_keypoints2d.dim() == 2:
                        camera_keypoints2d = camera_keypoints2d[None]


                    cps.append(camera_keypoints2d[0, :, :2].cpu().detach().numpy())
                    # cgs.append(self.action_loader.keypoints_2d[self.frame][cam_str])
                    cgs.append(camera_keypoints2d_gt[self.cur_ind].cpu().detach().numpy())

                    ## Debugging of labels in image
                    # camera_keypoints2d_gt = camera_keypoints2d_gt[remap_list_omc, :]
                    # camera_keypoints3d_imag = camera_keypoints3d_imag[0, :, :2].cpu().detach().numpy() * self.reduction_sampling_image_size
                    # segm_mask_pred = image_pred.cpu().detach().numpy()[0, :, :, 3]
                    # plt.imshow(gt_mask)
                    # plt.scatter(x=camera_keypoints2d_gt[:, 0], y=camera_keypoints2d_gt[:, 1], s=10, c='k')
                    # plt.show()

                    #self.image_preds.append(image_pred.cpu().detach().numpy()[0, :, :, 3])

                    # Just to show the silhouette
                    if len(image_pred.size()) > 2:
                        image_pred = image_pred[0, :, :, 3]
                    self.image_preds.append(image_pred.cpu().detach().numpy())
                    self.gt_masks.append(gt_mask.cpu().detach().numpy())


            # For a single frame aggregate the camera losses.

            ### Re-write iter loss dict and also cam dict
            non_view_loss = 0
            for loss_key, loss in loss_dict.items():
                non_view_loss += loss
                loss_dict_for_show[loss_key].append(loss.item())

            for alpha_key in cam_loss_dict.keys():
                for camera_loss_index, cam_spec_loss in enumerate(cam_loss_dict[alpha_key]):

                    if self.use_bbox_conf:

                        cam_for_loss = camera_keys_to_retain[camera_loss_index]
                        bbox_conf = self.action_loader.probs_ratings[frame][cam_for_loss][self.cur_ind]
                        sil_loss_over_cams += (1 / camera_keys_to_retain.__len__()) * cam_spec_loss * bbox_conf
                    else:
                        sil_loss_over_cams += (1 / camera_keys_to_retain.__len__()) * cam_spec_loss

                    # For the monitor
                    loss_dict_for_show[alpha_key].append(cam_spec_loss.item())


            ## Assign loss for backprop
            self.loss += sil_loss_over_cams + non_view_loss




            if self.verbose:

                # Update the loss monitor,
                self.loss_monitor.update_losses(self.alphas, loss_dict_for_show)

                self.loss_monitor.update_lr(self.body_optimizer.param_groups)

                if self.frame % self.mod_iter_show == 0:
                    # Plot predictions and masks

                    self.vis.plot_mask_and_pred(self.image_preds, self.gt_masks, cps, cgs,
                                                self.loss_monitor.silhouette_loss.__len__() - 1,
                                                self.frame,
                                                self.iter)
                    # Plot the mesh in 3D space
                    if self.plot_3d:
                        self.cams_bigmac.show(lim=3, cam_scale=1 / 10, dim_mm=False,
                                              object_points=self.posed_verts.cpu().detach().numpy()[0])



    def __pose_mesh__(self):

        ## Save the mesh if we want to
        #self.save_mesh_template = None
        if self.save_mesh and self.iter == (self.steps + self.iters_total - 1):
            self.save_mesh_template = join(self.mesh_path, f"{str(self.frame).zfill(4)}_{self.opt_state}_{self.alphas['Sil']}"
                                                           f"_{self.alphas['Vertex_smooth']}_{self.alphas['Max_rot_loss']}_{self.alphas['Edge']}"
                                                           f"_{self.alphas['Laplacian']}_{self.alphas['IoU']}.obj")

        if self.sparse:
            # Vertex sparsity
            if self.vertex_normals_flag:
                vertex_weights = torch.matmul(self.vertex_sparse, self.vertex_offsets)
            else:
                vertex_weights = torch.cat(
                    (torch.unsqueeze(torch.matmul(self.vertex_sparse, self.vertex_offsets[:, 0]), dim=1),
                     torch.unsqueeze(torch.matmul(torch.abs(self.vertex_sparse), self.vertex_offsets[:, 1]), dim=1),
                     torch.unsqueeze(torch.matmul(torch.abs(self.vertex_sparse), self.vertex_offsets[:, 2]), dim=1)),
                    dim=1)

            bone_lengths_LBS = torch.ones(self.mesh_model.get_nb_bones(), device=self.device)
            if not self.scale_hands_feet:
                # Bone length sparsity
                bone_lengths_LBS[torch.where(self.bone_length_sparse == 1)] = self.bone_length
            else:
                # print("All bone lengths", self.bone_length)
                bone_lengths_LBS[torch.where(self.bone_length_sparse == 1)] = self.bone_length[0, :-1]
                # Unify the bone lengths of hands and feet
                # Take the last entry of bone length


                # Scales all the bones in hands and feet by single scalar...
                # This must be accompanied by mesh deformations, otherwise it looks slenderish

                # print(self.bone_length[0, -1])
                # Scales the hand and feet only by a single scalar
                bone_lengths_LBS[torch.where(self.bone_length_sparse_hf == 1)] = self.bone_length[0, -1]

            # Collapse the tail if necessary
            bone_lengths_LBS = bone_lengths_LBS * self.bone_length_tail_vec

            bone_lengths_LBS = bone_lengths_LBS[None]

            body_pose_LBS = self.body_pose[self.action_frame_index, :, :]

        else:
            vertex_weights = self.vertex_offsets
            bone_lengths_LBS = self.bone_length

            # Leave the body pose for now, because after rotation it worked
            body_pose_LBS = self.body_pose[self.action_frame_index, :, :]

        self.bone_lengths_LBS = bone_lengths_LBS
        self.body_pose_LBS = body_pose_LBS
        self.vertex_weights = vertex_weights
        self.global_scale_LBS = self.global_scale


        #self.mesh_model.vert_colors = self.vertex_colors

        if self.render_default_col:
            self.mesh_model.vert_colors = self.mesh_model.default_vertex_colors
        else:
            ## If not using sigmoid, then values larger than the RGB range come that might skew the gradient...
            self.mesh_model.vert_colors = 255 * torch.sigmoid(self.vertex_colors)



        ## Pose the mesh
        self.posed_verts, self.posed_mesh, self.rest_pose_shifted_V, self.joints_coordinates_posed = (
            self.mesh_model(body_pose=self.body_pose_LBS,
                            bone_length=self.bone_lengths_LBS,
                            global_orientation=self.global_orientation[self.action_frame_index, :, :],
                            global_t=self.global_t[self.action_frame_index, :, :],
                            vertex_normal_weights=self.vertex_weights
                            , scale=self.global_scale_LBS,
                            return_mesh=True,
                            save_mesh_template=self.save_mesh_template,
                            vertex_normals=self.vertex_normals_flag,
                            body_pose_sparse_P=self.body_pose_sparse, sparsify_roll=self.body_pose_sparse_P_roll,
                            sparsify_roll_index=self.restricted_index_in_to_opt))

        ## Extract the joint positions
        if self.infer_from_rig:
            self.posed_3d_keypoints = return_labeled_rig_joints(self.joints_coordinates_posed, self.mesh_model,
                                                                self.HIERARCHY_MAPPING)
        else:
            # Calculate the specific keypoints of the model in 3D
            self.posed_3d_keypoints = calculate_keypoints3D_from_posed_mesh(self.posed_verts, self.device)

    def check_gradients(self):

        ret = True
        for model_param_idx, model_param in enumerate(self.model_parameters.values()):
            if model_param.grad is not None:
                if torch.all(torch.isnan(model_param.grad)):
                    print(f"Gradient computations are None for {model_param_idx}th param")
                    return False
                    #raise ValueError(f"Gradient computations are None for {model_param_idx}th param")
        return ret
    def __opt_loop__(self):

        # Update the gradient flags
        self.update_learnable_flags_optimizer()

        for iter in range(self.iters_total, self.steps + self.iters_total, 1):

            self.iter = iter

            self.__pose_mesh__()

            self.__cam_loop__()

            if self.save_iters_as_plots:
                self.display_mesh_on_images()

            self.body_optimizer.zero_grad()
            self.loss.backward()
            self.check_gradients()
            # print(self.global_scale.grad)
            # print(self.model_parameters[2].grad)
            # Throw an error if there is a nan in gradient for global scale

            # Check if None if not None check that it is not nan!!!!
            #self.print_cuda_memory()
            self.body_optimizer.step()

            ### Adjust the lr scheduler
            # self.scheduler.step(self.loss.item())


            # If the learning rate is too small break the loop
            if self.body_optimizer.param_groups[0]["lr"] < 0.003:
                break

        # Update total iterations elapsed
        self.iters_total += self.steps

        ### Create a color map
        if self.opt_state == 4:
            self.render_colormaps()

        ### Save Renderings
        if self.save_renderings and self.action_loader.action_path:
            self.display_mesh_on_images()

        if self.single_view:
            ### Assign again all variables to the model_parameters
            self.model_parameters = [self.global_orientation, self.global_t, self.global_scale, self.body_pose,
                                     self.bone_length, self.vertex_offsets, self.focal_lengths]
        else:
            ### Assign again all variables to the model_parameters
            self.model_parameters = [self.global_orientation, self.global_t, self.global_scale, self.body_pose,
                                     self.bone_length, self.vertex_offsets]

    def render_colormaps(self):
        """
        Following: https://github.com/facebookresearch/pytorch3d/issues/126
        :return:
        """

        vertex_colors_saved = {}

        # Load the images
        self.load_rgb_labeled_images()

        # Iterate the cameras
        for cam_key, camera in self.cameras.items():
            raster_camera = self.colormap_rasterizer(camera)

            fragments = raster_camera(self.posed_mesh)

            meshes = self.posed_mesh

            # pix_to_face is of shape (N, H, W, 1)
            pix_to_face = fragments.pix_to_face

            # plt.figure()
            # plt.imshow(pix_to_face[0, :, :, 0].clone().detach().cpu().numpy())
            # plt.show()

            # (F, 3) where F is the total number of faces across all the meshes in the batch
            # 3 vertices are per face then, row indicates the face_idx, 3 are then the vertices
            packed_faces = meshes.faces_packed()

            pix_to_face_image = pix_to_face[0, :, :, 0]
            unique_faces = pix_to_face_image.unique()

            # Iterate over all faces that are in the rendering
            for face_id in tqdm(unique_faces):  # 20948

                spots = torch.where(pix_to_face_image == face_id)

                for occurence_idx in range(spots[0].shape[0]):
                    x = spots[0][occurence_idx]
                    y = spots[1][occurence_idx]

                    # Retrieve the color of the image at that location
                    color = self.rgbs[cam_key][x, y, :]

                    # Iter associateed vertices for that face, and save the color
                    for vert_id in packed_faces[face_id]:

                        if vert_id in vertex_colors_saved:
                            # Key exists, append the item to the existing list
                            vertex_colors_saved[vert_id].append(color)
                        else:
                            # Key does not exist, create a new list with the item
                            vertex_colors_saved[vert_id] = [color]

        num_vertices = self.mesh_model.V.shape[1]

        # Initialize RGB array
        rgb_array = np.zeros((num_vertices, 3), dtype=np.float64)

        # Iterate over the dictionary to compute the average colors
        for vertex_index, color_list in vertex_colors_saved.items():
            # Compute the average for each color channel
            if color_list:  # Check if there are colors to avoid division by zero
                averaged_color = np.mean(color_list, axis=0)  # .astype(int)
                rgb_array[vertex_index] = averaged_color

        # Put the new texture on the mesh

        # vert rgb as (1, V, 3)
        self.mesh_model.vert_colors = torch.from_numpy(rgb_array).to(torch.float32)[None].to(
            self.device)  # torch.from_numpy(rgb_array).to(self.device)[None]

        # maybe just generate a uv map with maya? how does this relate to obJ?
        """

        textures = Textures(verts_rgb=self.mesh_model.vert_colors)

        posed_mesh = Meshes([self.mesh_model.V.squeeze()],
                            faces=[torch.as_tensor(self.mesh_model.dd["F"], device=self.device)],
                            textures=textures)

        texture_img = posed_mesh.textures.maps_padded()
        plt.figure()
        plt.imshow(texture_img.squeeze().cpu().numpy())
        plt.show()
        print("Hi")
        """

    def colormap_rasterizer(self, camera):
        raster_settings = RasterizationSettings(
            image_size=(int(2056 * self.reduction_sampling_image_size),
                        int(2464 * self.reduction_sampling_image_size)),
            blur_radius=0.0,
            faces_per_pixel=1,  # K closest faces
        )
        rasterizer = MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        )

        return rasterizer

    def copy_gt_single_view(self, image_orig, species_path):

        image_orig = cv2.cvtColor(image_orig, cv2.COLOR_RGB2BGR)
        orig_image_path = join(species_path, f"{str(self.frame).zfill(7)}_i.png")

        if not os.path.exists(orig_image_path):
            # Save the original
            cv2.imwrite(orig_image_path, image_orig)

            # Make a mask of the image that is actually the silhouette
            bm = self.action_loader.masks[self.frame]["1"]
            bm = np.stack([bm] * 3, axis=-1)

            image_orig = image_orig * bm

            # Iterate the keypoints and make some dots
            for kp_index in self.remap_list:

                # Keypoint location
                (x, y) = self.action_loader.keypoints_2d[self.frame]["1"][kp_index, :]
                kp_name = self.kps_names[kp_index]
                cv2.circle(image_orig, (x, y), radius=2, color=(0, 255, 0), thickness=2)

                try:
                    cv2.putText(image_orig, f"{kp_name}",
                                (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                (142, 10, 234), 1)
                except:
                    print("The image boundaries was hit?")

            cv2.imwrite(join(species_path, f"{str(self.frame).zfill(7)}_labels.png"), image_orig)

    def save_mesh_on_images_single_view(self, image_combined, image_orig, mesh_image, copy_gt=True, save_mesh=True):

        species_path = join(self.action_loader.action_path, self.species)

        if not os.path.exists(species_path):
            os.makedirs(species_path)

        img_path = join(species_path, f"{str(self.frame).zfill(7)}_o_{self.opt_state}_"
                                      f"{self.opt_state_titles[self.opt_state]}.png")
        # Save the image
        cv2.imwrite(img_path, cv2.cvtColor(image_combined, cv2.COLOR_RGB2BGR))

        if save_mesh:
            if self.opt_state == 2:
                img_path = join(species_path, f"{str(self.frame).zfill(7)}_o_{self.opt_state}_"
                                              f"{self.opt_state_titles[self.opt_state]}_only.png")
                # Save the image
                cv2.imwrite(img_path, cv2.cvtColor(mesh_image, cv2.COLOR_RGB2BGR))

        if copy_gt:
            self.copy_gt_single_view(image_orig, species_path)


    def create_render_path(self):

        ### Creating folders for saving
        render_path = join(self.action_loader.action_path, "rendered_images_cropped_new")
        if not os.path.exists(render_path):
            os.makedirs(render_path)

        self.render_path = render_path
        self.render_path_base = self.render_path


    def save_mesh_on_image(self, cam_name, image_combined, mesh_alpha, background=None, mesh_rgb=None):


        self.create_render_path()

        #print(self.render_path)

        cam_path = join(self.render_path, f"{cam_name.zfill(2)}")
        if not os.path.exists(cam_path):
            os.makedirs(cam_path)

        if self.save_iters_as_plots:

            iters_folder = join(cam_path, "as_iters")
            if not os.path.exists(iters_folder):
                os.makedirs(iters_folder)

            img_path = join(iters_folder, f"{self.action_frame_index_single}_{self.cur_ind}_{str(self.frame).zfill(4)}_"
                                          f"{self.stage}_{self.epoch}_{len(self.cameras)}_"
                                          f"{str(self.iter).zfill(5)}.png")
            # Save the image
            cv2.imwrite(img_path, cv2.cvtColor(image_combined, cv2.COLOR_RGB2BGR))

        else:

            # if self.stage == "Time":
            #     time_path = join(cam_path, f"time_{self.smooth}")
            #     if not os.path.exists(time_path):
            #         os.makedirs(time_path)
            #
            #     img_path = join(time_path, f"{self.cur_ind}_{str(self.frame).zfill(4)}_{self.stage}_"
            #                                f"{self.stage}_{self.epoch}_{len(self.cameras)}.png")
            #     # Save the image
            #     cv2.imwrite(img_path, cv2.cvtColor(image_combined, cv2.COLOR_RGB2BGR))
            # else:

            if self.render_mesh_wo_bg:
                img_path = join(cam_path,
                                f"{self.action_frame_index_single}_{self.cur_ind}_{str(self.frame).zfill(4)}_"
                                f"{self.stage}_{self.epoch}_{len(self.cameras)}_mesh.png")
                cv2.imwrite(img_path, cv2.cvtColor(mesh_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))

            else:
                img_path = join(cam_path, f"{self.action_frame_index_single}_{self.cur_ind}_{str(self.frame).zfill(4)}_"
                                          f"{self.stage}_{self.epoch}_{len(self.cameras)}.png")
                # Save the image
                cv2.imwrite(img_path, cv2.cvtColor(image_combined, cv2.COLOR_RGB2BGR))

            if self.render_sil:
                img_path = join(cam_path, f"{self.action_frame_index_single}_{self.cur_ind}_{str(self.frame).zfill(4)}_"
                                          f"{self.stage}_{self.epoch}_{len(self.cameras)}_sil.png")
                cv2.imwrite(img_path, mesh_alpha[:, :, 0] * 255)

            if self.render_raw_rgb:
                img_path = join(cam_path,
                                f"{self.action_frame_index_single}_{self.cur_ind}_{str(self.frame).zfill(4)}_"
                                f"{self.stage}_{self.epoch}_{len(self.cameras)}_raw.png")
                cv2.imwrite(img_path, cv2.cvtColor(background, cv2.COLOR_RGB2BGR))







    def process_display_images(self, cam_key, mesh_in_image):
        """
        Process the rendered images based on settings
        :param cam_key:
        :param mesh_in_image:
        :return:
        """

        mesh_img = mesh_in_image[0, ...].cpu().numpy().astype(dtype=np.uint8)

        # if not self.action_loader.load_undist_rgb:
        #
        #     if self.stage != 'Time':
        #         raise ValueError("This is currently not implemented!")
        #
        #     else:
        #         # Load the original format without max img_size and crop, then distort the image
        #         pass

        # Split the mesh into RGB and alpha channels
        mesh_rgb = mesh_img[:, :, :3]

        bg_color = np.ones(3)
        mesh_alpha = np.expand_dims(np.all(mesh_rgb != bg_color, axis=-1), axis=-1).astype(float)
        mesh_alpha = np.repeat(mesh_alpha, 3, axis=-1)

        if self.single_view:

            frame_in_data = self.action_and_keyframes[self.action_frame_index][0]

            background = cv2.cvtColor(cv2.imread(join(self.action_loader.masks_path,
                                                      f"{str(frame_in_data).zfill(7)}_i.png")), cv2.COLOR_BGR2RGB)

            # Ensure both images are the same size
            background = cv2.resize(background, (mesh_img.shape[1], mesh_img.shape[0]))

            # Perform alpha blending to combine the images
            foreground = cv2.multiply(mesh_alpha, mesh_rgb.astype(float))
            background_ch = cv2.multiply(1.0 - mesh_alpha, background.astype(float))
            combined = cv2.add(foreground, background_ch).astype(np.uint8)

            self.save_mesh_on_images_single_view(image_combined=combined, image_orig=background,
                                                 mesh_image=foreground.astype(np.uint8))

            # self.save_mesh_on_image(cam_name=cam_key, image_combined=combined)

        else:

            if self.CROP and self.action_loader.load_undist_rgb:
                background = self.action_loader.rgbs[self.frame][cam_key][self.cur_ind]
            else:
                # Loads the frame from the video file.
                # background = self.get_background_image_non_single_views(cam_key)
                background = self.action_loader.undist_rgbs[cam_key][self.frame]

            if not self.action_loader.load_undist_rgb:
                ## Undistort the image
                big_cam = self.action_loader.big_cams.cam_dict[cam_key]

                if cam_key not in self.undistort_maps.keys():

                    # Create the undistort map
                    w, h = big_cam.im_size

                    ## Exec is 15ms, compared to 40ms for cv2 undist

                    # Initialize the rectify map
                    map1, map2 = cv2.initUndistortRectifyMap(big_cam.K, big_cam.d, None, big_cam.K,
                                                             (w, h), cv2.CV_32FC1)

                    # Use it
                    background = cv2.remap(background, map1, map2, interpolation=cv2.INTER_LINEAR)
                    # Assign to undistort maps
                    self.undistort_maps[cam_key] = [map1, map2]

                else:
                    # Retrieve the undistort map
                    map1, map2 = self.undistort_maps[cam_key]

                    # Use it, 4ms exec
                    background = cv2.remap(background, map1, map2, interpolation=cv2.INTER_LINEAR)
                # Just undistort the background image,
                # background = cv2.undistort(background, big_cam.K, big_cam.d)

            background = utils.convert_tensor_to_numpy(background)

            background_np_copy = copy.copy(background)

            # Ensure both images are the same size
            background = cv2.resize(background, (mesh_img.shape[1], mesh_img.shape[0]))

            # Perform alpha blending to combine the images
            foreground = cv2.multiply(mesh_alpha, mesh_rgb.astype(float))

            if self.stage != "Color":
                background = cv2.multiply(1.0 - mesh_alpha, background.astype(float))
                combined = cv2.add(foreground, background).astype(np.uint8)

                if self.plot_skeleton and self.action_loader.load_undist_rgb:
                    # Put circles at keypoint locations gt

                    # Get the height and width of the image
                    height, width = combined.shape[:2]

                    # Use a fraction of the larger dimension as the radius
                    max_dim = max(width, height)
                    radius = int(max_dim * 0.015)

                    for (x, y) in self.action_loader.keypoints_2d[self.frame][cam_key][self.cur_ind]:
                        cv2.circle(combined, (int(x), int(y)), 3, (0, 0, 255), -1)

                if not self.action_loader.load_undist_rgb and self.render_bbox:
                    utils.draw_inset_bounding_box(combined,
                                                  bool(self.camera_selection_matrix[self.frame, int(cam_key) - 1]))

                # combined = mesh_rgb - 1 + background
                # Get the actual frame of the video by selecting the appropriate frame by addition
                # Only change values of the video frame where the mask is
                self.save_mesh_on_image(cam_name=cam_key, image_combined=combined, mesh_alpha=mesh_alpha,
                                        background=background_np_copy, mesh_rgb=foreground)
            else:

                render_path = join(self.action_loader.action_path, "rendered_images_cropped_new")

                cam_path = join(render_path, f"{cam_key.zfill(2)}")

                img_path = join(cam_path, f"{self.cur_ind}_{str(self.frame).zfill(4)}_"
                                          f"cropped.png")

                # Save the original cropped image
                cv2.imwrite(img_path, cv2.cvtColor(background.astype(np.uint8), cv2.COLOR_RGB2BGR))

                img_path = join(cam_path, f"{self.cur_ind}_{str(self.frame).zfill(4)}_"
                                          f"{self.stage}_{self.epoch}_{len(self.cameras)}_wo_bg.png")
                # Save the rendered mesh
                cv2.imwrite(img_path, cv2.cvtColor(foreground.astype(np.uint8), cv2.COLOR_RGB2BGR))

                background = cv2.multiply(1.0 - mesh_alpha, background.astype(float))
                combined = cv2.add(foreground, background).astype(np.uint8)

                # combined = mesh_rgb - 1 + background
                # Get the actual frame of the video by selecting the appropriate frame by addition
                # Only change values of the video frame where the mask is
                self.save_mesh_on_image(cam_name=cam_key, image_combined=combined, mesh_alpha=mesh_alpha)



    def display_mesh_on_images(self, alternate_img_name=None):

        # Updates the cropping information for saving the mesh
        self.action_loader.load_YOLO()

        with torch.no_grad():


            mesh_in_image_list = []

            if self.action_loader.same_size:
                # Batched
                # Not only cams but also frames!!!

                meshes = []
                for frame_index, frame in enumerate(self.bundled_frames):
                    meshes.append(self.posed_mesh[frame_index].extend(len(self.cameras_to_retain[0])))

                meshes = pytorch3d.structures.join_meshes_as_batch(meshes)


                batched_cameras = self.batched_cameras[self.flattened_idx]



                # Create a disp renderer with Point lights
                lights_example = PointLights(device=self.device, location=[[0, 0, 0]])


                if self.action_loader.load_undist_rgb:
                    im_h, im_w = batched_cameras.image_size[0, :].cpu()
                else:
                    im_h, im_w = batched_cameras.image_size[0, :].cpu() // self.image_scale


                disp_renderer = get_rgb_renderer(batched_cameras[0], lights_example,
                                                 display_img_size=(int(im_h), int(im_w)))

                batched_cameras_sel = {}
                batched_frames_sel = {}

                # Get all the cameras to be rendered
                cams_to_be_rendered = list(set([x for cams_per_frame in self.cameras_to_retain for x in cams_per_frame]))

                for cam_render_key in cams_to_be_rendered:

                    batch_sel_indices = []
                    batch_frame_indices = []

                    for frame_index, frame in enumerate(self.bundled_frames):
                        self.frame = frame
                        for cam_key_index, cam_key in enumerate(self.cameras_to_retain[frame_index]):
                            cam_iter_index = frame_index * len(self.cameras_to_retain[0]) + cam_key_index

                            if cam_key == cam_render_key:
                                batch_sel_indices.append(cam_iter_index)
                                batch_frame_indices.append(frame_index)

                    batched_cameras_sel[cam_render_key] = batch_sel_indices
                    batched_frames_sel[cam_render_key] = batch_frame_indices

                for cam_key_index, cam_key in enumerate(self.cameras_to_retain[frame_index]):

                    #batched_cameras_sel =
                    cam_loc = torch.from_numpy(self.action_loader.big_cams.get_camera_in_world(cam_key).T).to(self.device)

                    indices_meshes = batched_frames_sel[cam_key]
                    indices_cams = batched_cameras_sel[cam_key]

                    # Repeat light locations to match the batch size of indices_cams
                    #num_cams = len(indices_cams)  # Determine batch size of cameras or meshes
                    #light_location_batched = cam_loc.unsqueeze(0).repeat(num_cams, 1, 1)  # Repeat for batch size
                    #light_location_batched = light_location_batched.view(-1, 3)

                    # Not for the entirety of the dataset, before cam loc only
                    light_pos = cam_loc + torch.tensor([[0, 0, 0.5]], device=self.device)

                    light = PointLights(device=self.device, location=light_pos)
                    #light = AmbientLights(device=self.device)

                    #light = [light for x in range(len(indices_cams))]

                    if self.render_bright:
                        # Otherwise the light pose is not updated...
                        disp_renderer = get_rgb_renderer(batched_cameras, light,
                                                         display_img_size=(int(im_h), int(im_w)))

                    rgb_preds = disp_renderer(meshes[indices_cams], cameras=batched_cameras[indices_cams], light=light)

                    for frame_iter_index, frame_index in enumerate(indices_meshes):

                        self.frame = self.bundled_frames[frame_index]
                        self.action_frame_index_single = self.action_frame_index[frame_index]

                        mesh_in_image = rgb_preds[frame_iter_index, :, :, :][None]
                        self.process_display_images(cam_key, mesh_in_image)

            else:
                # Unbatched

                for frame_index, frame in enumerate(self.bundled_frames):
                    self.frame = frame

                    posed_mesh = self.posed_mesh[frame_index]


                    if self.action_loader.labels_only:
                        # There is no frame specific camera selection process
                        cameras_to_retain = self.cameras_to_retain
                    else:
                        cameras_to_retain = self.cameras_to_retain[frame_index]

                    for cam_key_index, cam_key in enumerate(cameras_to_retain):
                        cam_iter_index = frame_index * self.NUM_CAMS + cam_key_index
                        frame_cam_key = f"{frame}_{cam_key}"
                        self.action_frame_index_single = self.action_frame_index[frame_index]


                        cam_to_render = self.cameras[frame_cam_key]


                        if self.single_view:
                            cam = cam_to_render
                            #cam = cam_calib.change_focal_length_pytorch3d(cam,
                            #                                              self.focal_lengths[self.action_frame_index, :, :])

                            sil_renderer = self.silhouette_renderer[frame_cam_key]
                            #sil_renderer, disp_renderer = self.single_view_rend.get_renderers(cam)

                            # Put here a point light from the camera location
                            cam_loc = -cam.R[0,:,:]@cam.T.T

                            light = PointLights(device=self.device, location=cam_loc.T)
                            mesh_in_image = self.display_renderer[frame_cam_key](posed_mesh, cameras=cam, lights=light)

                        else:
                            # The lights scale heavily memory consumption
                            # mesh_in_image = self.display_renderer[cam_key](posed_mesh, cameras=cam_to_render,
                            #                                                lights=PointLights(device=self.device,
                            #                                                                   location=[[0.0, 0.0, 10.0]]))
                            # mesh_in_image = self.display_renderer[cam_key](posed_mesh, cameras=cam_to_render,
                            #                                                lights=self.lights)

                            cam_loc = self.action_loader.big_cams.get_camera_in_world(cam_key).T
                            light = PointLights(device=self.device, location=cam_loc)

                            # This should already contain the correct light for that viewpoint that aligns with the camera origin
                            mesh_in_image = self.display_renderer[frame_cam_key](posed_mesh, cameras=cam_to_render, lights=light)


                        self.process_display_images(cam_key, mesh_in_image)





    def get_background_image_non_single_views(self, cam_key):

        min_frames_action = int(min(self.action_loader.additive_frames_cam.values()))

        # Get the video files in the action path folder
        video_files = [x for x in os.listdir(self.action_loader.action_path) if '.mp4' and 'CoreView' in x]

        # Get the position of the second underscore
        underscore_pos = find_second_underscore(video_files[0])

        video_name = video_files[0][:underscore_pos + 1] + f"{cam_key}.mp4"

        # Open the video with opencv
        if self.action_loader.additive_frames_cam[cam_key] > min_frames_action:
            add = 1
        else:
            add = 0

        unaligned_frame = self.frame + add
        # Load the background image
        background = extract_frame(join(self.action_loader.action_path, video_name),
                                   frame_number=unaligned_frame)



        return background

    def get_kernel(self):

        sigma = 1
        truncate = 3.0  # where to clip the filter (truncate) at this many std
        sd = float(sigma)
        lw = int(truncate * sd + 0.5)

        order = 0  # An order of 0 corresponds to convolution with a Gaussian kernel. A positive order corresponds to convolution with that derivative of a Gaussian.
        kernel_weights = _gaussian_kernel1d(sigma, order, lw)[::-1]

        return torch.from_numpy(np.array(kernel_weights))

    def get_intermediate_terms(self):

        go_i = torch.zeros_like(self.global_orientation)
        gt_i = torch.zeros_like(self.global_t)
        gs_i = torch.zeros_like(self.global_scale)
        bp_i = torch.zeros_like(self.body_pose)
        bl_i = torch.zeros_like(self.bone_length)
        vo_i = torch.zeros_like(self.vertex_offsets)

        return go_i, gt_i, gs_i, bp_i, bl_i, vo_i

    def update_parameters_convolution(self, frame, start_range, end_range, reduced_kernel):

        go_i, gt_i, gs_i, bp_i, bl_i, vo_i = self.get_intermediate_terms()

        # make the reduced kernel so that it adds up to 1! otherwise weird scaling behavior

        normalized_kernel = reduced_kernel / reduced_kernel.sum()

        for kernel_index, red_frame in enumerate(range(start_range, end_range)):
            # self.model_params_time_smoothed[frame] += self.model_params_time[red_frame] * reduced_kernel[red_frame]

            go_i += self.model_params_time[red_frame][0] * normalized_kernel[kernel_index]
            gt_i += self.model_params_time[red_frame][1] * normalized_kernel[kernel_index]
            gs_i += self.model_params_time[red_frame][2] * normalized_kernel[kernel_index]
            bp_i += self.model_params_time[red_frame][3] * normalized_kernel[kernel_index]
            bl_i += self.model_params_time[red_frame][4] * normalized_kernel[kernel_index]
            vo_i += self.model_params_time[red_frame][5] * normalized_kernel[kernel_index]

        self.model_params_time_smoothed[frame] = [go_i, gt_i, gs_i, bp_i, bl_i, vo_i]

    def smooth_parameters(self):

        kernel = self.get_kernel()

        self.model_params_time_smoothed = {}

        lt = self.model_params_time.keys().__len__()

        kernel_size = kernel.size(0)
        pad = kernel_size // 2

        print("Start Smoothing process")
        initial_frame = list(self.model_params_time.keys())[0]

        # Shift the keys to 0:
        self.model_params_time = {key - initial_frame: val for key, val in self.model_params_time.items()}

        for i in tqdm(range(lt)):
            if i < pad:
                skip = pad - i
                reduced_kernel = kernel[skip:]

                self.update_parameters_convolution(i, 0, reduced_kernel.size(0), reduced_kernel)

                # reduced_time_series = time_series[0:reduced_kernel.size(0)]
                # self.model_params_time_smoothed[i] = torch.sum(reduced_kernel * reduced_time_series)

            elif i >= lt - pad:

                # For even kernels
                if kernel.size(0) % 2 == 0:
                    clip_index = lt - (i + pad)
                else:
                    clip_index = lt - (i + pad + 1)

                if clip_index == 0:
                    reduced_kernel = kernel
                else:
                    reduced_kernel = kernel[:clip_index]

                start_index_series = i - pad

                self.update_parameters_convolution(i, start_index_series,
                                                   reduced_kernel.size(0) + start_index_series, reduced_kernel)

                # for red_frame in range(start_index_series, reduced_kernel.size(0) + start_index_series, 1):
                #    self.model_params_time_smoothed[i] += self.model_params_time[red_frame] * reduced_kernel[red_frame]

                # reduced_time_series = time_series[start_index_series:reduced_kernel.size(0) + start_index_series]
                # self.model_params_time_smoothed[i] = torch.sum(reduced_kernel * reduced_time_series)

            else:
                ## Not in padding area weighting
                start_index_series = i - pad

                self.update_parameters_convolution(i, start_index_series,
                                                   reduced_kernel.size(0) + start_index_series, kernel)

                # for red_frame in range(start_index_series, reduced_kernel.size(0) + start_index_series, 1):
                #    self.model_params_time_smoothed[i] += self.model_params_time[red_frame] * kernel[red_frame]

                # reduced_time_series = time_series[start_index_series:kernel.size(0) + start_index_series]
                # self.model_params_time_smoothed[i] = torch.sum(kernel * reduced_time_series)

        # Shift the keys to 0:
        self.model_params_time = {key + initial_frame: val for key, val in self.model_params_time.items()}
        self.model_params_time_smoothed = {key + initial_frame: val for key, val in
                                           self.model_params_time_smoothed.items()}

        # Assign smooth parameters to time
        self.model_params_time = self.model_params_time_smoothed

    def plot_time_as_images(self, smooth=False):
        """

        :return:
        """

        # Set the smoothing parameter
        self.smooth = smooth

        ## Load parameters from file
        if os.path.exists(self.model_time_opt_save_path):
            self.model_params_time = torch.load(self.model_time_opt_save_path, map_location=self.device)

        # Check if there are any parameters over time
        if not self.model_params_time:
            raise ValueError("There are not parameters for the time sequence")

        # If we want smooth the parameters
        if self.smooth:
            self.smooth_parameters()

        print("Rendering Time Sequence")
        # Iterate the frames and create images
        for frame in tqdm(self.model_params_time.keys()):

            # Set the current frame
            self.frame = frame

            # Set the model parameters for that time
            self.model_parameters = self.model_params_time[frame]

            if self.smooth:
                (self.global_orientation, self.global_t, self.global_scale, self.body_pose,
                 self.bone_length, self.vertex_offsets) = self.model_parameters

            else:
                (self.global_orientation, self.global_t, self.global_scale, self.body_pose,
                 self.bone_length, self.vertex_offsets, self.posed_3d_keypoints) = self.model_parameters

            # Pose the model
            self.__pose_mesh__()

            # Render the images
            self.display_mesh_on_images()




