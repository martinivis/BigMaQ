import matplotlib.pyplot as plt
import numpy as np
import pytorch3d.transforms
import torch
import torch.nn.functional as F
from torch.linalg import vector_norm
from pose_reconstruction.src.utils.LBS import MeshModel
import time
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
from pytorch3d.structures import Meshes
from einops import rearrange

BODY_PARTS_OMC = ['Right eye','Left eye','Nose','Head','Neck','Right shoulder','Right elbow','Right wrist',
                 'Left shoulder','Left elbow','Left wrist','Hip','Right knee','Right ankle','Left knee','Left ankle',
                 'Tail']

VERTEX_GROUPS =	{ "Right eye": [10187, 10195, 10219, 10227],
		 "Left eye": [1159, 1161, 1167, 1169],
		 "Nose": [1244, 5399, 5403, 9836],
		 "Head": [1016, 1643, 6948, 6957],
		 "Neck": [1565, 3285, 6634, 6710],
		 "Right shoulder": [6393, 6520, 6611, 6618],
		 "Right elbow": [5674, 6167, 6174, 6346],
		 "Right wrist": [7898, 7910, 7919, 7931],
		 "Left shoulder": [1560, 1561, 3119, 3185],
		 "Left elbow": [1331, 3000, 3003, 3095],
		 "Left wrist": [3863, 3869, 3874, 3880],
		 "Hip": [58, 2976, 5771, 6116],
		 "Right knee": [5565, 6447, 6458, 6460],
		 "Right ankle": [7576, 7805, 8317, 8330],
		 "Left knee": [1503, 2699, 3145, 3152],
		 "Left ankle": [526, 533, 535, 4077],
 "Tail": [1225, 1292, 3675, 7494]}

ANGLE_WEIGHTING = torch.Tensor([])



def angle_consistency(body_pose, device, coupled_indices, angle_lim=0.25*torch.pi):
    """

    :param body_pose: (1, 3*Nb_Red_Joints) Tensor
    :param device:
    :param coupled_indices: List of List of joint indices in sequence that should be coupled
    :return:
    """

    body_pose = body_pose.squeeze()

    angle_consis_loss = torch.Tensor([0]).to(device)

    for joint_seq in coupled_indices:
        for seq_index, joint_index in enumerate(joint_seq):

            first_rot_pose = body_pose[joint_index*3:(joint_index+1)*3]
            sec_rot_pose = body_pose[(joint_index+1)*3:(joint_index+2)*3]


            # Calculate the amplitude
            first_angle = torch.norm(first_rot_pose)
            second_angle = torch.norm(sec_rot_pose)


            angle_consis_loss += min_max_relu(first_angle-second_angle, low_lim=-angle_lim, up_lim=angle_lim)

    print(angle_consis_loss)
    return angle_consis_loss

def max_rotation_loss(body_pose, angle_lim=torch.pi):

    body_pose = body_pose.squeeze()

    nb_red_joints = body_pose.shape[0] // 3

    rot_angels = torch.zeros(nb_red_joints)

    for i in range(nb_red_joints):

        rot_angels[i] = torch.norm(body_pose[i*3:(i+1)*3])

    return torch.sum(min_max_relu(rot_angels, -angle_lim, angle_lim)**2)



def joint_angle_loss(body_pose, device, angle_weighting=None):


    # Expecting (1, 3*nb_joints) for bodypose
    assert body_pose.ndim == 2

    if angle_weighting is not None:

        body_pose = body_pose.squeeze()

        # Reshape body_pose to group elements into sets of 3
        body_pose_batched = body_pose.view(-1, 3).to(device)

        # Compute the norms for each batch of 3 elements
        body_pose_norms = torch.norm(body_pose_batched, dim=1)  # Norm along last dimension

        # Weight each norm by the corresponding angle_weighting
        weighted_norms = angle_weighting * body_pose_norms

        # Compute the final weighted loss
        angle_loss = weighted_norms.sum() / torch.sum(angle_weighting)

        return angle_loss
    else:
        print("Norm of body pose")
        return torch.norm(body_pose)



def skew_matrix(u):
    # u: (..., 3)
    zero = torch.zeros_like(u[..., 0])
    ux = torch.stack([
        zero,     -u[..., 2], u[..., 1],
        u[..., 2], zero,     -u[..., 0],
        -u[..., 1], u[..., 0], zero
    ], dim=-1)
    return ux.view(*u.shape[:-1], 3, 3)


def compute_angular_velocity(axis_angles: torch.Tensor, dt: float = 1.0, eps: float = 1e-6):
    """
    axis_angles: Tensor of shape (T, J, 3), axis-angle vectors (ϕ * u)
    dt: timestep
    Returns:
        angular velocity tensor of shape (T-1, J, 3)
    """

    axis_angles = axis_angles.view(axis_angles.size(0), -1, 3)

    angular_veloc_sum = 0

    for joint_idx in range(axis_angles.size(1)):


        joint_axisangle_time = axis_angles[:, joint_idx, :]


        # This is the amplitude or the rotation amount around the axis
        joint_angle_time =  torch.norm(joint_axisangle_time, dim=-1, keepdim=True).clamp(min=eps, max=2 * torch.pi)  # (T, 1)#torch.norm(joint_axisangle_time, dim=-1)[:, None]

        joint_axis_time = F.normalize(joint_axisangle_time, dim=-1, eps=eps)
        #joint_axis_time = joint_axisangle_time/joint_angle_time

        # finite sequence
        angle_fin = joint_angle_time[:-1]
        axis_fin = joint_axis_time[:-1]


        # Differences
        ang_dt = joint_angle_time[1:] - joint_angle_time[:-1]
        ax_dt = joint_axis_time[1:] - joint_axis_time[:-1]

        #
        identities = torch.eye(3, device=axis_angles.device).repeat(ang_dt.size(0), 1, 1)



        part1 = ang_dt * axis_fin

        interm_ = (torch.sin(angle_fin)[:, :, None] * identities + (1-torch.cos(angle_fin)[:, :, None]) * skew_matrix(axis_fin))

        part2 = torch.bmm(interm_, ax_dt.unsqueeze(-1)).squeeze(-1)


        angular_veloc =  part1 + part2

        angular_veloc_sum += torch.mean(torch.sum(angular_veloc**2, dim=-1))



    return angular_veloc_sum





def sm_time_loss_angles(param_over_time, squared=True):
    """
    Computes the mean squared dot product differences for the entire time sequence of a parameter
    :param param_over_time: Tensor of shape [num_frames, num_joints, 3]
                            where the last dimension corresponds to the rotation axis
    :param squared: If True, computes squared differences, otherwise raw differences
    :return: A scalar representing the time loss
    """

    # Reformat the angles
    param_over_time = param_over_time.view(param_over_time.size(0), -1, 3)

    # Get the normalized unit vectors for the parameter (rotation axes)
    param_normalized = torch.nn.functional.normalize(param_over_time, dim=-1)

    # Compute dot products for consecutive frames
    dot_products = torch.sum(
        param_normalized[:-1] * param_normalized[1:], dim=-1
    )  # Shape: [num_frames - 1, num_joints]

    # Compute differences from the dot product (e.g., deviation from 1, perfect alignment)
    diff = 1 - dot_products

    if squared:
        # Square the differences
        diff = diff ** 2

    # Mean across frames and joints
    loss = diff.sum()

    return loss



def sm_time_loss(param_over_time, squared=True, mean=True):
    """
    Computes the mean squared differences for the entire time sequence of a parameter
    :return:
    """

    # Differences across frames
    diff = torch.diff(param_over_time, dim=0)

    if squared:
        # Get the norm of them, per frame step
        l2_norm = torch.norm(diff, dim=-1) ** 2
    else:
        l2_norm = torch.norm(diff, dim=-1)

    if mean:
        # Divide by the sequence length - 1
        l2_time_normalized = l2_norm.sum() / diff.shape[0]
    else:
        l2_time_normalized = l2_norm.sum()

    return l2_time_normalized


def return_labeled_rig_joints(posed_joints, mesh, hierarchy_mapping):

    index_list = []

    for joint_str in hierarchy_mapping:
        index_list.append(mesh.dd["joint_to_index"][joint_str])

    return posed_joints[:, index_list, :]

def return_rig_selection(posed_joints, mesh, joints_list):

    index_list = []

    for joint_str in joints_list:
        index_list.append(mesh.dd["joint_to_index"][joint_str])

    return posed_joints[:, index_list, :]


def calculate_keypoints3D_from_posed_mesh(posed_verts, device):
    """

    :param posed_verts: (1, N_V, 3)
    :param device:
    :return: (1, N_K, 3)
    """


    keypoints_3D_posed = torch.zeros(posed_verts.shape[0], len(BODY_PARTS_OMC), 3)
    keypoints_3D_posed = keypoints_3D_posed.to(device)

    for bp_index, bp in enumerate(BODY_PARTS_OMC):

        vertices = VERTEX_GROUPS[bp]

        relevant_vertices_3D = posed_verts[:, vertices, :]

        keypoints_3D_posed[:, bp_index, :] = torch.mean(relevant_vertices_3D, dim=1)


    return keypoints_3D_posed


def keypoint_loss_2D(keypoints_pred, keypoints_gt, device, weighted=True,
                     keypoint_weights=None, det_confidences=None):
    """

    :param keypoints_pred: (1, 17, 3) Tensor, Keypoints in camera screen space
    :param keypoints_gt: (17, 2) Array, Keypoint labels
    :param device:
    :return:
    """
    #keypoints_gt_tensor = torch.from_numpy(keypoints_gt)
    #keypoints_gt_tensor = keypoints_gt_tensor.to(device)
    keypoints_gt_tensor = keypoints_gt

    # If it is
    # if not isinstance(keypoints_gt_tensor, torch.Tensor):
    #    keypoints_gt_tensor = torch.from_numpy(keypoints_gt).to(device)

    # Weighted
    if keypoint_weights is not None:

        if det_confidences is not None:
            # Extend by detection confidences
            weights = keypoint_weights * det_confidences

            loss = weighted_euclidean_loss(keypoints_pred[:, :2][None], keypoints_gt_tensor[None],
                                           weights.to(device))

        else:
            loss = weighted_euclidean_loss(keypoints_pred[:, :2][None], keypoints_gt_tensor[None],
                                       keypoint_weights.to(device))
    else:
        loss = mean_euclidean_distance(keypoints_pred[:, :2][None], keypoints_gt_tensor[None])




    return loss


def keypoints_in_image(keypoints, image_shape):
    """
    Returns the subset of `keypoints` (shape: (N, 2)) that lie
    within the image region defined by `image_shape[:2]`.

    :param keypoints:    NumPy array of shape (N, 2), each row [x, y].
    :param image_shape:  A tuple/list (H, W) or a shape with at least 2 elements.
    :return:             Subset of keypoints in the image region.
    """
    H, W = image_shape[:2]

    # Extract x and y in vectorized form
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]

    # Build boolean mask for valid points:
    #   0 <= x < W and 0 <= y < H
    valid_mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)

    # Return only those keypoints that are within the image bounds
    return valid_mask


def calculate_iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) of two binary masks.

    Parameters
    ----------
    mask1 : torch.Tensor
        The first binary mask.
    mask2 : torch.Tensor
        The second binary mask.

    Returns
    -------
    float
        IoU of mask1 and mask2.
    """

    # Ensure the masks are boolean tensors
    mask1 = mask1.bool()
    mask2 = mask2.bool()

    # Calculate intersection and union
    intersection = torch.logical_and(mask1, mask2)

    union = torch.logical_or(mask1, mask2)

    # Calculate IoU
    iou = torch.sum(intersection).float() / torch.sum(union).float()

    return iou

def get_IOU_single(predicted_image, binary_gt_mask, device):

    # Put the tensor on the GPU and reshape it
    gt_mask_tensor = torch.Tensor(binary_gt_mask)
    gt_mask_tensor = gt_mask_tensor.to(device)

    predicted_mask = predicted_image[0, :, :, 3]

    predicted_mask = predicted_mask != 0

    iou = calculate_iou(gt_mask_tensor, predicted_mask)

    return iou


def scaling_loss(scale, min_scale=0.5, max_scale=2):
    """
    Custom loss function for scale.

    Parameters:
    scale (torch.Tensor): The tensor containing the scale values.
    min_scale (float): The minimum acceptable scale value.
    max_scale (float): The maximum acceptable scale value.

    Returns:
    torch.Tensor: The calculated loss.
    """

    # Loss for values below the minimum scale
    loss_min = F.relu(min_scale - scale)

    # Loss for values above the maximum scale
    loss_max = F.relu(scale - max_scale)

    # Total loss is the sum of both losses
    total_loss = loss_min + loss_max

    return total_loss


def min_max_relu(value, low_lim, up_lim):

    # Loss for values below the minimum scale
    loss_min = F.relu(low_lim - value)

    # Loss for values above the maximum scale
    loss_max = F.relu(value - up_lim)

    # Total loss is the sum of both losses
    total_loss = loss_min + loss_max


    return total_loss


def quadratic_penalty_maximum(input_tensor, clipping_value):
    penalty = torch.relu(input_tensor - clipping_value)**2
    return penalty.sum()

def mean_euclidean_distance(t1, t2):
    """
    Mean Euclidean distance over keypoints, typically for two or three-dimensional keypoints
    :param t1: Shape (1, N_K, 2) or (1, N_K, 3)
    :param t2:
    :return:
    """

    assert (t1.dim() == 3 and t2.dim() == 3)

    return (1/t1.shape[1])*torch.sqrt(((t1 - t2)**2).sum(dim=-1)).sum()

def weighted_euclidean_loss(t1, t2, keypoint_weights):
    """
    Mean Euclidean distance over keypoints, typically for two or three-dimensional keypoints
    :param t1: Shape (1, N_K, 2) or (1, N_K, 3)
    :param t2:
    :return:
    """

    assert (t1.dim() == 3 and t2.dim() == 3)

    return (1 / keypoint_weights.sum()) * (torch.sqrt(((t1 - t2) ** 2).sum(dim=-1)) * keypoint_weights).sum()
    #return (1/t1.shape[1])*(torch.sqrt(((t1 - t2)**2).sum(dim=-1)) * keypoint_weights).sum()


def speed_lim_loss(posed_3d_keypoints, prev_posed_3d_kp, FPS=40, speed_lim=2.77):
    """

    :param posed_3d_keypoints:
    :param prev_posed_3d_kp:
    :param FPS: Recording Frame Rate
    :param speed_lim: In m/s, default 2.77 = 10km/h
    :return:
    """


    mean_distance_per_frame = mean_euclidean_distance(posed_3d_keypoints, prev_posed_3d_kp)

    mean_speed_per_frame = mean_distance_per_frame * FPS

    # Loss for mean_speed_per_frame larger speed_lim
    #speed_loss = 100 * F.relu(mean_speed_per_frame - speed_lim)
    speed_loss = quadratic_penalty_maximum(mean_speed_per_frame, speed_lim)

    return speed_loss

# Cage loss alone will not be sufficient, because as soon as there is no overlap between masks that could happen this
# diverges maybe into a small region of the cage, maybe initialize with a big scale then
def cage_loss(translation, x_range=[-1.3, 2],
              y_range=[-1.2, 1],
              z_range=[-1, 0.4]):


    loss_x = min_max_relu(translation[0, 0], x_range[0], x_range[1])
    loss_y = min_max_relu(translation[0, 1], y_range[0], y_range[1])
    loss_z = min_max_relu(translation[0, 2], z_range[0], z_range[1])

    return loss_x + loss_y + loss_z





def optimized_smoothness_loss(vertex_weights, adjacency_matrix):
    """
    Calculate the smoothness loss using an adjacency matrix.

    Parameters:
    - vertex_weights: Tensor of shape (num_vertices, 1) containing the weights of each vertex.
    - adjacency_matrix: Sparse tensor representing the adjacency matrix.

    Returns:
    - A scalar tensor representing the smoothness loss.
    """

    # Calculate the weight difference for each edge
    diff = adjacency_matrix @ vertex_weights - vertex_weights

    # Compute the smoothness loss as the sum of squared differences
    loss = (diff ** 2).sum()
    #loss = torch.exp(diff**2).sum()

    return loss



def normal_consistency_loss(mesh_model, vertices, adjacent=True, hull_loss=False, angle_threshold = 0.85):
    """

    :param mesh_model:
    :param vertices:
    :param adjacent:
    :param hull_loss:
    :param angle_threshold: 0.7 is approx 45 degrees, 0.5 is 60 degrees
    :return:
    """
    # Assuming `faces` is a tensor of shape [num_faces, 3], indexing into `vertices`
    # And `vertices` is a tensor of shape [num_vertices, 3] representing vertex positions

    faces = mesh_model.faces
    vertices = vertices.squeeze()

    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = normals / normals.norm(dim=1, keepdim=True)  # Normalize normals


    if adjacent:
    # Compute loss based on the angle between adjacent face normals
    # This requires determining adjacent faces, which depends on your mesh topology
    # For simplicity, let's assume we have a list of pairs of indices of adjacent faces
        adjacent_faces = mesh_model.mesh.face_adjacency  # This needs to be defined based on your mesh
        normal_dot_products = torch.sum(normals[adjacent_faces[:, 0]] * normals[adjacent_faces[:, 1]], dim=1)
    else:
        ## Compute the difference to the normals of the rest pose
        vertices_ref = mesh_model.V.squeeze()

        # Compute face normals
        v0 = vertices_ref[faces[:, 0]]
        v1 = vertices_ref[faces[:, 1]]
        v2 = vertices_ref[faces[:, 2]]
        normals_ref = torch.cross(v1 - v0, v2 - v0, dim=1)
        normals_ref = normals_ref / normals_ref.norm(dim=1, keepdim=True)

        normal_dot_products = torch.sum(normals * normals_ref, dim=1)

    # Then define a loss on the dot products of the normals, dot product is 1 if parallel

    #loss = ((1 - normal_dot_products) ** 2).mean()  # Encourage normals to be parallel

    loss = (1 - normal_dot_products).sum() ## According to BITE paper

    if not adjacent:
        return loss
    else:

        if hull_loss:
            # Penalizing sharp angles
            penalty = torch.relu(angle_threshold - normal_dot_products)
            print("Angle threshold: ", angle_threshold)
            return penalty.mean()
        else:
            # Adjacent faces
            return loss

def compute_edge_loss(vertices, mesh_model: MeshModel, variance=True):


    edges = mesh_model.mesh.edges

    # Extract the positions of the vertices forming each edge
    start_positions = vertices[edges[:, 0]]
    end_positions = vertices[edges[:, 1]]


    if not variance:
        # Calculate the Summed Euclidean distance between vertices of each edge
        edge_lengths = torch.sqrt(((end_positions - start_positions) ** 2).sum(dim=1))
        return edge_lengths.sum()
    else:
        # Variance of euclidean distance scaled by number of edges
        edge_lengths = torch.sqrt(((end_positions - start_positions) ** 2).sum(dim=1))
        # scalar
        var = torch.var(edge_lengths)
        loss = var * edge_lengths.shape[0]

        return loss



def index_by_length(lists):
    length_dict = {}
    for index, sublist in enumerate(lists):
        sublist_length = len(sublist)
        if sublist_length not in length_dict:
            length_dict[sublist_length] = []
        length_dict[sublist_length].append(index)
    return length_dict


def compute_center_deviation(vertices, vertex_index, mesh_model: MeshModel):
    """

    :param vertices: Vertices of either the template or shifted template pose, torch tensor Nx3
    :param vertex_index: The vertex index to calculate the neighborhood deviation from
    :param mesh_model: Mesh model object that contains neighborhood information
    :return:
    """

    return vertices[vertex_index, :] - torch.mean(vertices[mesh_model.mesh.vertex_neighbors[vertex_index]], dim= 0)

def compute_center_deviations_batched(vertices, mesh_model: MeshModel):
    """

    :param vertices: (N, 3) tensor
    :param mesh_model: Mesh model object
    :return:
    """
    vertex_indices = list(range(mesh_model.num_vertices))

    deviations = 0

    # Nx3
    vertices[vertex_indices]


    # Create different sets of neighboring vertices in the mesh, 5, 6, 7, etc.


    torch.mean(vertices)

    return deviations

def compute_center_by_selection_batched(all_vertices, vertices_selection, indices_vertices_selection):

    # Should be the same length for all lists
    N_selection_per_vertex = len(indices_vertices_selection[0])

    rows_to_select = torch.tensor(indices_vertices_selection, dtype=torch.int64)

    rows_to_select_flattened = rows_to_select.flatten()

    selected_rows = all_vertices[rows_to_select_flattened]

    selected_rows_tensor = selected_rows.view(len(indices_vertices_selection), N_selection_per_vertex, -1)

    centroids_per_row = torch.mean(selected_rows_tensor, dim=1)

    return all_vertices[vertices_selection] - centroids_per_row


def compute_laplacian(rest_pose_shifted_V, mesh_model: MeshModel, batched=True, smooth=False):
    """
    Computes Laplacian as cited in BITE paper
    :param rest_pose_shifted_V:
    :param mesh_model:
    :param batched: Compute it in a batched way to save time, 50x speed up
    :return:

    Example:
    a = torch.Tensor([[1, 1, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10]])
    second_dim = 3
    a = torch.rand(size=(4, second_dim))
    print(a)
    b = torch.rand(size=(4, second_dim))
    vertex_indices_list = [[0, 1, 2], [2, 3], [0, 1, 2, 3], [0, 1]]
    length_dictionary_to_vertex_index = index_by_length(vertex_indices_list)

    stacked_shift = torch.zeros(1, 3)

    for length_neighbors, vertex_index_list in length_dictionary_to_vertex_index.items():
        indices_vertices_selection = []
        for v_index in vertex_index_list:
            indices_vertices_selection.append(vertex_indices_list[v_index])
        length_template = compute_center_by_selection_batched(all_vertices=a,
                                                vertices_selection=vertex_index_list,
                                                indices_vertices_selection=indices_vertices_selection)
        print(length_template)
        length_shifted = compute_center_by_selection_batched(all_vertices=b,
                                                              vertices_selection=vertex_index_list,
                                                indices_vertices_selection=indices_vertices_selection)
        stacked_shift = torch.vstack((stacked_shift, length_template - length_shifted))
    length_loss = (stacked_shift**2).sum()
    print(length_loss)

    loss = 0
    for v_index in range(a.shape[0]):
        # Calculate the centroid deviation of meshes, shape=(3,)
        v_t_dist = compute_center_deviation(a, v_index, vertex_indices_list)

        print(v_t_dist)
        v_s_dist = compute_center_deviation(b, v_index, vertex_indices_list)
        # Should reduce to scalar
        loss += ((v_t_dist - v_s_dist)**2).sum()

    """

    mesh_template_vertices = mesh_model.V.squeeze()

    if smooth:
        ## Create meshes for pytorch3d laplace smoothing
        template_mesh = Meshes([mesh_template_vertices],
                               faces=[torch.as_tensor(mesh_model.dd["F"], device=mesh_model.device)])

        altered_mesh = Meshes([rest_pose_shifted_V],
                               faces=[torch.as_tensor(mesh_model.dd["F"], device=mesh_model.device)])


        template_mesh_laplace_error = mesh_laplacian_smoothing(template_mesh)
        altered_mesh_laplace_error = mesh_laplacian_smoothing(altered_mesh)

        #print(f"Template Mesh error: {template_mesh_laplace_error}")
        #print(f"Altered Mesh error: {altered_mesh_laplace_error}")

        return altered_mesh_laplace_error

    else:
        if batched:
            length_dictionary_to_vertex_index = index_by_length(mesh_model.mesh.vertex_neighbors)
            ## batched takes
            stacked_shift = torch.zeros(1, 3).to(mesh_model.device)

            for length_neighbors, vertex_index_list in length_dictionary_to_vertex_index.items():

                indices_vertices_selection = []

                for v_index in vertex_index_list:
                    indices_vertices_selection.append(mesh_model.mesh.vertex_neighbors[v_index])

                length_template = compute_center_by_selection_batched(all_vertices=mesh_template_vertices,
                                                        vertices_selection=vertex_index_list,
                                                        indices_vertices_selection=indices_vertices_selection)
                length_shifted = compute_center_by_selection_batched(all_vertices=rest_pose_shifted_V,
                                                                      vertices_selection=vertex_index_list,
                                                                      indices_vertices_selection=indices_vertices_selection)

                stacked_shift = torch.vstack((stacked_shift, length_template - length_shifted))

            return (stacked_shift**2).sum()
        else:

            stacked_shift = torch.zeros(1, 3).to(mesh_model.device)
            # Get vertex neighbors
            for v_index in range(mesh_model.num_vertices):

                # Calculate the centroid deviation of meshes, shape=(3,)
                v_t_dist = compute_center_deviation(mesh_template_vertices, v_index, mesh_model)
                v_s_dist = compute_center_deviation(rest_pose_shifted_V, v_index, mesh_model)

                stacked_shift = torch.vstack((stacked_shift, v_t_dist - v_s_dist))

            # Should reduce to scalar
            return (stacked_shift**2).sum()




def vertex_smooth_loss(vertex_weights, mesh_model: MeshModel,
                weighted_adjacency=False, optimized_squared=True):
    """
    Regularization for the vertex weights when learning individual body mesh models
    :param vertex_weights: Shape (N_V, 1)
    :param vertex_neighbors: (N_V, 3) vertices included in the face
    :return:

    Example:
    vertex_weights = torch.Tensor([[1], [2], [3], [1], [5]])

    vertex_neighbors = [[1, 2, 3], [1, 2, 3], [2, 3, 4], [3, 1, 0], [4, 1, 2]]
    loss = 29
    """

    if weighted_adjacency:


        smooth_loss = optimized_smoothness_loss(vertex_weights, mesh_model.adjacency_matrix_vertices)

    else:

        if not optimized_squared:

            smooth_loss = 0

            for v_index, neighbors in enumerate(mesh_model.mesh.vertex_neighbors):
                if len(neighbors) > 0:
                    neigh_weights = vertex_weights[neighbors]
                    diff = vertex_weights[v_index] - neigh_weights
                    smooth_loss += torch.sum(diff ** 2)

        else:
            """ Example calculation for dimension larger then 1 in vertex weights, when using offsets
            a = torch.Tensor([[1, 1], [2, 1]])
            b = torch.Tensor([[3, 0], [2, 2]])
            dif = a-b
            print(dif)
            print(dif ** 2)
            print(torch.sum(dif**2))
            """
            diffs = vertex_weights[mesh_model.src_vertices] - vertex_weights[mesh_model.target_neighbors]
            # Compute the smoothness loss as the sum of squared differences
            smooth_loss = torch.sum(diffs ** 2)


    return smooth_loss


def silhouette_segm_loss(predicted_bp_masks, binary_gt_mask, device, background_substr=False):

    gt_mask_tensor = torch.Tensor(binary_gt_mask)
    gt_mask_tensor = gt_mask_tensor.to(device)


    ## The differences
    alpha_diffs = (predicted_bp_masks[0, :, :, 3] - gt_mask_tensor) ** 2

    if background_substr:
        # Background is rendered as 1, substract first the background
        weights = (predicted_bp_masks[0, :, :, 0] - 1)
    else:
        weights = predicted_bp_masks[0, :, :, 0]

    weighted_sum =  weights * alpha_diffs

    # Normalize the weighted sum by the max value to be more controllable with alpha values
    loss = torch.mean(weighted_sum)

    return loss, weighted_sum#weighted_sum/weighted_sum.max()


def silhouette_loss(predicted_mask, binary_gt_mask, device, L2=True):


    # Put the tensor on the GPU and reshape it
    #gt_mask_tensor = torch.Tensor(binary_gt_mask)
    #gt_mask_tensor = gt_mask_tensor.to(device)

    gt_mask_tensor = binary_gt_mask
    # if not isinstance(gt_mask_tensor, torch.Tensor):
    #    gt_mask_tensor = torch.from_numpy(gt_mask_tensor).to(device)

    #predicted_mask = predicted_image[0, :, :, 3]

    if L2:
        # Mean didn't work as well
        #loss = ((predicted_mask - gt_mask_tensor) ** 2).mean()
        diff_map = (predicted_mask - gt_mask_tensor) ** 2
        loss = torch.mean(diff_map)
        #loss = torch.sum((predicted_mask - gt_mask_tensor) ** 2)
    else:
        # Looks way better with this
        #loss = F.smooth_l1_loss(predicted_mask, gt_mask_tensor, reduction='none').mean()


        # Get the points that are non-zero
        # make chamfer
        print("CHAMFER")
        loss = chamfer_distance(predicted_mask[None].view(1, -1, 1), gt_mask_tensor[None].view(1, -1, 1))[0]



    return loss, diff_map #+ 100*inter_loss # 100 seemed to work


def rgb_loss(input, target, mask, device):


    # if not isinstance(target, torch.Tensor):
    #    target = torch.from_numpy(target).to(device)
    #    mask = torch.from_numpy(mask).to(device)

    input = input[:, :, :, :3] * input[:, :, :, -1][:, :, :, None]   # get rid of alpha

    mask = mask / 255
    input = input / 255
    #input = rearrange(input, 'b h w c -> b c h w')

    target = target * mask[:, :, None]

    #target = torch.from_numpy(target).to(device)

    deviation = (target.to(device) - input.squeeze(axis=0)) ** 2


    return deviation.sum() / (mask.shape[0] + mask.shape[1])

    #return deviation.mean()

def mahalanobis_dist(vec, mean, covI):
    """
    Torch implementation of the mahalanobis distance
    :param vec:
    :param mean:
    :param cov:
    :return:
    """


    delta = vec - mean
    m = (delta @ covI) @ delta

    return torch.sqrt(m)




def narrow_gaussian(x, ell):
    return torch.exp(-0.5 * (x / ell) ** 2)









