import numpy as np
import pytorch3d.io
import torch
from torch.nn import functional as F
import os
import json
from scipy.spatial.transform import Rotation as R
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, Textures
from pytorch3d.io.ply_io import MeshPlyFormat
from iopath.common.file_io import PathManager
from pytorch3d.io import IO
from os.path import join
import pose_reconstruction.src.utils.utils as utils
from einops import rearrange


def build_weighted_adjacency_matrix(neighbors, num_vertices):
    """
    Build a weighted adjacency matrix from neighbor information.

    Parameters:
    - neighbors: List of tensors with indices of neighbors for each vertex.
    - num_vertices: Total number of vertices.

    Returns:
    - A dense tensor representing the weighted adjacency matrix.
    """
    adjacency_matrix = torch.zeros((num_vertices, num_vertices))

    for i, neigh_indices in enumerate(neighbors):
        if len(neigh_indices) > 0:
            # Set the adjacency value to 1 divided by the number of neighbors for each connected vertex
            adjacency_matrix[i, neigh_indices] = 1 / len(neigh_indices)

    return adjacency_matrix


class MeshModel():
    def __init__(self, device=torch.device('cpu'), mesh=None, shift_initially=True, load_segments=False,
                 convert_trimesh=True):
        self.device = device

        # read in bird_mesh from the same dir
        this_dir = os.path.dirname(__file__)
        mesh_file = os.path.join(this_dir, mesh)

        self.mesh_dir = os.path.dirname(mesh_file)

        with open(mesh_file, 'r') as infile:
            dd = json.load(infile)

        self.dd = dd

        # Converts the faces to Trimesh-faces
        mesh = trimesh.Trimesh(self.dd["V"], self.dd["F"])

        self.mesh = mesh

        ## For the casa meshes the rearranging led to errors in the face array!
        if convert_trimesh:
            self.dd["F"] = mesh.faces
        else:
            self.dd["F"] = np.array(self.dd["F"])


        self.faces = torch.as_tensor(self.dd["F"])

        self.kintree_table = torch.tensor(dd['kintree_table']).to(device)
        self.parents = self.kintree_table[0]




        # Make a list of pairs of adjacent faces like [1, 2], [1, 3], [1, 4], but not [4,1] so only directed
        self.adjacent_faces = []


        self.weights = torch.tensor(dd['weights']).to(device)
        #self.vert2kpt = torch.tensor(dd['vert2kpt']).to(device)

        self.J = torch.tensor(dd['J']).unsqueeze(0).to(device)
        self.V = torch.tensor(dd['V']).unsqueeze(0).to(device)



        self.num_vertices = self.V.shape[1]

        # blueish [0, 172, 223]
        # greenish [0, 223, 120]
        default_colors = [0, 223, 120]

        self.default_MESH_COLOR = torch.tensor(default_colors, device=self.device)
        self.vert_colors = torch.ones_like(self.V) * self.default_MESH_COLOR  # (1, V, 3)


        self.default_vertex_colors = torch.ones_like(self.V) * self.default_MESH_COLOR#self.default_MESH_COLOR.unsqueeze(0).repeat(1, self.dd["V"].__len__(), 1).to(device)

        self.segm_color_texture = 0
        self.bp_color_mult_index_map = {}
        self.vertices_as_dicts = {}
        if load_segments:
            self.load_vertex_segments()



        self.src_vertices = []
        self.target_neighbors = []

        #self.create_src_target_indices_shape()

        self.adjacency_matrix_vertices = build_weighted_adjacency_matrix(self.mesh.vertex_neighbors, self.V.shape[1]).to(device)


        # Compute vertex normals
        self.posed_mesh = Meshes([self.V.squeeze()], faces=[torch.as_tensor(self.dd["F"], device=self.device)])


        self.V_normals = self.posed_mesh.verts_normals_list()[0]
        #posed_mesh = posed_mesh.to(self.device)

        self.f = torch.from_numpy(self.dd["F"]).to(self.device).unsqueeze(0)


        hip = self.J[0, 0, :].clone()

        # Center the hip to be at 0
        # hip = self.J[0, 0, :].clone()
        self.V -= hip
        self.J -= hip


        # Shift the Joints, and Vertices initially depending on the camera setup
        if shift_initially:

            # Center the model
            centroid = torch.mean(self.V, dim=1)
            self.V = self.V - centroid #- 0.75 * torch.tensor([0, 0, 1], device=device)
            self.J = self.J - centroid #- 0.75 * torch.tensor([0, 0, 1], device=device)

            # Scale the model a bit
            self.J *= 0.1
            self.V *= 0.1


        # Align with Low Poly, due to an initial misalignment between high and low poly version
        hip = self.J[0, 0, :].clone()

        target_hip = torch.tensor(
            [3.9462e-11, 2.4515e-01, -1.9256e-01],
            device=self.device,
            dtype=self.J.dtype,
        )

        hip_delta = target_hip - hip
        self.V += hip_delta
        self.J += hip_delta



        self.LBS = LBS(self.J, self.parents, self.weights, self.device)

        self.color_weights = None



    def load_vertex_segments(self):


        segmented_vertices_path = join(self.mesh_dir, "SegmentedVertices")

        json_files = [x for x in os.listdir(segmented_vertices_path) if '.json' in x]


        for json_file in json_files:
            bp = json_file[:-5]
            self.vertices_as_dicts[bp] = utils.load_json_from_path(join(segmented_vertices_path, json_file))[bp]

    def set_weight_texture(self, color_weights):

        # Set the new color weights
        self.color_weights = color_weights

        # Init the color texture
        self.segm_color_texture = torch.zeros(size=(1, self.V.shape[1], 3), device=self.device)

        # Create the bp colors
        for bp_index, (bp, vertices) in enumerate(self.vertices_as_dicts.items()):

            # self.bp_color_mult_index_map[bp] = bp_index + 1
            #
            # # Create the color map
            # color_value = 15 * self.bp_color_mult_index_map[bp]

            weight_value = self.color_weights[bp]
            #print(f"Bp: {bp} with weight: {weight_value}")

            for v_id in vertices:
                self.segm_color_texture[0, v_id, :] = weight_value * torch.ones(3)


    def create_src_target_indices_shape(self):

        # Prepare tensors for source vertices and their corresponding neighbors
        src_vertices = []
        target_neighbors = []

        for v_index in range(self.num_vertices):
            n_indices = self.mesh.vertex_neighbors[v_index]
            if len(n_indices) > 0:
                # Repeat the current vertex index for each of its neighbors
                src_vertices.extend([v_index] * len(n_indices))
                # Append the neighbor indices
                target_neighbors.extend(n_indices)

        # Convert lists to tensors
        self.src_vertices = torch.tensor(src_vertices, dtype=torch.long)
        self.target_neighbors = torch.tensor(target_neighbors, dtype=torch.long)


    def __call__(self, body_pose, bone_length, global_orientation, global_t, vertex_normal_weights, scale=1,
                 pose2rot=True, return_mesh=False,  save_mesh_template=None, vertex_normals=False,
                 body_pose_sparse_P=None, sparsify_roll=None, sparsify_roll_index=None):


        batch_size = body_pose.shape[0]


        if vertex_normals:

            ## Reshape the vertices individually by the normals of each vertex and a scaling
            V = self.V + vertex_normal_weights * self.V_normals

        else:
            ## Reposition the vertices
            V = self.V + vertex_normal_weights


        rest_pose_shifted_V = V.clone().detach()


        #V = self.V.repeat([batch_size, 1, 1]) * scale
        V = V.repeat([batch_size, 1, 1]) * scale

        bone_length = bone_length.repeat([batch_size, 1])
        # concatenate bone and pose
        bone = torch.cat([torch.ones([batch_size, 1], device=self.device), bone_length], dim=1)
        #pose = torch.cat([global_pose, body_pose], dim=1)

        # LBS
        verts, joint_transforms = self.LBS(V, body_pose, bone, scale, to_rotmats=pose2rot,
                                           sparsify_pose=body_pose_sparse_P, sparsify_roll=sparsify_roll,
                                           sparsify_roll_index=sparsify_roll_index)




        # Apply global transform
        #batched_global_pose = batch_rodrigues(global_orientation.view(-1, 3))
        # verts = torch.transpose(batched_global_pose @ torch.transpose(verts, 1, 2),
        #                         1, 2) + rearrange(global_t, 'b d -> b 1 d')
        #
        # # Joint transformations
        # joints_coordinates_posed = joint_transforms[:, :, :3, 3]
        # joints_coordinates_posed = torch.transpose(
        #     batched_global_pose @ torch.transpose(joints_coordinates_posed, 1, 2),
        #     1, 2) + rearrange(global_t, 'b d -> b 1 d')

        # Apply global transform
        batched_global_pose = batch_rodrigues(global_orientation.view(-1, 3))

        # Rotate vertices
        verts = batched_global_pose @ torch.transpose(verts, 1, 2)
        # Shift vertices
        verts = torch.transpose(verts, 1, 2) + global_t


        # Joint transformations
        joints_coordinates_posed = joint_transforms[:, :, :3, 3]

        # Rotate joints
        joints_coordinates_posed = batched_global_pose @ torch.transpose(joints_coordinates_posed, 1, 2)

        joints_coordinates_posed = torch.transpose(joints_coordinates_posed,1, 2) + global_t



        if not return_mesh:

            return verts, rest_pose_shifted_V
        else:

            f = self.f.expand(batch_size, -1, -1)
            t = self.vert_colors.expand(batch_size, -1, -1)
            textures = TexturesVertex(verts_features=t)
            posed_mesh = Meshes(verts, faces=f, textures=textures)


            if save_mesh_template is not None:
                # Accepts (V,3) and (F,3)
                IO().save_mesh(posed_mesh, save_mesh_template, binary=False, colors_as_uint8=True)


            return verts, posed_mesh, rest_pose_shifted_V, joints_coordinates_posed

    def get_segmented_posed_mesh(self, posed_mesh):

        posed_mesh_copy = posed_mesh.clone()

        posed_mesh_copy.textures = Textures(verts_rgb=self.segm_color_texture)

        return posed_mesh_copy


    def get_nb_bones(self):
        return self.J.shape[1] - 1

    def get_simple_pose(self):


        nb_joints = self.J.shape[1]

        # Body posture
        body_pose = torch.zeros(size=(1, 3 * nb_joints))
        body_pose = body_pose.to(self.device)

        # Bone lengths
        bone_length = torch.ones(size=(1, nb_joints - 1))
        bone_length = bone_length.to(self.device)

        return body_pose, bone_length


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def remove_roll_from_rot(R):
    """

    :param R: [B, 3, 3]
    :return: R without roll
    """

    # Step 1: Extract yaw (psi) and pitch (theta)
    psi = -torch.atan2(R[:, 0, 2], R[:, 1, 2])
    theta = torch.arccos(R[:, 2, 2])

    psi = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    theta = torch.asin(-R[:, 2, 0])


    phi = torch.atan2(R[:, 2, 0], R[:, 2, 1])

    # Create batched rotation matrices for yaw and pitch
    R_yaw = torch.zeros(R.shape, device=R.device, dtype=R.dtype)
    R_yaw[:, 0, 0] = torch.cos(psi)
    R_yaw[:, 0, 1] = -torch.sin(psi)
    R_yaw[:, 1, 0] = torch.sin(psi)
    R_yaw[:, 1, 1] = torch.cos(psi)
    R_yaw[:, 2, 2] = 1

    R_pitch = torch.zeros(R.shape, device=R.device, dtype=R.dtype)
    R_pitch[:, 0, 0] = torch.cos(theta)
    R_pitch[:, 0, 2] = torch.sin(theta)
    R_pitch[:, 1, 1] = 1
    R_pitch[:, 2, 0] = -torch.sin(theta)
    R_pitch[:, 2, 2] = torch.cos(theta)

    # Compute the final batched rotation matrices without roll
    R_no_roll = torch.bmm(R_yaw, R_pitch)

    return R_no_roll


def batch_rodrigues(theta, dtype=torch.float32):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """

    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat).float()



"""
Code taken and adapted from avian-mesh:
https://github.com/marcbadger/avian-mesh/blob/master/models/LBS.py
@Inproceedings{badger2020,
  Title          = {3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View},
  Author         = {Badger, Marc and Wang, Yufu and Modh, Adarsh and Perkes, Ammon and Kolotouros, Nikos and Pfrommer, Bernd and Schmidt, Marc and Daniilidis, Kostas},
  Booktitle      = {ECCV},
  Year           = {2020}
}
"""

class LBS():
    '''
    Implementation of linear blend skinning, with additional bone and scale
    Input:
        V (BN, V, 3): vertices to pose and shape
        pose (BN, J, 3, 3) or (BN, J, 3): pose in rot or axis-angle
        bone (BN, K): allow for direct change of relative joint distances
        scale (1): scale the whole kinematic tree
    '''

    def __init__(self, J, parents, weights, device):

        # J (BN, N_J, 3)

        # h_joints (BN, N_J, 4, 1)

        # J unsqueeze, expand dims at loc -1 --> J (BN, N_J, 3, 1)
        # pad zeros at pen-ultimate location h_joints --> (BN, N_J, 4, 1(
        self.h_joints = F.pad(J.unsqueeze(-1), [0, 0, 0, 1], value=0)
        self.n_joints = J.shape[1]


        # Kin tree (BN, N_J, 3, 1)
        # J[:, [0], :] is basically the first joint (there is no relative shift)
        # that is concatenated with the following joints that are substracted by the kinematic tree
        # Relative differences to parent only
        self.kin_tree = torch.cat([J[:, [0], :], J[:, 1:] - J[:, parents[1:]]], dim=1).unsqueeze(-1)

        # Parents as tensor
        self.parents = parents

        # Basically Inserts Dimension at beginning, here (1, N_V, N_J)
        self.weights = weights[None].float()

        self.device = device

    def __call__(self, V, pose, bone, scale, to_rotmats=True, sparsify_pose=None, sparsify_roll=None,
                 sparsify_roll_index=None):
        """

        :param V: (BN, N_V, 3) B_N == 1 in these scenarios?
        :param pose: (BN, 3*N_J), where first rotation is world rotation
        :param bone: (BN, N_J)
        :param scale: scalar
        :param to_rotmats: boolean if conversion from rodriguez to rotmat should be done
        :return:
        """



        # Get the batch_size by checking the vertices
        batch_size = len(V)
        # Get the device where poses are stored as tensors
        device = pose.device

        # V(B_N, N_V, 3) --> V (B_N, N_V, 4, 1), expanded last dim, and padded ones
        V = F.pad(V.unsqueeze(-1), [0, 0, 0, 1], value=1)

        # Expand Bone by two dimensions and apply scaling of single bones to kin_tree and global scale
        # kin_tree (B_N, N_J, 3, 1)
        kin_tree = (scale * self.kin_tree) * bone[:, :, None, None]

        #sparsify_pose = None

        if to_rotmats:
           pose_with_roll = batch_rodrigues(pose.view(-1, 3))   # Pose out: (N_J, 3, 3), Pose In (1, N_J*3)
           #pose_without_roll = remove_roll_from_rot(batch_rodrigues(pose.view(-1, 3)))

           #pose_with_roll = batch_rodrigues(rearrange(pose, 'b joints dim -> (b joints) dim'))

           if sparsify_pose is not None:
               #  Sparsify pose (N_J, N_J_red + 1), pose (N_J_red+1, 3, 3) last entry is zero vector (not necessary because of zeros)

               pose_with_roll = pose_with_roll.view(batch_size, -1, 3, 3)



               all_pose = torch.eye(3, device=self.device)#.reshape((batch_size, 3, 3)).repeat(sparsify_pose.__len__(), 1, 1).to(self.device)
               all_pose = all_pose.repeat(batch_size, sparsify_pose.shape[0], 1, 1)

               indices_sparse_pose = torch.where(sparsify_pose == 1)[0]
               all_pose[:, indices_sparse_pose, :, :] = pose_with_roll
               # For now disregarded, and pose changes are penalized more heavily
               #all_pose[torch.where(sparsify_roll == 1)] = pose_without_roll[torch.where(sparsify_roll_index == 1)]

               pose = all_pose
           else:
               pose = pose_with_roll


        # Pose (N_J, 3, 3), then (B_N, N_J, 3, 3)
        pose = pose.view([batch_size, -1, 3, 3])
        # Create Transformations tensor (B_N, N_J, 4, 4) (for homogeneous coordinates)
        T = torch.zeros([batch_size, self.n_joints, 4, 4], device=self.device, dtype=torch.float)
        # Put for every joint at lower right transformation matrix a 1, corresponds to shift
        T[:, :, -1, -1] = 1
        # Concat the rotation with the relative shift
        T[:, :, :3, :] = torch.cat([pose, kin_tree], dim=-1)

        # Get the root joint transformation
        T_rel = [T[:, 0]]
        # Iterate the number of joints, excluding the first joint
        for i in range(1, self.n_joints):
            # Take the parent transformation and apply the joint rotation
            T_rel.append(T_rel[self.parents[i]] @ T[:, i])
        # Concatenates list of relative transforms into one tensor
        T_rel = torch.stack(T_rel, dim=1)


        T_rel_save = T_rel.clone()


        # Combine with inverse rest pose transformation
        T_rel[:, :, :, [-1]] -= T_rel.clone() @ (self.h_joints * scale)

        # Reshape as tensor with collapsed transformation dimension (B_N, N_J, 4)
        T_ = self.weights @ T_rel.view(batch_size, self.n_joints, -1)

        # Build final transformation per vertex as linear combination given by weights
        T_ = T_.view(batch_size, -1, 4, 4)
        # Transform Vertices
        V = T_ @ V

        # The @ transform allows to do matrix multiplication of the last dimensions, here (V_N, 4, 4) @ (V_N, 4, 1)
        # And also incorporate batch size dimension, same was done with t_rel.clone etc.

        return V[:, :, :3, 0], T_rel_save
