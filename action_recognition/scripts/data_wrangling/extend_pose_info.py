"""
Extend the information about the pose into another file instead.
"""
import torch
import numpy as np

import os
from os.path import join

import pandas as pd
from scipy.spatial.transform import Rotation as R


base_path = r"/media/lucas/FastInternal/BigMacaque/ActionDataset"

csv_path    = "/media/lucas/FastInternal/BigMacaque/ActionDataset/tracked_actions.csv"

opt_cams_num = 6
# Save as action_params extended as npz

df = pd.read_csv(csv_path)

df = df[df["Session"].notna()]

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

#todo: save the global translation mean and variance of all centered translations.


all_translations = []

for idx, row in df.iterrows():

    path = row["path"]

    # --- load pose (same for all cams) ---
    action_params = [f for f in os.listdir(path)
                     if "action_params" in f and
                     f"{str(opt_cams_num).zfill(2)}.pth" in f]

    for ind_index in range(row["n_individuals"]):

        file_ending = action_params[ind_index]

        params = torch.load(join(path, file_ending), map_location="cpu", weights_only=False)

        pose_loaded = params["body_pose"] # (T,
        frame_seq_i = params["frame_sequence"].numpy()


        transl = params["t"].detach().numpy()
        global_rot = params["R"]

        #pose_loaded = torch.nan_to_num(pose_loaded, nan=0.0, posinf=0.0, neginf=0.0)
        #global_rot = torch.nan_to_num(global_rot, nan=0.0, posinf=0.0, neginf=0.0)

        assert not torch.isnan(pose_loaded).any(), "NaNs found in pose_loaded"
        assert not torch.isinf(pose_loaded).any(), "Infs found in pose_loaded"

        assert not torch.isnan(global_rot).any(), "NaNs found in global_rot"
        assert not torch.isinf(global_rot).any(), "Infs found in global_rot"

        T = pose_loaded.shape[0]

        # Saving the pose ----

        # Pose in: (T, 1, N_j*3)

        pose_matrix = batch_rodrigues(pose_loaded.view(-1, 3))


        assert not torch.isnan(pose_matrix).any(), "NaNs found in pose_matrix"
        assert not torch.isinf(pose_matrix).any(), "Infs found in pose_matrix"

        # Rearrange the pose to (T, N_J, 3, 3)
        pose_matrix = pose_matrix.view([T, -1, 3, 3])

        N_J = pose_matrix.shape[1]

        pose_matrix_flattened = pose_matrix.view(T, 1, N_J*9)

        # Put this to numpy
        pose_matrix_flattened = pose_matrix_flattened.detach().numpy()

        # Reshaping the global rot into 3x3
        global_rot_mat = batch_rodrigues(global_rot.squeeze())
        global_rot_mat_flattened = global_rot_mat.view(T, 1, 9).detach().numpy()

        assert not torch.isnan(global_rot_mat).any(), "NaNs found in global_rot_mat_flattened"
        assert not torch.isinf(global_rot_mat).any(), "Infs found in global_rot_mat_flattened"

        all_translations.append(transl.squeeze())

        np.savez(join(path, file_ending[:-4] + "_rotmat.npz"), pose_matrix=pose_matrix_flattened, frame_seq_i=frame_seq_i,
                 global_rot=global_rot_mat_flattened, transl=transl)

transl_vecs = np.vstack(all_translations)

mean_trans = np.nanmean(transl_vecs, axis=0)  # shape (3,)
std_trans  = np.nanstd(transl_vecs, axis=0)

np.savez(join(base_path, "trans_stats.npz"), mean_trans=mean_trans, std_trans=std_trans)


### Save the mean and variance of the translation
