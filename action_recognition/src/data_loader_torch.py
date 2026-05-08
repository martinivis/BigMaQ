import copy
import os
from os.path import join
import ast
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import action_recognition.src.utils as utils
from math import ceil
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import json

individuals_map = {"C": "Cuba",
                    "J": "Jekyll", "H": "Hyde",
                    "T": "Tonic", "G": "Gin", "L": "Libre", "O": "Odin", "N": "Nacho"}

ACTION_KEY_MAP = {
    '1': "Moving",
    '2': "Climbing",
    '3': "Resting",
    '4': "Standing up/down",
    '5': "Jumping",
    '11': "Solitary Object Playing",
    '12': "Drinking",
    '13': "Eating",
    #'14': "Manipulating objects",
    '21': "Aggression",
    '22': "Dominance Display",
    '23': "Mounting",
    '24': "Holding Tail Erect",
    '25': "Neutral Approach",
    '26': "Ignoring",
    '27': "Anxiety",
    '28': "Avoiding",
    '29': "Fleeing",
    '31': "Touch",
    #'32': "Play",
    '33': "Follow",
    '34': "Walking past",
    '35': "Presentation for grooming",
    '36': "Grooming",
    '37': "Erection",
    '38': "Shaking",
    '39': "Exploration"
}

# 513 actions that passed the minimum amount of poses with images compared to video length during labeling

class ActionDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 drive_loc: str,
                 model_name: str = "videoprism_public_v1_base_hf",
                 single_only: bool = True,
                 global_info: bool = False,
                 opt_cams_num: int = 6,
                 split: str = "train",
                 fold_idx: int = -1,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.2, # increased to 0.2#todo:
                 seed: int = 42,
                 subsample_time: int = 4,
                 spec_cameras_sel_idx: int = 0,
                 pose_space="3D-AA"):

        assert split in ("train", "test", "val")
        self.model_name   = model_name
        self.drive_loc    = drive_loc
        self.single_only  = single_only
        self.global_info = global_info
        self.opt_cams_num = opt_cams_num


        self.pose_spaces = ["3D-AA", "3D-KP", "3D-Vert", "2D-KP"]
        assert pose_space in self.pose_spaces
        self.pose_space=pose_space


        if spec_cameras_sel_idx == 0:
            camera_array = np.arange(1, 17)
        elif spec_cameras_sel_idx == 1:
            ### Views that see the entire table
            specific_cameras_only = [3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16]
            camera_array = np.array(specific_cameras_only)
        elif spec_cameras_sel_idx == 2:
            ### Top-views only, less occlusion
            specific_cameras_only = [3, 5, 7, 9, 11, 13, 15]
            camera_array = np.array(specific_cameras_only)
        elif spec_cameras_sel_idx == 3:
            specific_cameras_only = [3]
            camera_array = np.array(specific_cameras_only)
        else:
            raise ValueError("This camera set selection index is not implemented!")

        # Inc: 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16
        # Inc_maybe: 1, 12
        # Drop: 2, 10

        # Top views should have less occlusion?

        #2


        self.split        = split
        self.fold_idx = fold_idx


        self.num_cams = len(camera_array)
        self.label_list = list(ACTION_KEY_MAP.values())
        self.num_classes = len(self.label_list)

        # 1) Load and filter CSV
        df = pd.read_csv(csv_path)


        df = df[df["Session"].notna()]

        if single_only:
            df = df[df["n_individuals"] == 1]

        self.number_individuals = df["n_individuals"].max()


        stats = np.load(join(self.drive_loc, "trans_stats.npz"))
        self.mean_trans, self.std_trans = stats["mean_trans"], stats["std_trans"]


        def load_offsets(path):

            npz_path = join(path, f"VideoEncodings/additive_frames.npz")
            offsets = np.load(npz_path, allow_pickle=True)
            add_f = offsets["add_frames"]  # ndarray
            min_f = offsets["min_frames"]  # ndarray

            return add_f, min_f

        # apply over each row’s `path`
        offsets = df["path"].apply(load_offsets)
        # split the tuples into two new columns
        df["add_frames"] = offsets.apply(lambda x: x[0])
        df["min_frames"] = offsets.apply(lambda x: x[1])


        df = self.get_overlap_indices(df)

        # Get the maximum pose sequence
        self.max_pad_length = self.get_max_input_length(df)

        self.subsample_time = subsample_time
        self.max_pad_length = ceil(self.max_pad_length / self.subsample_time)

        # 363 single, multi 441 max pad length
        self.T_idx = {}

        # Load the splits
        with open(join(drive_loc, "data_split.json"), 'r') as f:
            data_split = json.load(f)

        train_idxs = data_split['tr']
        val_idxs = data_split['val']
        test_idxs = data_split['test']

        if split == "train":
            chosen = train_idxs
        elif split == "val":
            chosen = val_idxs
        else:  # split == "test"
            chosen = test_idxs

        ### Load the splits for Cross-validation
        if fold_idx == -1:
            pass
        else:

            assert 0 <= fold_idx < 5, f"fold_idx must be in [0,4], got {fold_idx}"

            split_path = join(drive_loc, f"data_split_{fold_idx}.json")
            with open(split_path, "r") as f:
                data_split = json.load(f)

            tr_cv = data_split["tr"]
            val_cv = data_split["val"]
            test_idxs = data_split["test"]

            if split == "train":
                chosen = tr_cv
            elif split == "val":
                chosen = val_cv
            else:  # split == "test"
                chosen = test_idxs


        df = df.loc[chosen].reset_index(drop=True)
        self._compute_pos_weight_from_df(df, clamp_max=None)

        # 3) Expand each row into camera‐rows
        df = df.loc[df.index.repeat(self.num_cams)].reset_index(drop=True)
        df["cam"] = np.tile(camera_array, len(chosen))
        self.df = df

    def create_hot_label_data(self, df):

        df_hot = np.zeros(shape=(len(df), len(self.label_list)))

        for row_indx, (idx, row) in enumerate(df.iterrows()):

            # --- build multi‐hot labels ---
            loc = ast.literal_eval(row["Locomotion"])
            soc = ast.literal_eval(row["SocialAndOthers"])
            combined = loc + soc

            labels = sorted({self.label_list.index(x)
                         for sub in combined for x in sub})

            labels_np = np.zeros((self.num_classes,), dtype=np.float32)
            labels_np[labels] = 1.0

            df_hot[row_indx, :] = labels_np

        return df_hot


    def _check_presence_multi(self, Y_mat, classes, name):
        present = set(np.array(classes)[(Y_mat.sum(axis=0) > 0).ravel()])
        missing = set(classes) - present
        if missing:
            raise ValueError(
                f"[{name}] split is missing classes: {sorted(missing)}. "
                f"Reduce test/val ratio or ensure each rare class has at least 3–5 samples."
            )


    def get_overlap_indices(self, df):

        start_index_path = "start_index.npy"

        df["seq_inf"] = None

        for idx, row in df.iterrows():

            path = row["path"]
            nb_inds = row["n_individuals"]
            min_f = row["min_frames"]

            # Compute and safe it
            action_params_frame_seq = []
            # Either load from file
            for f in os.listdir(path):
                if f.endswith(".pth") and 'action_params' in f:  # get rid of labeled opt params
                    if f"{str(self.opt_cams_num).zfill(2)}.pth" in f:
                        params = torch.load(join(path, f), map_location="cpu", weights_only=False)
                        action_params_frame_seq.append(params['frame_sequence'].int())
            ## Create the overlap between the two
            overlap_matrix = np.zeros(shape=(nb_inds, min_f))
            # First check if there is overlap
            for anim_index, [s_f, e_f] in enumerate(action_params_frame_seq):
                overlap_matrix[anim_index, s_f:e_f] = 1
            # Collapse the rows
            overlap_sum = overlap_matrix.sum(axis=0)

            # Check for the longest sequence of nb_animals
            start_idx, end_idx = utils.longest_common_interval(overlap_sum, nb_inds)
            seq_array = np.array([start_idx, end_idx])
            np.save(join(path, start_index_path), seq_array)

            df.at[idx, "seq_inf"] = seq_array

        return df


    def get_max_input_length(self, df):

        frame_seq_list = []

        ## Iterate through the dataset
        for idx, row in df.iterrows():

            start_index, end_index = row["seq_inf"]
            frame_seq_list.append(end_index-start_index)

            # action_params = [x for x in os.listdir(path)
            #                  if 'action_params' in x and f"{str(self.opt_cams_num).zfill(2)}.pth" in x]
            #
            # for act_par in action_params:
            #     frame_seq = torch.load(join(path, act_par), weights_only=False)["frame_sequence"]
            #     frame_seq_list.append(frame_seq[1] - frame_seq[0])

        return int(np.stack(frame_seq_list).max())


    ### Pos weight calculation
    def _compute_pos_weight_from_df(self, df_rows, eps: float = 1.0, clamp_max: float = 50.0):
        """
        Compute per-class pos_weight using only the selected (pre-expansion) rows.
        pos_weight[c] = (N_neg[c] + eps) / (N_pos[c] + eps)
        """
        num_classes = len(self.label_list)
        label_to_idx = {name: i for i, name in enumerate(self.label_list)}

        # TODO: some classes are heavily underrepresented,
        # Solitary object playing, climbing, <= 5, aggression, mounting, ignoring?, fleeing, shaking

        # many samples for moving, resting, eating and exploration

        # Build multi-hot for each row, then sum
        pos_counts = np.zeros(num_classes, dtype=np.int64)

        for _, row in df_rows.iterrows():
            loc = ast.literal_eval(row["Locomotion"])
            soc = ast.literal_eval(row["SocialAndOthers"])
            # flatten + unique
            labs = set(x for sub in (loc + soc) for x in sub)
            # increment counts for present labels
            for lab in labs:
                idx = label_to_idx.get(lab, None)
                if idx is not None:
                    pos_counts[idx] += 1

        N = len(df_rows)
        # neg_counts = N - pos_counts
        #
        # pos_weight = (neg_counts + eps) / (pos_counts + eps)  # shape [C]
        # pos_weight = np.clip(pos_weight, a_min=0.0, a_max=clamp_max).astype(np.float32)

        # store a torch tensor on the dataset
        self.pos_weight = self.effective_num_pos_weight(pos_counts)#torch.from_numpy(pos_weight)
        # (optional) keep rates for logging
        self.pos_rate = (pos_counts / max(N, 1)).astype(np.float32)

    def effective_num_pos_weight(self, pos_counts, beta=0.999, cap=20.0):
        pos_counts = np.asarray(pos_counts, dtype=np.float64)
        eff = 1.0 - np.power(beta, pos_counts)
        w = (1.0 - beta) / np.maximum(eff, 1e-8)
        w = np.clip(w, 0.0, cap)
        w /= (w.mean() + 1e-8)
        return torch.tensor(w, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        cam  = int(row["cam"])            # 1..16

        # --- build multi‐hot labels ---
        loc = ast.literal_eval(row["Locomotion"])
        soc = ast.literal_eval(row["SocialAndOthers"])
        combined = loc + soc

        labels = sorted({self.label_list.index(x)
                          for sub in combined for x in sub })

        labels_np = np.zeros((self.num_classes,), dtype=np.float32)
        labels_np[labels] = 1.0

        #
        # one‐cam pose: just duplicate or leave as is
        #pose_feat = body_poses   # shape [T, pose_dim]

        # --- load only this cam’s visual features ---
        subsample_path = join(path, f"VideoEncodings/act_{self.model_name}_{cam}_{self.subsample_time}.npz")
        npy_path = join(path, f"VideoEncodings/act_{self.model_name}_{cam}.npy")

        #T_path = join(path, f"VideoEncodings/act_{self.model_name}_{cam}_{self.subsample_time}_T.npy")


        reload = False
        if not os.path.exists(subsample_path) or reload:
            feats = np.load(npy_path)
            #key   = utils.get_video_names(path) + f"{cam}.mp4"  ## video_key
            vid   = feats#[key]  # get features from encoder

        add_f = row["add_frames"]  # a dict
        min_f = row["min_frames"]  # a scalar

        # The common frame sequence
        frame_seq = row["seq_inf"]

        offset = add_f[str(cam)] - min_f  # get additive frame
        sf, ef = (frame_seq + offset).astype(int)  # adapt sequence to video features


        ### For videoprism [T, H, W, C]
        ### for resnet-50 [T, C]
        ### for [T, C]

        if not os.path.exists(subsample_path) or reload:
            vis_feat = vid[sf:ef]
            T = vis_feat.shape[0]
            #np.save(T_path, T)
            # Subsample the vis_features
            vis_feat = vis_feat[::self.subsample_time]# Subsample the vis_features
        else:
            data_loaded = np.load(subsample_path, allow_pickle=False)
            vis_feat = data_loaded["vis_feat"]
            T = data_loaded["T"]


        # --- load pose (same for all cams) ---
        action_params = [f for f in os.listdir(path)
                         if "action_params" in f and
                         f"{str(self.opt_cams_num).zfill(2)}_rotmat.npz" in f]



        ### Initialize poses

        video_encodings_path = join(path, "VideoEncodings")
        if self.pose_space == "3D-AA":
            if self.global_info:
                # The number is extended by 9+ 3
                body_poses = np.zeros(shape=(T, self.number_individuals, 9 + 3 + 27 * 9))  ## 255
            else:
                body_poses = np.zeros(shape=(T, self.number_individuals, 27 * 9))  ## 243
        elif self.pose_space == "3D-KP":
            ### 3d keypoints is 20 x 3
            body_poses = np.zeros(shape=(T, self.number_individuals, 20 * 3))

            pose_spec_paths = [x for x in os.listdir(video_encodings_path) if 'keypoints' in x]
        elif self.pose_space == "3D-Vert":
            body_poses = np.zeros(shape=(T, self.number_individuals, 3625 * 3))
            pose_spec_paths = [x for x in os.listdir(video_encodings_path) if 'vertices' in x]
        elif self.pose_space == "2D-KP":
            body_poses = np.zeros(shape=(T, self.number_individuals, 20 * 2))
            pose_spec_paths = [x for x in os.listdir(video_encodings_path) if 'keypoints' in x]

        for ind_index, ind in enumerate(row["individuals"]):

            params = np.load(join(path, action_params[ind_index]))

            pose_loaded = params["pose_matrix"]
            frame_seq_i = params["frame_seq_i"]


            # As common is always >= individual this will be positive >= 0
            s_i = int(frame_seq[0] - frame_seq_i[0])
            assert s_i >= 0

            if self.pose_space == "3D-AA":



                pose_rotmats = pose_loaded[s_i:s_i + T, 0, :]

                if self.global_info:

                    # Reduce dimensions of rot_global
                    rot_glob = params["global_rot"]
                    abs_trans = params["transl"]


                    # Process the info
                    rot_glob = rot_glob[s_i:s_i + T, 0, :]
                    abs_trans = abs_trans[s_i:s_i + T, 0, :]

                    # 1) normalize absolute positions
                    abs_trans = (abs_trans - self.mean_trans[None]) / (self.std_trans[None] + 1e-6)  # still shape (T,3)

                    # 2) replace any NaNs (or ±inf) with 0, #todo: check maybe if Nans remain
                    #    this will cover both NaNs and numeric overflows
                    #abs_trans = np.nan_to_num(abs_trans, nan=0.0, posinf=0.0, neginf=0.0)

                    body_poses[:, ind_index, :] = np.concatenate([abs_trans, rot_glob, pose_rotmats], axis=1)

                else:
                    body_poses[:, ind_index, :] = pose_rotmats

            else:
                pose_spec_path = [x for x in pose_spec_paths if ind in x]

                assert len(pose_spec_path) == 1
                pose_spec_params = np.load(join(video_encodings_path, pose_spec_path[0]))

            if self.pose_space == "3D-KP":

                ## Load the parameters
                pose_loaded = pose_spec_params["kps_per_ind"] # verts_per_ind, projected_kps_per_ind

                ### Reshape (T, N, 3) -> (T, N*3)
                pose_loaded = pose_loaded.reshape(pose_loaded.shape[0], 60)
                #pose_loaded = np.nan_to_num(pose_loaded, nan=0.0, posinf=0.0, neginf=0.0)

                body_poses[:, ind_index, :] = pose_loaded[s_i:s_i + T, :]

            elif self.pose_space == "3D-Vert":
                ## Load the parameters
                pose_loaded = pose_spec_params["verts_per_ind"]  #, projected_kps_per_ind

                ### Reshape (T, N, 3) -> (T, N*3)
                pose_loaded = pose_loaded.reshape(pose_loaded.shape[0], 3*3625)
                #pose_loaded = np.nan_to_num(pose_loaded, nan=0.0, posinf=0.0, neginf=0.0)

                body_poses[:, ind_index, :] = pose_loaded[s_i:s_i + T, :]
            elif self.pose_space == "2D-KP":

                ## Load the parameters
                pose_loaded = pose_spec_params["projected_kps_per_ind"]  # verts_per_ind, projected_kps_per_ind

                # Aranged from 1-16
                pose_loaded = pose_loaded[cam-1, :, :, :]
                ### Reshape (T, N, 3) -> (T, N*3)
                pose_loaded = pose_loaded.reshape(pose_loaded.shape[0], 40)
                #pose_loaded = np.nan_to_num(pose_loaded, nan=0.0, posinf=0.0, neginf=0.0)

                body_poses[:, ind_index, :] = pose_loaded[s_i:s_i + T, :]


        if not os.path.exists(subsample_path) or reload:
            np.savez(subsample_path, vis_feat=vis_feat, T=T)

        body_poses = body_poses[::self.subsample_time]
        T = ceil(T / self.subsample_time)

        # --- to tensors ---
        return (
            torch.from_numpy(vis_feat).float(),      # [T, C, H, W]
            torch.from_numpy(body_poses).float(),     # [T, N_animals, N_J*3 or N_J*9]
            torch.from_numpy(labels_np).float(),     # [num_labels]
            cam,                                     # int camera index
            T
        )
