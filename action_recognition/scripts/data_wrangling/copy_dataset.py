"""
Copies all the relevant files from the dataset to a new location
"""

import os
from os.path import join
import pandas as pd
from tqdm import tqdm
import shutil
import re
import torch
import numpy as np
from src.utils import longest_common_interval

source_drive = r"/media/lucas/V-2"
dest_drive = r"/media/lucas/FastInternal/BigMacaque/ActionDataset"
csv_path    = "/media/lucas/V-2/LabelInformation/EntireDataset/tracked_actions.csv"

#todo: uncomment for real data exports to a bigger drive

def create_folder(path):
    os.makedirs(path, exist_ok=True)

def extract_session(path: str) -> str | None:
    m = re.search(r"(Session\d+(?:_\d+)?)", path)
    return m.group(1) if m else None

def safe_copy(src: str, dst: str) -> None:
    """Copy if exists, otherwise warn."""
    if os.path.isfile(src):
        shutil.copy(src, dst)
    else:
        print(f"⚠️ Missing file: {src}")

# Create the actions folder
actions_folder_dest = join(dest_drive, "Actions")
calibration_folder_dest = join(dest_drive, "Calibration")
create_folder(actions_folder_dest)
#create_folder(calibration_folder_dest)

## Copy the calibration data
opt_cams = 6
min_seq_length_percent = 0.95
passed = 0
# --- COPY CALIBRATION FILES ---
# for name in [f"S{i}" for i in range(1, 11)] + ["S3_2"]:
#     src = join(source_drive, "Calibration", name, "calibration.toml")
#     dst = join(calibration_folder_dest, f"calibration{name[1:]}.toml")
#     safe_copy(src, dst)

# Load the dataset
df = pd.read_csv(csv_path)
df_new = df.copy()
df_new["Session"] = None

# Iterate rows
for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Check if there exists action params
    path = row["path"]




    nb_inds = row["n_individuals"]

    # Only for single-animal first, multi not sure if it fits the disk
    #if row["n_individuals"] > 1:
    #    continue

    # Check for length of the union ---------------

    ## Load the min_frames
    npz_path = join(path, f"VideoEncodings/additive_frames.npz")
    offsets = np.load(npz_path, allow_pickle=True)
    add_f = offsets["add_frames"]  # ndarray
    min_f = offsets["min_frames"]  # ndarray

    action_params_frame_seq = []

    none_detected = False

    for f in os.listdir(path):
        if f.endswith(".pth") and 'action_params' in f:  # get rid of labeled opt params

            if f"{str(opt_cams).zfill(2)}.pth" in f:
                params = torch.load(join(path, f), map_location="cpu", weights_only=False)
                action_params_frame_seq.append(params['frame_sequence'].int())

                pose_test = params["body_pose"]

                if torch.isnan(pose_test).any():
                    none_detected = True

    if none_detected:
        print(f"{path} has NAN in the pose")
        continue


    ## Create the overlap between the two
    overlap_matrix = np.zeros(shape=(nb_inds, min_f))

    # First check if there is overlap
    for anim_index, [s_f, e_f] in enumerate(action_params_frame_seq):
        overlap_matrix[anim_index, s_f:e_f] = 1

    # Collapse the rows
    overlap_sum = overlap_matrix.sum(axis=0)

    # Check for the longest sequence of nb_animals
    start_idx, end_idx = longest_common_interval(overlap_sum, nb_inds)

    if start_idx is None:
        continue

    if end_idx-start_idx < min_seq_length_percent*min_f:  # 0.01 => , 0.1 => 642, 0.2 => 631,0.5 => 595, 0.8 => 549, 0.9 => 531, 0.95 => 513, 0.98 => 501, 1=> 498
        # single actions alone have 391 without filtering!
        continue
    else:
        passed += 1

    # -------------

    session = extract_session(path)
    df_new.at[idx, "Session"] = session


    # New path for that action
    new_action_path = join(actions_folder_dest, f"{idx}")
    create_folder(new_action_path)
    df_new.at[idx, "path"] = new_action_path

    ## Copy the contents
    sub_dirs_low_mem = ["SAM", "YOLO_results", "VideoEncodings"] #todo: change again
    sub_dirs_low_mem = ["VideoEncodings"]

    os.makedirs(join(new_action_path, sub_dirs_low_mem[0]))
    # # copy sub‑dirs
    # for sub in sub_dirs_low_mem:
    #     src_dir = join(path, sub)
    #     dst_dir = join(new_action_path, sub)
    #     try:
    #         shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    #     except FileNotFoundError:
    #         print(f"⚠️ Missing directory: {src_dir}")


    copy_renderings = False
    if copy_renderings:
        ## OptimizationRenderings
        prefix = r"Optimization_Renderings/Videos_last/Cams(6)_MaxIm(100)_Square(True)_KPconf(True)_BBconf(True)_HPFalse_Note(EntireSetOpt100.000_HandExtension_EntireRun_higher_temporal_smooth_tail_angle_weights)"

        rend_dir = join(path, prefix)
        rend_out_dir = join(new_action_path, "Optimization_Renderings")
        create_folder(rend_out_dir)
        if os.path.isdir(rend_dir):
            for vid in os.listdir(rend_dir):
                if vid.endswith(".mp4"):
                    shutil.copy(join(rend_dir, vid), join(rend_out_dir, vid))
        else:
            print(f"⚠️ Missing render dir: {rend_dir}")


    files_in_root = [x for x in os.listdir(path)]

    # copy any .pth files
    for f in os.listdir(path):
        if f.endswith(".pth") and f"{str(opt_cams).zfill(2)}.pth" in f:  # get rid of labeled opt params
            shutil.copy(join(path, f),
                        join(new_action_path, f))

    # copy any CoreView mp4s
    # for f in os.listdir(path):
    #     if f.endswith(".mp4") and "CoreView" in f:
    #         shutil.copy(join(path, f),
    #                     join(new_action_path, f))

    ## Copy the tracking data
    # copy the tracking and pose files
    # for f in ("tracking_data_100_True.pkl", "pose_map.json"):
    #     src = join(path, f)
    #     dst = join(new_action_path, f)
    #     safe_copy(src, dst)

print("Accepted number of actions: ", passed)
df_new.to_csv(join(dest_drive, "tracked_actions.csv"))







# Folder structure
# Overview pandas
# Actions folder
# - same as before
# Calibration folder
# - put the calibration files and add the session number.