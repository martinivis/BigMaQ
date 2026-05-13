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
from action_recognition.src.utils import longest_common_interval



source_drive = r"/media/lucas/V-2"
source_drive = r"/media/lucas/W-2/BigMaQ"

dest_drive = r"/media/lucas/FastInternal/BigMacaque/ActionDataset"
csv_path    = "/media/lucas/V-2/LabelInformation/EntireDataset/tracked_actions.csv"

csv_path = r"/media/lucas/W-2/BigMaQ/dataset_overview.csv"

def create_folder(path):
    os.makedirs(path, exist_ok=True)

def extract_session(path):
    m = re.search(r"(Session\d+(?:_\d+)?)", path)
    return m.group(1) if m else None

def safe_copy(src: str, dst: str):
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

### Paths not to include...
paths_to_skip = ["/media/lucas/V-2/Session9/Actions/Interaction/CollectiveInterest_G_T_2",
                 "/media/lucas/V-2/Session9/Actions/Interaction/Scratch_T_3"]

model_activations = ["resnet50", "movinet-a2", "dinov2-base-cls", "vit-base-cls",
                     "dinov2-base-patch", "vit-base-patch","timesformer-base-finetuned-k400", "videoprism_public_v1_base_hf"]

copy_model = 3

model_name = model_activations[copy_model]#"resnet50"

if copy_model >= 4:
    del_prev_act = True
else:
    del_prev_act = False

specific_cameras_only = None
#specific_cameras_only = [3, 5, 7, 9, 11, 13, 15]
specific_cameras_only = [3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16]

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

    none_detected=False

    for f in os.listdir(path):
        if f.endswith(".pth") and 'action_params' in f:  # get rid of labeled opt params

            ## Check if the individuals full name is in the file


            if f"{str(opt_cams).zfill(2)}.pth" in f:
                params = torch.load(join(path, f), map_location="cpu", weights_only=False)
                action_params_frame_seq.append(params['frame_sequence'].int())

                pose_test = params["body_pose"]

                if torch.isnan(pose_test).any():
                    none_detected = True



    if none_detected:
        #print(f"{path} has NAN in the pose")
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


    video_encodings_folder = join(new_action_path, "VideoEncodings")



    # For bigger models
    if del_prev_act:
        activation_files = [x for x in os.listdir(video_encodings_folder) if '.npy' in x]
        # Delete these files
        for act_file_old in activation_files:
            os.remove(join(video_encodings_folder, act_file_old))

        # Delete also files that are subgroups?
        activation_files = [x for x in os.listdir(video_encodings_folder) if '.npz' in x]
        for act_file_old in activation_files:
            for model in model_activations:
                if model in act_file_old:
                    os.remove(join(video_encodings_folder, act_file_old))


    # Copy the new ones based on the model_name
    video_encodings_general_folder = join(path, "VideoEncodings")
    activation_files = [x for x in os.listdir(video_encodings_general_folder) if model_name in x]

    for act_file_new in activation_files:

        if specific_cameras_only is not None:
            for spec_cam in specific_cameras_only:
                # Only copy if the camera is in the selected cameras
                if str(spec_cam) in act_file_new:
                    shutil.copy(join(video_encodings_general_folder, act_file_new), join(video_encodings_folder, act_file_new))
        else:
            shutil.copy(join(video_encodings_general_folder, act_file_new), join(video_encodings_folder, act_file_new))



    ### Copy the npz files per individual
    keypoint_files = [x for x in os.listdir(video_encodings_general_folder) if 'keypoints' in x or 'vertices' in x]

    for keypoint_file in keypoint_files:
        shutil.copy(join(video_encodings_general_folder, keypoint_file), join(video_encodings_folder, keypoint_file))

    ### Copy the additive frames
    npz_path = join(path, f"VideoEncodings/additive_frames.npz")
    new_npz_path = join(video_encodings_folder, "additive_frames.npz")
    shutil.copy(npz_path, new_npz_path)


print("Accepted number of actions: ", passed)


# Folder structure
# Overview pandas
# Actions folder
# - same as before
# Calibration folder
# - put the calibration files and add the session number.