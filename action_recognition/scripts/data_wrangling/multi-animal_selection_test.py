import os
from os.path import join
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from src.utils import longest_common_interval

csv_path    = "/media/lucas/V-2/LabelInformation/EntireDataset/tracked_actions.csv"

# Load the dataset
df = pd.read_csv(csv_path)


passed = 0

opt_cams = 6

min_seq_length_percent = 0.95

for idx, row in tqdm(df.iterrows(), total=len(df)):

    # Check if there exists action params
    path = row["path"]
    nb_inds = row["n_individuals"]

    ## Load the min_frames
    npz_path = join(path, f"VideoEncodings/additive_frames.npz")
    offsets = np.load(npz_path, allow_pickle=True)
    add_f = offsets["add_frames"]  # ndarray
    min_f = offsets["min_frames"]  # ndarray

    action_params_frame_seq = []

    # copy any .pth files
    for f in os.listdir(path):
        if f.endswith(".pth") and 'action_params' in f:  # get rid of labeled opt params

            if f"{str(opt_cams).zfill(2)}.pth" in f:
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
    start_idx, end_idx = longest_common_interval(overlap_sum, nb_inds)

    #TODO: maybe get the start and end idx directly in a file.

    ## Process

    if start_idx is None:
        continue

    # Thinking
    # subset nehmen da sonst sequence length im transformer zu lange... außerdem aus praktischen gründen is die disk zu klein
    # man könnte natürlich einfach padden, aber dann ist der vergleich mit den posen nicht fair und bisschen random.

    if end_idx-start_idx < min_seq_length_percent*min_f:  # 0.01 => , 0.1 => 642, 0.2 => 631,0.5 => 595, 0.8 => 549, 0.9 => 531, 0.95 => 513, 0.98 => 501, 1=> 498
        # single actions alone have 391 without filtering!
        continue
    else:
        passed += 1
        # TODO: for the copying part copy
        # TODO: for the training part, allow for training (if the dataset is not already filtered like now)
print("Actions passing criterion: ", passed)
