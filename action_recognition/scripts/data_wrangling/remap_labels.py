import numpy as np
import os
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm
import src.utils as utils


csv_path    = "/media/lucas/V-2/LabelInformation/EntireDataset/tracked_actions.csv"

# Load the dataset
df = pd.read_csv(csv_path)

model_name = "videoprism_public_v1_base_hf"

num_cams = 16

# Iterate rows
for idx, row in tqdm(df.iterrows()):
    # Check if there exists action params
    path = row["path"]
    individuals = row["individuals"]


    # Get the folder of the video embeddings
    # Get the camera names and the additive dict

    video_encoder_path = join(path, rf"VideoEncodings/act_{model_name}.npz")
    video_name_base = utils.get_video_names(path)

    video_feat_dict = np.load(video_encoder_path)


    add_frames, min_frames = utils.get_additive_frames_from_file(path)

    # Write each parameter as a single file
    # Iterates the cameras from 1 - 16, not as it would be listed by the file system!
    for i in range(1, num_cams + 1, 1):
        video_name = video_name_base + f"{i}.mp4"
        video_feat = video_feat_dict[video_name]

        np.save(join(path, rf"VideoEncodings/act_{model_name}_{i}.npy"), video_feat)


    np.savez(join(path, rf"VideoEncodings/additive_frames.npz"), add_frames=add_frames, min_frames=min_frames)

    os.remove(video_encoder_path)


# TODO: move all the video files to the B-drive otherwise too slow.