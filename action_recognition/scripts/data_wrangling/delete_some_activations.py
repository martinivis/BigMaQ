import os



path_actions = r"/media/lucas/FastInternal/BigMacaque/ActionDataset/Actions"

actions = [x for x in os.listdir(path_actions)]

model_activations = ["resnet50", "movinet-a2", "dinov2-base-cls", "vit-base-cls",
                     "dinov2-base-patch", "vit-base-patch","timesformer-base-finetuned-k400", "videoprism_public_v1_base_hf"]
model_idx = 4
model_name = model_activations[model_idx]

from os.path import join

for action in actions:

    path_action = join(path_actions, action)

    if os.path.exists(path_action):
        path_enc = join(path_action, "VideoEncodings")



        for i in range(1, 17, 1):
            path_act_file = join(path_enc, f"act_{model_name}_{i}_4.npy")

            if os.path.exists(path_act_file):
                os.remove(path_act_file)

            path_act_file = join(path_enc, f"act_{model_name}_{i}_4.npz")

            if os.path.exists(path_act_file):
                os.remove(path_act_file)

            path_act_file = join(path_enc, f"act_{model_name}_{i}_4_T.npy")

            if os.path.exists(path_act_file):
                os.remove(path_act_file)
