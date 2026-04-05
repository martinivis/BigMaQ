import numpy as np
import pandas as pd

ACTION_KEY_MAP = {
    '1': "Moving",
    '2': "Climbing",
    '3': "Resting",
    '4': "Standing up/down",
    '5': "Jumping",
    '11': "Solitary Object Playing",
    '12': "Drinking",
    '13': "Eating",
    '14': "Manipulating objects",
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
    '32': "Play",
    '33': "Follow",
    '34': "Walking past",
    '35': "Presentation for grooming",
    '36': "Grooming",
    '37': "Erection",
    '38': "Shaking",
    '39': "Exploration"
}

labels_as_string = list(ACTION_KEY_MAP.values())

labels_as_string.pop(8)
labels_as_string.pop(18)
### Labels as string are the corresponding labels, until it is the same category
categories_labels = [0, 5, 8, 17, 25]
category_names = ["Locomotion", "Object Interaction", "Social Interaction", "Others"]

def switch(labels, swap1, swap2):

    # Swap labels
    labels[swap1], labels[swap2] = labels[swap2], labels[swap1]
    # Swap array entries

    return labels

# Switch Rest with Jumping
i = 2
j = 4
labels_as_string = switch(labels_as_string, i, j)

# Switch Eating with solit
i = 6
j = 7
labels_as_string = switch(labels_as_string, i, j)

# Switch Eating with solit
i = -1
j = -2
labels_as_string = switch(labels_as_string, i, j)



# ------------------

# For each
import itertools
from os.path import join
import os
drive_loc   = "/media/lucas/FastInternal/BigMacaque/ActionDataset"

training_path = join(drive_loc, "model_weights")

flags = {
    "single_animal": [False],
    "global_inf":    [False],  # Reduced because not necessary
    "use_pose":      [False, True],
    "add_feat":      [False], # Additive features doesn't work for the 2 MLP encodings, hence concat the representations
    "use_vis":       [False, True],
    "ws":           [0.5]
}

all_runs = [
    (SA, GI, UP, AF, UV, WO)
    for SA, GI, UP, AF, UV, WO in itertools.product(
        flags["single_animal"],
        flags["global_inf"],
        flags["use_pose"],
        flags["add_feat"],
        flags["use_vis"],
        flags["ws"]
    )

    if (UP or UV)
    #if (UP or (not GI and not AF)) and (UV or UP)  # Only include cases where either use_pose, or if not use_pose drop also all cases except for no general info or add features
]
camera_set_idx = 1#2#2#1

## movi-net activations sort of weird
#model_activations = ["resnet50", "movinet-a2", "timesformer-base-finetuned-k400", "videoprism_public_v1_base_hf"]
#model_activations = ["resnet50", "movinet-a2", "dinov2-base-cls","dinov2-base-patch" ,"timesformer-base-finetuned-k400", "videoprism_public_v1_base_hf"]
model_activations = ["resnet50", "movinet-a2", "dinov2-base-cls", "vit-base-cls",
                     "dinov2-base-patch", "vit-base-patch","timesformer-base-finetuned-k400", "videoprism_public_v1_base_hf"]

multiple_vision_runs = True

model_idx = 0

run_idx = 2

use_stopper = True
sub_sample = 4
model_name = model_activations[model_idx]

# to return to faster training, unmount and mount and restart pycharm
pose_spaces = ["3D-AA", "3D-KP", "3D-Vert", "2D-KP"]
pose_idx = 0
pose_space = pose_spaces[pose_idx]

all_runs = [all_runs[run_idx]]

for SA, GI, UP, AF, UV, WO in all_runs:



    if run_idx == 1:
        if pose_idx <3:
            camera_set_idx = 1#2 #todo: because of wrong run names before, it is anyway on the 3 cam set
            folds = [0, 1, 2, 3, 4]
        else:
            # todo: also do multiple 2d views runs!
            if multiple_vision_runs:
                folds = [-1, 43, 44]
            else:
                folds = [-1]
    else:
        if multiple_vision_runs:
            folds = [-1, 43, 44]
        else:
            folds = [-1]

    val_ratio = 0.1#0.15 #0.1#0.1  # 0.05#0.05
    cls_reweight = False
    test_ratio = 0.2#0.15



    maps_over_folds = []
    maps_all_over_folds = []
    micro_map_over_folds = []

    for fold in folds:
        run_name = (f"{model_name}_camS_{camera_set_idx}_SA{SA}_GI{GI}_UP{UP}_AF{AF}_UV{UV}_{sub_sample}_{WO}_"
                    f"{pose_space}_valratio{val_ratio}_clsw_{cls_reweight}_stop_{use_stopper}_fold_{fold}")

        print("mAP per classes for: ", run_name)

        run_dir = os.path.join(training_path, run_name)

        metrics = pd.read_csv(join(run_dir, "metrics.csv"))

        all_map_to_check = []
        maps = []

        for category_idx, category_name in enumerate(category_names):

            cat_cur_labels = labels_as_string[categories_labels[category_idx]:categories_labels[category_idx+1]]
            #print(len(cat_cur_labels))
            #print(f"Category {category_name} with labels: {cat_cur_labels}")

            map_per_category_list = []
            for cat_cur_l in cat_cur_labels:
                map_per_category_list.append(metrics.loc[0, f"AP::"+cat_cur_l])
            all_map_to_check.extend(map_per_category_list)
            map_array = np.array((map_per_category_list))

            maps.append(map_array.mean())
            print(f"Category {category_name} with mAP: {map_array.mean():.3f}")

        maps_over_folds.append(maps)
        print(f"& {maps[0]:.3f} & {maps[1]:.3f} & {maps[2]:.3f} & {maps[3]:.3f}")

        all_map_to_check = np.array(all_map_to_check).mean()
        maps_all_over_folds.append(all_map_to_check)
        print(f"{all_map_to_check:.3f} and full prec:", all_map_to_check)

        micro_map_over_folds.append(metrics.loc[0, "mAP_micro"])

    # MAP per category over folds
    maps_over_folds = np.array(maps_over_folds)
    maps_over_folds_std = np.std(maps_over_folds, axis=0)
    maps_over_folds = np.mean(maps_over_folds, axis=0)



    # Overall map over folds
    maps_all_over_folds_std = np.array(maps_all_over_folds).std()
    maps_all_over_folds = np.array(maps_all_over_folds).mean()


    # Overall micro map
    micro_map_over_folds_std = np.array(micro_map_over_folds).std()
    micro_map_over_folds = np.array(micro_map_over_folds).mean()

    maps_over_folds *= 100
    maps_all_over_folds *= 100
    micro_map_over_folds *= 100
    maps_over_folds_std *= 100
    maps_all_over_folds_std *= 100
    micro_map_over_folds_std *= 100


    print(f"Over folds:")
    print(f"{maps_all_over_folds:.1f} & {maps_over_folds[0]:.1f} & {maps_over_folds[1]:.1f} & {maps_over_folds[2]:.1f} & {maps_over_folds[3]:.1f}")
    print(f"{maps_all_over_folds:.1f} and full prec:", maps_all_over_folds)
    print(f"Micro_map {micro_map_over_folds:.1f} and full prec:", micro_map_over_folds)
    print("================= STD ===================")
    print(f"Over folds:")
    print(
        f"{maps_all_over_folds_std:.1f} & {maps_over_folds_std[0]:.1f} & {maps_over_folds_std[1]:.1f} & {maps_over_folds_std[2]:.1f} & {maps_over_folds_std[3]:.1f}")
    print(f"STD overall: {maps_all_over_folds_std:.1f} and full prec:", maps_all_over_folds_std)
    print(f"Micro_map STD: {micro_map_over_folds_std:.1f} and full prec:", micro_map_over_folds_std)

    print("=============== Latex =====================")
    print(f"${maps_all_over_folds:.1f} \pm {maps_all_over_folds_std:.1f}$")
    print(
        f"${maps_all_over_folds:.1f} \pm {maps_all_over_folds_std:.1f} $& ${maps_over_folds[0]:.1f}\pm {maps_over_folds_std[0]:.1f} $&$ {maps_over_folds[1]:.1f}\pm {maps_over_folds_std[1]:.1f}$&$ {maps_over_folds[2]:.1f}\pm {maps_over_folds_std[2]:.1f}$&$ {maps_over_folds[3]:.1f}\pm {maps_over_folds_std[3]:.1f}$")


