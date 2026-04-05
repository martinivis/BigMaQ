import sys
import os
from os.path import join
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import numpy as np
import itertools
import pandas as pd
from sklearn.metrics import average_precision_score, multilabel_confusion_matrix

import gc, torch
# project imports
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(project_root)

from action_recognition.src.data_loader_torch import ActionDataset
from action_recognition.src.utils_torch import collate_fn, EarlyStopping
from action_recognition.model.base_model import ActionTransformerMAP   # or ActionTransformer
import random

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def reseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

### Results:
# - Seq complete accuracy is def higher with pose only vs vis only... for earlier epochs.
#


# 1) Hyper‐params & device
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
csv_path    = "/media/lucas/FastInternal/BigMacaque/ActionDataset/tracked_actions.csv"
#csv_path = "/media/lucas/V-2/LabelInformation/EntireDataset/tracked_actions.csv"

drive_loc   = "/media/lucas/FastInternal/BigMacaque/ActionDataset"

training_path = join(drive_loc, "model_weights")
os.makedirs(training_path, exist_ok=True)

### Combinations of single animal, global inf, use pose

#single_animal = False
#global_inf = True
sub_sample = 4#16#4#4# 16#16# 4
#use_pose = False
w1 = 0

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


# depending on subsampling different batch size
sub_sample_map = {4: 4, 1: 4}


batch_size  = 1#4#sub_sample_map[sub_sample]
lr          = 1e-4
wd          = 1e-2
num_epochs  = 20#10#10# 30#15#30#1#30#10#20 # trained 30 epochs for the other cases



model_dim_D = 256
n_heads = 8
n_layers = 2 # before this was 8

### Models that overfit train seq with pose info only
model_dim_D = 256
n_heads = 8
n_layers = 2 # before this was 8


#testing
#model_dim_D = 32
#n_heads = 4
#n_layers = 2

# 0=all, 1=table views, 2=top-views, 3=single view (used for training the pose pipeline only, because it sees the dataset n_cams more often)
camera_set_idx = 1#3#1#2#2#1

## movi-net activations sort of weird
#model_activations = ["resnet50", "movinet-a2", "timesformer-base-finetuned-k400", "videoprism_public_v1_base_hf"]
#model_activations = ["resnet50", "movinet-a2", "dinov2-base-cls","dinov2-base-patch" ,"timesformer-base-finetuned-k400", "videoprism_public_v1_base_hf"]
model_activations = ["resnet50", "movinet-a2", "dinov2-base-cls", "vit-base-cls",
                     "dinov2-base-patch", "vit-base-patch","timesformer-base-finetuned-k400", "videoprism_public_v1_base_hf"]

pose_spaces = ["3D-AA", "3D-KP", "3D-Vert", "2D-KP"]

model_idx = 3
pose_idx = 3

run_idx = 1

use_stopper = True

model_name = model_activations[model_idx]


@torch.inference_mode()
def measure_flops_cpu(model, *inputs_cpu):
    import copy
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import addmm_flop_jit, matmul_flop_jit, bmm_flop_jit
    from thop import profile

    # Copy model to CPU and eval mode
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    flops = None

    try:
        # Custom handle for scaled dot-product attention (if you use it)
        def sdpa_flop_jit(inputs_, outputs_):
            q, k, v = inputs_[:3]
            B, H, N, Dh = q.shape
            return int(4 * B * H * N * N * Dh)  # qk^T + attn * v

        # ---- fvcore path ----
        fca = FlopCountAnalysis(model_cpu, inputs_cpu)
        fca.set_op_handle("aten::addmm", addmm_flop_jit)
        fca.set_op_handle("aten::matmul", matmul_flop_jit)
        fca.set_op_handle("aten::bmm",    bmm_flop_jit)
        fca.set_op_handle("aten::scaled_dot_product_attention", sdpa_flop_jit)
        flops = int(fca.total())

    except Exception:
        # ---- thop fallback ----
        try:
            macs, _ = profile(model_cpu, inputs=inputs_cpu, verbose=False)
            flops = int(macs * 2)
        except Exception:
            flops = None

    finally:
        del model_cpu
        gc.collect()

    return flops

# Make it iterate, single yes no, subsample by 4 maybe, and global_info yes/no and write to a table

# Change the order of the runs, most potent at the start

# Run for 20 epochs for now and look at results. early stopping uses very early

# Bottleneck of training is likely just the disk retrieval!, for less features way faster

# Skip the pose run,
skip_pose_run=True

# to return to faster training, unmount and mount and restart pycharm

#all_runs = [all_runs[run_idx]] #todo: comment if multiple
#for run in all_runs[run_idx]:
seed_list = [42, 43, 44]
seed_list = [42]
#seed_list = [43, 44]

#todo: currently it runs vision only on multiple pose selections? just run with run_idx 2?
pose_list = [0, 1, 2, 3]
pose_list = [0]

for pose_idx in pose_list:
    for seed in seed_list:
        for SA, GI, UP, AF, UV, WO in all_runs:

            #seed = seed_list[0]

            # Call the seed to be the same again
            reseed(seed=seed)

            val_ratio = 0.1#0.15 #0.1#0.1  # 0.05#0.05
            cls_reweight = False
            test_ratio = 0.2#0.15

            pose_space = pose_spaces[pose_idx]


            if seed == 42:
                fold = -1
            else:
                fold = seed
            run_name = (f"{model_name}_camS_{camera_set_idx}_SA{SA}_GI{GI}_UP{UP}_AF{AF}_UV{UV}_{sub_sample}_{WO}_"
                        f"{pose_space}_valratio{val_ratio}_clsw_{cls_reweight}_stop_{use_stopper}_fold_{fold}_flops")

            save_flops_path = join(r"/media/lucas/FastInternal/BigMacaque/ActionDataset/flops_measured",
                                   f"{model_name}_UP{UP}_{pose_space}_AF{AF}_UV{UV}.txt")


            w1 = WO

            if UP and not UV:

                if pose_space != "2D-KP":
                    spec_cameras_sel_idx = 3
                    spec_camera_test_idx = camera_set_idx
                else:
                    spec_cameras_sel_idx = camera_set_idx
                    spec_camera_test_idx = camera_set_idx
                if skip_pose_run:
                    continue
            else:
                spec_cameras_sel_idx = camera_set_idx
                spec_camera_test_idx = camera_set_idx

            run_dir = os.path.join(training_path, run_name)
            os.makedirs(run_dir, exist_ok=True)

            print("==== RUN:", run_name)

            print(f"Selection of cameras: ", spec_cameras_sel_idx)

            # 2) Datasets & Loaders
            train_ds = ActionDataset(csv_path, drive_loc,
                                     single_only=SA, global_info=GI,
                                     split="train", val_ratio=val_ratio, test_ratio=test_ratio,
                                     subsample_time=sub_sample, model_name=model_name, spec_cameras_sel_idx=spec_cameras_sel_idx,
                                     pose_space=pose_space)

            val_ds = ActionDataset(csv_path, drive_loc,
                                     single_only=SA, global_info=GI,
                                     split="val", val_ratio=val_ratio, test_ratio=test_ratio,
                                     subsample_time=sub_sample, model_name=model_name, spec_cameras_sel_idx=spec_cameras_sel_idx,
                                   pose_space=pose_space)


            test_ds   = ActionDataset(csv_path, drive_loc,
                                      single_only=SA, global_info=GI,
                                      split="test", val_ratio=val_ratio, test_ratio=test_ratio,
                                      subsample_time=sub_sample, model_name=model_name, spec_cameras_sel_idx=spec_camera_test_idx,
                                      pose_space=pose_space)

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True,
                collate_fn=lambda b: collate_fn(b, max_pad_length=train_ds.max_pad_length)
            )

            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True,
                collate_fn=lambda b: collate_fn(b, max_pad_length=val_ds.max_pad_length)
            )

            test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True, persistent_workers=True,
                collate_fn=lambda b: collate_fn(b, max_pad_length=test_ds.max_pad_length)
            )




            if val_ratio > 0 and use_stopper:
                stopper = EarlyStopping(min_delta_rel=1e-3, restore_best=True, run_dir=run_dir, device=device,
                                        patience=5)


            # 3) Model, loss, optimizer
            # compute dims from one batch (or hardcode)
            # Here: take one sample to infer H,W,C and N_animals,D
            sample = next(iter(train_loader))
            vis0  = sample["vis"]    # [B, T, H, W, C]
            pose0 = sample["pose"]   # [B, T, N_animals, D]

            B, T, H, W, C = vis0.shape
            _, _, N_animals, D = pose0.shape

            video_dim = H * W * C
            pose_dim  = N_animals * D


            ### Create the loop over different parameter settings

            model = ActionTransformerMAP(
                video_dim=video_dim,
                pose_dim =pose_dim,
                model_dim=model_dim_D,
                nhead    =n_heads,
                num_layers=n_layers,
                num_classes=len(train_ds.label_list),
                use_pose=UP, use_vis=UV,
                add_features=AF,
                in_channels=C,
                model_name=model_name
            ).to(device)


            # TODO: maybe without this pos weight
            if cls_reweight:
                criterion = nn.BCEWithLogitsLoss(pos_weight=train_ds.pos_weight.to(device=device))
            else:
                criterion = nn.BCEWithLogitsLoss()
            # Margin criterion
            margin_crit = nn.MultiLabelMarginLoss()

            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

            # (optional) LR scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=num_epochs)


            results_list = []

            loss_log = []



            # 4) Training loop
            for epoch in range(1, num_epochs+1):
                # ——— Train ———
                model.train()
                total_loss = 0.0
                seq_los_train = 0.0

                train_bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{num_epochs}", leave=True)
                for batch in train_bar:
                #for batch in train_loader:

                    print("Number of timesteps: ", T)

                    vis    = batch["vis"   ].to(device, non_blocking=True)  # [B, T, H, W, C]
                    pose   = batch["pose"  ].to(device, non_blocking=True)  # [B, T, N_animals, D]
                    mask   = batch["mask"  ].to(device, non_blocking=True)  # [B, T]
                    labels = batch["labels"].to(device, non_blocking=True)  # [B, L]

                    # flatten inputs
                    #video_in = vis.view(B, vis.size(1), -1)   # [B, T, video_dim]
                    pose_in  = pose.flatten(2)                # [B, T, pose_dim]

                    if torch.isnan(pose).any() or torch.isinf(pose).any():
                        print("NaNs/Infs in pose input!")

                    # Use one example batch (or a typical batch size/shape)
                    vis_cpu = batch["vis"].cpu()
                    pose_cpu = batch["pose"].cpu()
                    mask_cpu = batch["mask"].cpu()

                    pose_in_cpu = pose_cpu.flatten(2)  # match your forward: pose_in

                    flops = measure_flops_cpu(model, vis_cpu, pose_in_cpu, mask_cpu)
                    print(f"Total FLOPs: {flops:,}")

                    with open(save_flops_path, 'w') as f:
                        f.write(f"{flops}")

                    #flop_list.append(flops)
                    break
                break

