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


batch_size  = 4#4#sub_sample_map[sub_sample]
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
pose_idx = 0

run_idx = 2

use_stopper = True

model_name = model_activations[model_idx]

# Make it iterate, single yes no, subsample by 4 maybe, and global_info yes/no and write to a table

# Change the order of the runs, most potent at the start

# Run for 20 epochs for now and look at results. early stopping uses very early

# Bottleneck of training is likely just the disk retrieval!, for less features way faster

# Skip the pose run,
skip_pose_run=True

# to return to faster training, unmount and mount and restart pycharm

all_runs = [all_runs[run_idx]] #todo: comment if multiple
#for run in all_runs[run_idx]:
seed_list = [42, 43, 44]
seed_list = [42]
seed_list = [43, 44]

seed_list = [45]
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
                        f"{pose_space}_valratio{val_ratio}_clsw_{cls_reweight}_stop_{use_stopper}_fold_{fold}")



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
                    vis    = batch["vis"   ].to(device, non_blocking=True)  # [B, T, H, W, C]
                    pose   = batch["pose"  ].to(device, non_blocking=True)  # [B, T, N_animals, D]
                    mask   = batch["mask"  ].to(device, non_blocking=True)  # [B, T]
                    labels = batch["labels"].to(device, non_blocking=True)  # [B, L]

                    # flatten inputs
                    #video_in = vis.view(B, vis.size(1), -1)   # [B, T, video_dim]
                    pose_in  = pose.flatten(2)                # [B, T, pose_dim]

                    if torch.isnan(pose).any() or torch.isinf(pose).any():
                        print("NaNs/Infs in pose input!")


                    logits = model(vis, pose_in, mask)   # [B, L] raw scores

                    B, C = labels.shape # get rid of this
                    # start with all -1
                    target_idx = torch.full((B, C), -1, dtype=torch.long, device=labels.device)

                    for i in range(B):
                        # find the positive classes for sample i
                        pos = (labels[i] > 0.5).nonzero(as_tuple=False).view(-1)
                        # fill the first pos.numel() entries with those indices
                        target_idx[i, :pos.numel()] = pos

                    # 3) compute loss
                    loss_margin = margin_crit(logits, target_idx)

                    # todo: test without
                    loss   = criterion(logits, labels) + w1 * loss_margin

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # --- compute micro-F1 for this batch ---
                    with torch.no_grad():
                        # 1) probabilities → binary preds (threshold=0.5)
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).long()  # [B, L]
                        target = labels.long()  # [B, L]
                        # 2) flatten to vectors of length (B×L)
                        preds_flat = preds.view(-1).cpu().numpy()
                        target_flat = target.view(-1).cpu().numpy()
                        # 3) compute micro-F1 via sklearn
                        batch_f1 = f1_score(target_flat, preds_flat, average="micro", zero_division=0)

                        # sequence‐level accuracy: correct iff **all** labels match
                        seq_correct = preds.eq(target).all(dim=1).float()  # [B]
                        batch_seq_acc = seq_correct.mean().item()  # scalar


                    total_loss += loss.item() * vis.size(0)
                    seq_los_train += batch_seq_acc * vis.size(0)


                    train_bar.set_postfix(
                        batch_loss=f"{loss.item():.4f}",
                        batch_f1=f"{batch_f1:.4f}",
                        batch_seq_acc=f"{batch_seq_acc:.4f}",
                        refresh=False
                    )

                avg_train_loss = total_loss / len(train_ds)
                avg_train_seq = seq_los_train / len(train_ds)

                scheduler.step()


                model.eval()
                total_val_loss = 0.0
                total_bin_loss = 0.0

                #if val_ratio > 0:

                all_preds = []
                all_targets = []
                all_probs = []

                if val_ratio > 0:
                    with torch.no_grad():
                        for batch in val_loader:
                            vis    = batch["vis"   ].to(device, non_blocking=True)
                            pose   = batch["pose"  ].to(device, non_blocking=True)
                            mask   = batch["mask"  ].to(device, non_blocking=True)
                            labels = batch["labels"].to(device, non_blocking=True)
                            B, C = labels.shape

                            pose_in = pose.flatten(2)

                            logits = model(vis, pose_in, mask)

                            # start with all -1
                            target_idx = torch.full((B, C), -1, dtype=torch.long, device=labels.device)

                            for i in range(B):
                                # find the positive classes for sample i
                                pos = (labels[i] > 0.5).nonzero(as_tuple=False).view(-1)
                                # fill the first pos.numel() entries with those indices
                                target_idx[i, :pos.numel()] = pos

                            # 3) compute loss
                            loss_margin = margin_crit(logits, target_idx)

                            #todo: margin loss in the val loss?
                            bin_loss = criterion(logits, labels)
                            loss = bin_loss + w1 * loss_margin

                            total_val_loss += loss.item() * vis.size(0)
                            total_bin_loss += bin_loss.item() * vis.size(0)

                            # For metrics
                            probs = torch.sigmoid(logits)
                            all_probs.append(probs.cpu().numpy())
                            preds = (probs > 0.5).long().cpu().numpy()
                            targs = labels.long().cpu().numpy()
                            all_preds.append(preds)
                            all_targets.append(targs)

                        # stack everything
                        Y_score = np.vstack(all_probs)  # [N, L] float
                        Y_true = np.vstack(all_targets)  # [N, L] 0/1
                        Y_pred = np.vstack(all_preds)  # [N, L] 0/1

                        ap_per_class = []

                        for k in range(Y_true.shape[1]):
                            # If a class never appears in GT, its AP is undefined; skip via NaN
                            if Y_true[:, k].sum() == 0:
                                ap_per_class.append(np.nan)
                            else:
                                ap_per_class.append(average_precision_score(Y_true[:, k], Y_score[:, k]))

                        mAP_macro = np.nanmean(ap_per_class)  # mean of class APs

                    avg_val_loss = total_val_loss / len(val_ds)
                    avg_bin_loss = total_bin_loss / len(val_ds)

                else:
                    avg_val_loss = 0
                    mAP_macro = 0
                    avg_bin_loss = 0

                loss_log.append({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "avg_train_seq": avg_train_seq,
                    "avg_val_loss": avg_val_loss,
                    "avg_bin_loss": avg_bin_loss,
                    "val_mAP": mAP_macro
                })



                # Optionally, write to CSV every epoch
                pd.DataFrame(loss_log).to_csv(
                    os.path.join(run_dir, "loss_log.csv"),
                    index=False
                )

                print(f"[Epoch {epoch}/{num_epochs}] "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Train Seq: {avg_train_seq:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} | "
                      f"Avg bin loss: {avg_bin_loss:.4f} | "
                      f"Val_mAP:   {mAP_macro:.4f}")

                if val_ratio > 0 and use_stopper:
                    if stopper.step(mAP_macro, model=model, epoch=epoch):
                        break

            if val_ratio > 0 and use_stopper:
                # Restore the best epoch
                stopper.restore(model)


            ### Make a nice plot of the loss log




            model.eval()
            # to accumulate all preds/targets
            all_preds = []
            all_targets = []
            all_probs = []  # collect raw scores for mAP
            with torch.no_grad():
                for batch in test_loader:
                    vis    = batch["vis"   ].to(device, non_blocking=True)
                    pose   = batch["pose"  ].to(device, non_blocking=True)
                    mask   = batch["mask"  ].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)


                    pose_in  = pose.flatten(2)
                    logits = model(vis, pose_in, mask)
                    # store for F1
                    probs = torch.sigmoid(logits)
                    all_probs.append(probs.cpu().numpy())
                    preds = (probs > 0.5).long().cpu().numpy()
                    targs = labels.long().cpu().numpy()
                    all_preds.append(preds)
                    all_targets.append(targs)
            # stack everything
            Y_score = np.vstack(all_probs)  # [N, L] float
            Y_true = np.vstack(all_targets)  # [N, L] 0/1
            Y_pred = np.vstack(all_preds)  # [N, L] 0/1

            # Save the numpy arrays
            np.savez(
                os.path.join(run_dir, f"test_outputs.npz"),
                Y_score=Y_score,  # sigmoid probabilities
                Y_pred=Y_pred,  # thresholded at 0.5
                Y_true=Y_true,  # ground truth
                labels=np.array(train_ds.label_list, dtype=object)
            )

            # compute micro-F1
            test_f1 = f1_score(Y_true.flatten(), Y_pred.flatten(), average="micro", zero_division=0)
            # Exact-match sequence accuracy (multi-hot sets must match exactly)
            seq_acc = np.all(Y_pred == Y_true, axis=1).mean()
            # 1) per-class accuracy: fraction of samples correctly predicted for each class
            per_class_acc = (Y_pred == Y_true).sum(axis=0) / Y_true.shape[0]
            # 2) optional: macro‐average accuracy
            macro_acc = per_class_acc.mean()
            # —— mAP (per-class AP averaged) —— #
            ap_per_class = []

            for k in range(Y_true.shape[1]):
                # If a class never appears in GT, its AP is undefined; skip via NaN
                if Y_true[:, k].sum() == 0:
                    ap_per_class.append(np.nan)
                else:
                    ap_per_class.append(average_precision_score(Y_true[:, k], Y_score[:, k]))

            mAP_macro = np.nanmean(ap_per_class)  # mean of class APs
            mAP_micro = average_precision_score(Y_true, Y_score, average="micro")  # micro-AP
            # (optional) keep per-class APs in results row with class names
            #per_class_ap_dict = {label: ap for label, ap in zip(train_ds.label_list, ap_per_class)}
            # Use prefixes so APs and Accs don't collide
            per_class_ap_dict = {f"AP::{label}": ap for label, ap in zip(train_ds.label_list, ap_per_class)}
            per_class_acc_dict = {f"acc::{label}": acc for label, acc in zip(train_ds.label_list, per_class_acc)}
            THRESH = 0.5
            cm = multilabel_confusion_matrix(Y_true, (Y_score > THRESH).astype(int))  # shape (L, 2, 2)
            # Save raw npz (good for later analysis)
            np.savez(
                os.path.join(run_dir, f"confmat.npz"),
                cm=cm,
                labels=np.array(train_ds.label_list),
                threshold=THRESH
            )

            # Also save a readable CSV (TN, FP, FN, TP per class)
            rows = []
            for lab, mat in zip(train_ds.label_list, cm):
                tn, fp, fn, tp = mat.ravel()
                rows.append({"label": lab, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
            pd.DataFrame(rows).to_csv(os.path.join(run_dir, f"confmat.csv"), index=False)
            # append to results
            row = {
                "test_f1": test_f1,
                "Seq-acc-test": seq_acc,
                "macro_acc": macro_acc,
                "mAP_macro": mAP_macro,
                "mAP_micro": mAP_micro,
                **per_class_ap_dict,
                **per_class_acc_dict
            }
            # instead of numeric indices, use the actual class names:
            for label, acc in zip(train_ds.label_list, per_class_acc):
                row[label] = acc
            results_list.append(row)
            print(
                f" Test F1:   {test_f1:.4f} | "
                f" MacroAcc: {macro_acc:.4f}  |  "
                f"mAP_macro: {mAP_macro}   |  "
                f"mAP_micro: {mAP_micro}   |  "
                f" Seq–level Accuracy (exact match): {seq_acc:.4f}"
            )
            # 4) print the entire per-class‐accuracy row
            labels = train_ds.label_list
            acc_items = [f"{lab}:{acc:.3f}" for lab, acc in zip(labels, per_class_acc)]
            acc_row = " | ".join(acc_items)
            print(f"    Per-class Acc: [{acc_row}]")


            # Save model + metrics
            torch.save(model.state_dict(), os.path.join(run_dir, "model.pth"))

            df = pd.DataFrame(results_list)

            # Build a sensible column order
            base_cols = ["test_f1", "Seq-acc-test",
                "macro_acc", "mAP_macro", "mAP_micro"
            ]

            ap_cols = [f"AP::{lab}" for lab in train_ds.label_list]
            acc_cols = [f"acc::{lab}" for lab in train_ds.label_list]

            cols = [c for c in base_cols if c in df.columns] + ap_cols + acc_cols
            df.to_csv(os.path.join(run_dir, "metrics.csv"), columns=cols, index=False)

            del train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, model
            gc.collect()
            torch.cuda.empty_cache()


