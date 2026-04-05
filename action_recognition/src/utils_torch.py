from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch
import torch
from copy import deepcopy
import os

def collate_fn(batch, max_pad_length=None):
    """
    batch: list of (vis, pose, labels, cam, T)
      - vis:  [T, H, W, C]
      - pose: [T, N_animals, D]            # here D == 27 joints * 3 = 81
      - labels: [L]
      - cam: int
      - T: sequence length
    """




    vis_list, pose_list, lab_list, cam_list, T_list = zip(*batch)

    # 1) determine pad length
    T_pad = max(T_list) if max_pad_length is None else max_pad_length
    B = len(batch)

    # 2) determine target H,W,C for vis
    #    Prefer H,W from any 4D sample; else default to 1,1.
    H = W = 1
    C = None
    device = vis_list[0].device if torch.is_tensor(vis_list[0]) else None
    dtype_vis = vis_list[0].dtype if torch.is_tensor(vis_list[0]) else torch.float32
    for v in vis_list:
        if v.ndim == 4:
            _, H, W, C = v.shape
            break
    if C is None:
        # all were [T,C]
        C = vis_list[0].shape[-1]


    #    and animals/features dims for pose
    n_animals, D = pose_list[0].shape[1:]
    dtype_pose = pose_list[0].dtype
    device = pose_list[0].device

    # 3) allocate zero‐buffers
    # 4) allocate padded tensors
    vis_padded = torch.zeros((B, T_pad, H, W, C), dtype=dtype_vis, device=device)
    pose_padded = torch.zeros((B, T_pad, n_animals, D), dtype=dtype_pose, device=device)

    # 5) copy each sample
    for i, (v, p, t) in enumerate(zip(vis_list, pose_list, T_list)):
        # v: [T,H,W,C] or [T,C] -> reshape to [T,H,W,C]
        if v.ndim == 2:  # [T,C] -> [T,1,1,C]
            v = v.unsqueeze(1).unsqueeze(1)
        elif v.ndim != 4:
            raise ValueError(f"vis item ndim={v.ndim}, expected 2 or 4")

        # basic shape check (H,W,C must match targets)
        if v.shape[1:] != (H, W, C):
            raise ValueError(f"vis shape mismatch: got {v.shape[1:]}, expected {(H, W, C)}")

        vis_padded[i, :t] = v
        pose_padded[i, :t] = p

    # 5) stack labels & cams
    labels = torch.stack(lab_list, dim=0)  # [B, L]
    cams = torch.tensor(cam_list, dtype=torch.long)  # [B]

    # 6) build attention mask [B, T_pad]
    lengths = torch.tensor(T_list, dtype=torch.long)  # [B]
    arange = torch.arange(T_pad, dtype=torch.long)[None]  # [1, T_pad]
    mask = (arange >= lengths[:, None])  # [B, T_pad]


    return {
        "vis": vis_padded,  # [B, T_pad, H, W, C]
        "pose": pose_padded,  # [B, T_pad, N_animals, D]
        "labels": labels,  # [B, L]
        "cam": cams,  # [B]
        "mask": mask  # [B, T_pad]
    }

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, min_delta_rel=0.0, restore_best=True, run_dir=None, device="cpu"):
        """
        Stop if mAP hasn't improved by at least:
          max(min_delta, min_delta_rel * best) for `patience` epochs.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_delta_rel = min_delta_rel
        self.restore_best = restore_best
        self.best = float('-inf')
        self.num_bad = 0
        self.best_state = None
        self.best_epoch = -1
        self.should_stop = False
        self.run_dir = run_dir
        self.device = device

    def step(self, val_loss, model=None, epoch=None):

        # Special-case the first observation
        # if self.best == float('inf'):
        #     improved = True
        # else:
        #     threshold = max(self.min_delta, self.min_delta_rel * self.best)
        #     improved = (self.best - val_loss) > threshold

        if self.best == float('-inf'):
            improved = True
        else:
            threshold = max(self.min_delta, self.min_delta_rel *  abs(self.best))
            #improved = (self.best - val_loss) > threshold
            improved = (val_loss - self.best) > threshold


        if improved:
            self.best = float(val_loss)
            self.num_bad = 0
            if self.restore_best and model is not None:
                self.best_state = deepcopy(model.state_dict())
                if self.run_dir is not None and (epoch is not None):
                    torch.save(model.state_dict(), os.path.join(self.run_dir, f"best.pth"))

            if epoch is not None:
                self.best_epoch = int(epoch)
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore(self, model):

        if not self.restore_best or model is None:
            return

        print(f"Load best epoch: {self.best_epoch}")
        path = os.path.join(self.run_dir, f"best.pth") if self.run_dir else None
        if path and os.path.isfile(path):
            state = torch.load(path, map_location=self.device)
            model.load_state_dict(state, strict=True)
        elif self.best_state is not None:
            model.load_state_dict(self.best_state, strict=True)

