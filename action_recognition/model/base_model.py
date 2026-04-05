import torch
import torch.nn as nn



class ActionTransformer(nn.Module):
    def __init__(self, video_dim, pose_dim, model_dim=512, nhead=8, num_layers=3, num_classes=20, use_pose=True):
        super().__init__()

        self.use_pose = use_pose

        # Linear projections for video and pose (flatten pose [J,3] to J*3)
        self.video_proj = nn.Linear(video_dim, model_dim)
        self.pose_proj  = nn.Linear(pose_dim,  model_dim)
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        # Transformer Encoder stack
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classification head
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, video_batch, pose_batch, padding_mask):
        """
        video_batch: tensor (B, T, D_v)
        pose_batch:  tensor (B, T, J, 3)
        padding_mask: bool tensor (B, T) with True at padded positions
        """
        B, T, _ = video_batch.shape

        # Project and fuse
        video_emb = self.video_proj(video_batch)  # (B, T, model_dim)

        # pose embedding only if enabled
        if self.use_pose:
            # pose_batch: (B, T, pose_dim)
            p_emb = self.pose_proj(pose_batch)  # (B, T, E)
        else:
            # zero‐out pose contribution
            p_emb = torch.zeros_like(video_emb)  # (B, T, E)

        frame_emb = video_emb + p_emb          # (B, T, model_dim)
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, model_dim)
        x = torch.cat([cls_tokens, frame_emb], dim=1)  # (B, T+1, model_dim)
        # Update padding mask to account for CLS (no padding for CLS)
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=padding_mask.device)
        src_key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, T+1)
        # Permute to Transformer input shape (S, B, E)
        x = x.permute(1, 0, 2)  # (S=T+1, B, model_dim)
        # Transformer encoder (ignoring padded positions in attention)
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (S, B, model_dim)
        # Classification using [CLS] token's output (first token)
        cls_out = out[0]           # (B, model_dim)
        logits = self.classifier(cls_out)  # (B, num_classes)

        return logits



class SpatialAttnPoolMLP(nn.Module):
    """
    Inputs:  x  [B, T, H, W, C]
    Output:  y  [B, T, D_out]   (default D_out=256)

    Steps:
      1) reshape to tokens [B*T, HW, C]
      2) learnable single/query multi-head attention to pool over HW
      3) 2-layer MLP to encode pooled vector -> D_out
    """
    def __init__(
        self,
        in_channels: int,           # C
        d_out: int = 256,           # target per-frame dim
        num_heads: int = 8,
        mlp_hidden: int = 512,
    ):
        super().__init__()
        self.pre_ln = nn.LayerNorm(in_channels)

        # learnable pooling query (one per batch element; expanded at runtime)
        self.query = nn.Parameter(torch.randn(1, 1, in_channels))

        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True  # [B*, L, C]
        )

        # 2-layer MLP head to get to d_out
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, d_out),
            nn.ReLU(inplace=True),
        )

        # used if an entire timestep is padded (avoid NaNs)
        self.pad_embedding = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x: torch.Tensor, t_mask: torch.Tensor):
        # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape
        BT, HW = B * T, H * W

        tokens = x.view(B*T, H*W, C)                 # [B*T, HW, C]
        tokens = self.pre_ln(tokens)                 # pre-norm

        # build key_padding_mask from temporal mask
        if t_mask is None:
            kpm = torch.zeros(BT, HW, dtype=torch.bool, device=x.device)
        else:
            kpm = t_mask.view(BT, 1).expand(BT, HW)  # [B*T, HW], True = ignore

        # handle fully-masked frames
        all_masked = kpm.all(dim=1)  # Get them in B*T if masked or not
        if all_masked.any():  # Check if there is masking to be done
            kpm = kpm.clone()
            kpm[all_masked, 0] = False  # unmask one dummy key, even for masked frames


        q = self.query.expand(BT, 1, C)  # [B*T, 1, C]
        pooled, _ = self.mha(q, tokens, tokens, key_padding_mask=kpm)        # [B*T, 1, C]
        pooled = pooled.squeeze(1)                     # [B*T, C]

        # Check for masking
        if all_masked.any():
            # Use the pad embedding for masked timesteps
            pooled[all_masked] = self.pad_embedding

        y = self.mlp(pooled)                           # [B*T, d_out]
        return y.view(B, T, -1)                        # [B, T, d_out]



class ActionTransformerMAP(nn.Module):
    def __init__(self,
                 video_dim: int,
                 pose_dim:  int,
                 in_channels: int = 768,
                 model_dim: int = 512,
                 nhead:     int = 8,
                 num_layers:int = 3,
                 num_classes:int = 20,
                 use_pose: bool = True,
                 use_vis: bool = True,
                 add_features: bool = False,
                 bottleneck_dim: int = 64,
                 hidden_layer_size: int = 128,
                 model_name = None):




        super().__init__()



        #self.model_names = ["resnet50", "movinet-a2", "dinov2-base-cls","dinov2-base-patch" ,"timesformer-base-finetuned-k400", "videoprism_public_v1_base_hf"]
        self.model_names = ["resnet50", "movinet-a2", "dinov2-base-cls", "vit-base-cls",
                     "dinov2-base-patch", "vit-base-patch","timesformer-base-finetuned-k400", "videoprism_public_v1_base_hf"]



        assert model_name in self.model_names

        self.model_dim = model_dim

        self.use_pose = use_pose
        self.use_vis = use_vis
        self.add_features = add_features

        # ─── 1) per-frame conv‐bottleneck ─────────────────────────────────────
        # we'll apply this to each frame independently:
        # input: (B*T, in_channels, H, W)
        # 1×1 → bottleneck_dim
        # 3×3 → bottleneck_dim (pad=1 so H,W unchanged)
        # 1×1 → model_dim

        # Normalize channels in case some models are not normalizing well enough
        # self.input_norm = nn.BatchNorm2d(in_channels)
        # self.frame_cnn = nn.Sequential(
        #     # apply normalization first
        #     self.input_norm,
        #     nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(bottleneck_dim, model_dim, kernel_size=1),
        #     nn.ReLU(inplace=True)
        # )

        #self.pose_proj  = nn.Linear(pose_dim,  model_dim)
        # ▶▶ New: pose MLP with two hidden layers
        #    pose_dim → hid1 → hid2 → model_dim
        hid1 = hidden_layer_size
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(pose_dim),
            nn.Linear(pose_dim, hid1),
            nn.ReLU(inplace=True),
            nn.Linear(hid1, model_dim),
            nn.ReLU(inplace=True),
        ) if use_pose else None



        if model_name in self.model_names[-4:]:
            # Visual features reduction
            self.vis_enc = SpatialAttnPoolMLP(
                in_channels=in_channels,  # C from the transformer features
                d_out=model_dim,  # 256
                num_heads=nhead,
                mlp_hidden=hid1
            )
        else:
            # Here change to the channels
            self.vis_enc = nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, hid1),
                nn.ReLU(inplace=True),
                nn.Linear(hid1, model_dim),
                nn.ReLU(inplace=True),
            ) if use_vis else None



        self.v_norm = nn.LayerNorm(model_dim)
        self.p_norm = nn.LayerNorm(model_dim)

        # extra fusion layer:
        self.fuse_lin = nn.Linear(2 * model_dim, model_dim)

        # --- remove the old CLS token stuff ---
        # instead, we’ll pool with a learned query
        self.map_query = nn.Parameter(torch.randn(1, model_dim))

        # the pooler itself
        # we set batch_first=True so we can work with [B, T, E]
        self.attn_pool = nn.MultiheadAttention(
            embed_dim   = model_dim,
            num_heads   = nhead,
            batch_first = True
        )

        # Transformer Encoder stack stays the same
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model = model_dim,
            nhead   = nhead
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers
        )

        # final classification head
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self,
                video_batch:  torch.Tensor,
                pose_batch:   torch.Tensor,
                padding_mask: torch.Tensor):
        """
        video_batch:  [B, T, video_dim]   (flattened H*W*C)
        pose_batch:   [B, T, pose_dim]    (flattened N_animals × 81)
        padding_mask: [B, T]              (True == padded)
        """


        B, T, H, W, C = video_batch.shape

        ### For movinet 640 = C, resnet50 2048


        if self.use_vis:

            if H == 1:
                # Entire frame encodings
                v = video_batch.view(B*T, -1)
                v = self.vis_enc(v)
                v = v.view(B, T, -1)
            else:
                # Pooling with SpatialAttention
                v = self.vis_enc(video_batch, padding_mask)
        else:
            v = torch.zeros(B, T, self.model_dim, device=video_batch.device, dtype=video_batch.dtype)

        # 2) Pose embedding via two‐layer MLP
        if self.use_pose:
            # flatten pose to [B*T, pose_dim] then unflatten
            p = pose_batch.view(B * T, -1)
            p = self.pose_proj(p)  # [B*T, model_dim]
            p = p.view(B, T, -1)  # [B, T, model_dim]
        else:
            p = torch.zeros_like(v)

        # Normalize before cat, for pose as the paper for on human pose 3D action recognition (for p at least)
        p = self.p_norm(p)
        v = self.v_norm(v)

        if self.add_features:
            x = v + p  # [B, T, E]
        else:
            # concat along feature dim
            x_cat = torch.cat([v, p], dim=-1)  # [B, T, 2E]
            x = self.fuse_lin(x_cat)  # [B, T, E]

        # 2) Transformer (it expects [T, B, E] by default)
        #    here we permute so we don’t have to do CLS-trick
        x = x.permute(1, 0, 2)  # -> [T, B, E]


        # todo: maybe add a positional encoding, so it is easier to learn the dynamics for visual features


        # but MultiheadAttention(pool) wants batch_first,
        # so we’ll convert back in a moment
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # [T, B, E]
        x = x.permute(1, 0, 2)  # -> [B, T, E]

        # 3) MAP pooling:
        #    expand our learned query to the batch
        #    shape [B, 1, E]
        B, T, E = x.shape
        q = self.map_query.unsqueeze(0).expand(B, -1, -1)

        #    attend q over x (keys=values=x)
        #    note: attn_pool returns (output, attn_weights)
        pooled, _ = self.attn_pool(
            query           = q,         # [B, 1, E]
            key             = x,         # [B, T, E]
            value           = x,         # [B, T, E]
            key_padding_mask=padding_mask
        )
        # pooled: [B, 1, E] → squeeze to [B, E]
        pooled = pooled.squeeze(1)

        # 4) classify
        logits = self.classifier(pooled)   # [B, num_classes]

        return logits


