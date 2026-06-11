# Render a single reconstructed macaque pose frame from an action folder.
# Designed for the action_recognition / pose_reconstruction sibling-project layout.

import argparse
import ast
import json
import os
import sys
from os.path import join

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as SciRot


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_individuals(value):
    """Parse dataset individuals cell robustly: "['J','H']", "J,H", or "JH"."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if pd.isna(value):
        return []
    s = str(value).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except Exception:
        pass
    if "," in s:
        return [x.strip().strip("'\"") for x in s.split(",") if x.strip()]
    return [c for c in s if c.strip() and c not in "[]'\" "]


def parse_args():
    p = argparse.ArgumentParser(
        description="Render one macaque reconstructed pose frame with configurable global transform/camera."
    )

    p.add_argument("--nb-cameras", type=int, default=6,
                   help="Number suffix used in action_params_<IND>_<NB>.pth, e.g. 06.")
    p.add_argument("--render-col", type=str2bool, nargs="?", const=True, default=True,
                   help="Load individual vertex colors if available.")
    p.add_argument("--ambient-light", action="store_true", default=False,
                   help="Use AmbientLights instead of PointLights for debugging dark renders.")
    p.add_argument("--high-poly", action="store_true", default=False,
                   help="Use high-poly mesh and high-poly individual vertex offsets/colors.")
    p.add_argument("--compute-cfg", type=str, default="Setup_Local_cfg.json",
                   help="Config filename inside <project_root>/cfgs/.")
    p.add_argument("--specific-action-index", type=int, default=0,
                   help="Action folder/index to render.")
    p.add_argument("--individual-index", type=int, default=0,
                   help="Index into the individuals listed for the selected action.")
    p.add_argument("--display-individual", type=str, default="L",
                   choices=["J", "H", "L", "O", "N", "T", "C", "G"],
                   help="Identity/body/texture to display the loaded pose as.")

    p.add_argument("--frame-idx", type=int, default=0,
                   help="Requested original frame index. If invalid/None/NaN, the first valid pose at or after it is used.")
    p.add_argument("--canonical", action="store_true", default=False,
                   help="Canonical rendering: ignore global rotation and translation.")
    p.add_argument("--use-global-rotation", type=str2bool, nargs="?", const=True, default=True,
                   help="Apply params['R'] unless --canonical is set.")
    p.add_argument("--use-global-translation", type=str2bool, nargs="?", const=True, default=True,
                   help="Apply params['t'] unless --canonical is set.")

    p.add_argument("--azim", type=float, default=0.0,
                   help="Camera azimuth for PyTorch3D look_at_view_transform.")
    p.add_argument("--elev", type=float, default=30.0,
                   help="Camera elevation for PyTorch3D look_at_view_transform.")
    p.add_argument("--dist", type=float, default=2.5,
                   help="Camera distance for PyTorch3D look_at_view_transform.")
    p.add_argument("--image-size", type=int, default=1200,
                   help="Rendered square image size.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory. Default: <action_folder>/rendered_frames.")

    p.add_argument("--axis-align-euler", type=float, nargs=3, default=[-90.0, 90.0, 0.0],
                   help="XYZ Euler degrees applied to global R/t to align reconstruction space to render space.")
    p.add_argument("--no-axis-align", action="store_true", default=False,
                   help="Disable the fixed reconstruction-to-render-space axis alignment.")

    return p.parse_args()


def first_valid_frame(body_pose, requested_idx):
    """Return first valid frame >= requested_idx, else first valid frame in sequence."""
    n_frames = int(body_pose.shape[0])
    requested_idx = max(0, min(int(requested_idx), n_frames - 1))

    def is_valid(i):
        pose_i = body_pose[i]
        if pose_i is None:
            return False
        if torch.isnan(pose_i).any().item():
            return False
        return True

    for i in range(requested_idx, n_frames):
        if is_valid(i):
            return i
    for i in range(0, requested_idx):
        if is_valid(i):
            return i
    raise RuntimeError("No valid non-NaN body pose found in this action parameter file.")


def tensor_to_device(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def main():
    args = parse_args()

    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_path, ".."))
    repo_root = os.path.abspath(os.path.join(project_root, ".."))
    for path in (repo_root, project_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    from pose_reconstruction.src.utils.data_loader import ActionLoader
    from pose_reconstruction.src.Optimizers.BaseOpt import BaseOptimizer
    from pose_reconstruction.src.utils.LBS import MeshModel
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        PointLights,
        AmbientLights,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print("GPU available")
    else:
        device = torch.device("cpu")
        print("CPU usage")

    with open(join(project_root, "cfgs", args.compute_cfg), "r") as f:
        compute_cfg = json.load(f)

    hard_drive_loc = compute_cfg["hard_drive_loc"]
    path_to_dataset = join(hard_drive_loc, "dataset_overview.csv")
    path_to_poses = join(hard_drive_loc, "BigMaQ_latest_pose_reconstructions")

    drive_loc = join(project_root, r"data")

    if args.high_poly:
        vertex_weights_symmetry = join(drive_loc, r"Mesh/vertex_symmetry.json")
        path_to_mesh = join(drive_loc, r"Mesh/macaque.json")
    else:
        vertex_weights_symmetry = join(drive_loc, r"Mesh/vertex_symmetry_LOWPOLY.json")
        path_to_mesh = join(drive_loc, r"Mesh/macaque_LOWPOLY.json")

    df = pd.read_csv(path_to_dataset)
    action_index = args.specific_action_index

    if action_index in df.index:
        row = df.loc[action_index]
    elif "Unnamed: 0" in df.columns and action_index in set(df["Unnamed: 0"].astype(int)):
        row = df[df["Unnamed: 0"].astype(int) == action_index].iloc[0]
    else:
        row = df.iloc[action_index]

    action_folder = join(path_to_poses, f"{action_index}")
    os.makedirs(action_folder, exist_ok=True)

    individuals = parse_individuals(row["individuals"])
    if not individuals:
        raise RuntimeError(f"No individuals found for action {action_index}: {row.get('individuals')}")

    print(f"Individuals of action {action_index}: {individuals}")
    if args.individual_index < 0 or args.individual_index >= len(individuals):
        print(f"Requested --individual-index {args.individual_index} is out of range. Falling back to 0.")
        individual_index = 0
    else:
        individual_index = args.individual_index

    pose_individual_loaded = individuals[individual_index]
    print(f"Original pose identity loaded from action: {pose_individual_loaded}")
    print(f"Display action as individual/body identity: {args.display_individual}")

    nb = str(args.nb_cameras).zfill(2)
    pose_path = join(action_folder, f"action_params_{pose_individual_loaded}_{nb}.pth")
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    params = torch.load(pose_path, map_location=device)

    body_pose = tensor_to_device(params["body_pose"], device)
    used_frame_idx = first_valid_frame(body_pose, args.frame_idx)
    print(f"Requested frame idx: {args.frame_idx}; rendered original frame idx: {used_frame_idx}")

    mesh_model = MeshModel(mesh=path_to_mesh, device=device)

    # Dummy loader/optimizer are used only to reuse the existing LBS posing pipeline.
    action_loader = ActionLoader(
        path_to_label_split=None,
        path_to_dataset=path_to_dataset,
        vertex_symm_path=vertex_weights_symmetry,
        device=device,
        max_image_size=None,
        labels_only=None,
        debug=None,
        same_size=None,
        high_poly=args.high_poly,
        dummy=True,
        project_root=project_root,
        hard_drive_loc=hard_drive_loc
    )

    opt = BaseOptimizer(
        device=device,
        mesh_model=mesh_model,
        action_loader=action_loader,
        cameras_to_retain=['1', '2'],
        template_stretch=True
    )

    opt.action_and_keyframes = [action_index]
    opt.action_frame_index = 0
    opt.init_model_params(save=False)

    # Load display identity/body parameters, while preserving the original pose sequence identity above.
    if args.high_poly:
        ind_params_path = join(hard_drive_loc, "IndividualFits", f"{args.display_individual}_Color.pth")
    else:
        ind_params_path = join(hard_drive_loc, "IndividualFits", f"{args.display_individual}_Color_False.pth")
    if not os.path.exists(ind_params_path):
        raise FileNotFoundError(f"Display individual fit not found: {ind_params_path}")

    ind_params = torch.load(ind_params_path, map_location=device)
    if "bone_lengths" in ind_params:
        opt.bone_length = tensor_to_device(ind_params["bone_lengths"], device)
    if "vertex_offsets" in ind_params:
        opt.vertex_offsets = tensor_to_device(ind_params["vertex_offsets"], device)
    if args.render_col and "rgb" in ind_params:
        opt.vertex_colors = tensor_to_device(ind_params["rgb"], device)
    if "s" in ind_params:
        opt.global_scale = tensor_to_device(ind_params["s"], device)

    opt.body_pose[0, :, :] = body_pose[used_frame_idx, :, :]

    canonical = args.canonical
    use_rot = args.use_global_rotation and not canonical
    use_t = args.use_global_translation and not canonical

    align_rot = SciRot.identity()
    if not args.no_axis_align:
        align_rot = SciRot.from_euler("xyz", args.axis_align_euler, degrees=True)

    # Global orientation.
    if use_rot and "R" in params:
        raw_rotvec = params["R"][used_frame_idx, 0, :].detach().cpu().numpy()
        rotvec = (align_rot * SciRot.from_rotvec(raw_rotvec)).as_rotvec()
    else:
        rotvec = align_rot.as_rotvec() if not args.no_axis_align else np.zeros(3, dtype=np.float32)

    # Global translation.
    if use_t and "t" in params:
        t = params["t"][used_frame_idx].detach().cpu().numpy().squeeze()
        if not args.no_axis_align:
            t = align_rot.as_matrix() @ t
    else:
        t = np.zeros(3, dtype=np.float32)

    opt.global_orientation.requires_grad = False
    opt.global_orientation[0, 0, :] = torch.as_tensor(rotvec, device=device, dtype=opt.global_orientation.dtype)
    opt.global_t.requires_grad = False
    opt.global_t[0, :, :] = torch.as_tensor(t, device=device, dtype=opt.global_t.dtype).view(1, 3)

    print(f"Using global rotation: {use_rot}; using global translation: {use_t}; canonical: {canonical}")

    opt.__pose_mesh__()
    posed_mesh = opt.posed_mesh

    R_cam, T_cam = look_at_view_transform(dist=args.dist, elev=args.elev, azim=args.azim, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R_cam, T=T_cam)
    raster_settings = RasterizationSettings(
        image_size=args.image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    light_loc = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32, device=device)
    if not args.no_axis_align:
        light_loc_np = light_loc.detach().cpu().numpy() @ SciRot.from_euler("y", args.azim, degrees=True).as_matrix().T
        light_loc = torch.as_tensor(light_loc_np, dtype=torch.float32, device=device)

    if args.ambient_light:
        lights = AmbientLights(device=device)
    else:
        lights = PointLights(device=device, location=light_loc)
        lights.location = light_loc

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    with torch.no_grad():
        image = renderer(posed_mesh)[0].detach().cpu().numpy()

    rgb_float = image[..., :3]
    alpha_float = image[..., 3]

    # Robust conversion:
    # PyTorch3D normally returns 0..1 floats, but older/local shader setups can
    # already behave like 0..255. Convert according to observed range.
    if rgb_float.max() <= 1.5:
        rgb = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.clip(rgb_float, 0, 255).astype(np.uint8)

    alpha = (alpha_float > 0).astype(np.uint8) * 255
    rgba = np.dstack([rgb, alpha])

    out_dir = args.output_dir or join(action_folder, "rendered_frames")
    os.makedirs(out_dir, exist_ok=True)
    out_name = (
        f"action_{action_index}"
        f"_poseInd_{pose_individual_loaded}"
        f"_displayInd_{args.display_individual}"
        f"_requestedFrame_{args.frame_idx}"
        f"_renderedFrame_{used_frame_idx}"
        f"_rot_{int(use_rot)}_t_{int(use_t)}_canon_{int(canonical)}"
        f"_azim_{args.azim:g}_elev_{args.elev:g}_dist_{args.dist:g}.png"
    )
    out_path = join(out_dir, out_name)
    Image.fromarray(rgba, mode="RGBA").save(out_path)
    print(f"Saved render: {out_path}")


if __name__ == "__main__":
    main()
