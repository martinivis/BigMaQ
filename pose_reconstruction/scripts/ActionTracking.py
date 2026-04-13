"""
Script that iterates the actions of the dataset
"""


import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description="Run macaque optimization with configurable CLI flags."
    )

    p.add_argument("--nb-cameras", type=int, default=6,
                   help="How many cameras to retain in optimization (1-16).")
    p.add_argument("--render-col", action="store_true", default=False,
                   help="Render in colors.")
    p.add_argument("--high-poly", action="store_true", default=False,
                   help="Use high-poly mesh.")

    return p.parse_args()

def main():
    args = parse_args()

    import sys
    import os
    from os.path import join
    # Get the path to the current file's directory
    current_path = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to the level where 'src' is located
    project_root = os.path.abspath(os.path.join(current_path, ".."))
    src_path = os.path.join(project_root, "src")

    sys.path.append(current_path)
    sys.path.append(project_root)
    sys.path.append(current_path)

    from pose_reconstruction.src.utils.data_loader import ActionLoader
    from pose_reconstruction.src.Optimizers.TimeOpt import TimeOptimizer
    from pose_reconstruction.src.utils.LBS import MeshModel
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print("GPU available")
    else:
        device = torch.device("cpu")
        print("CPU usage")

    high_poly = args.high_poly

    # Switch between exporting the action to MAMMAL or running the optimization here
    #export_path = None

    drive_loc = join(project_root, r"data")

    if high_poly:
        vertex_weights_symmetry = join(drive_loc, r"Mesh/Macaque_MDL_aslist_vertices.json")
        path_to_mesh = join(drive_loc, r"Mesh/Macaque_extended_joints_MeshInfoHighPoly.json")
    else:
        vertex_weights_symmetry = join(drive_loc, r"Mesh/vertex_info_hands_excluded_LOWPOLY.json")
        path_to_mesh = join(drive_loc, r"Mesh/Macaque_LOWPolyLastRotScaled.json")

    max_img_size = 100


    if max_img_size == 256:
        mini_batch_size = 30
    elif max_img_size == 200:
        mini_batch_size = 40 # high poly usually less
    else:
        mini_batch_size = 130


    time_scalar = 100


    print(
        f"(Image Size, HighPoly) ({max_img_size}, {high_poly}); (mini_batch_size, workers) ({mini_batch_size})")

    worker = 0
    nb_workers = 1

    force_action_reload = False
    squared_images = True
    #max_img_size = 250
    labels_only = False
    verbose = 0
    debug = False
    nb_ind_actions = 6

    hard_drive_loc = r"/media/lucas/X-2/BigMaQ"

    path_to_dataset = join(hard_drive_loc, "dataset_overview.csv")

    action_loader = ActionLoader(path_to_label_split=None, path_to_dataset=path_to_dataset,
                                 vertex_symm_path=vertex_weights_symmetry,
                                         device=device, max_image_size=max_img_size, labels_only=labels_only, debug=debug,
                                 high_poly=high_poly, force_action_reload=force_action_reload, same_size=squared_images,
                                 hard_drive_loc=hard_drive_loc)

    action_loader.disable_saving = False

    action_loader.project_root = project_root
    # Create the mesh model
    mesh_model = MeshModel(mesh=path_to_mesh, device=device)

    cameras_to_retain = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]


    nb_cameras = args.nb_cameras
    cameras_to_retain = cameras_to_retain[:nb_cameras]
    use_kp_conf = True

    use_bbox_conf = True
    # time scalar

    use_base_line = True

    cam_approach = 0
    small_change_note = f"EntireSetOpt{time_scalar:.3f}_complete_overlap_flow_mvtrack_basedist_{action_loader.min_cams_cross_view}_{use_base_line}_large_T"
    # Save time by not rendering all the time
    epoch_render_mod = 100 #todo: increase for test run, MLCLUSTER
    render_video = True
    # Take all individuals
    nb_ind_actions = 3
    changes_string = (f"Cams({nb_cameras})_MaxIm({max_img_size})_Square({squared_images})_KPconf({use_kp_conf})_"
                      f"BBconf({use_bbox_conf})_HP{high_poly}_Note({small_change_note})")
    time_optimizer = TimeOptimizer(device=device, mesh_model=mesh_model, action_loader=action_loader,
                                   cameras_to_retain=cameras_to_retain, CROP=True, verbose=verbose, debug=debug,
                                   nb_ind_actions=nb_ind_actions, nb_cameras=nb_cameras, use_kp_conf=use_kp_conf,
                                   use_bbox_conf=use_bbox_conf, changes_string=changes_string,
                                   epoch_render_mod=epoch_render_mod, cam_approach=cam_approach, mini_batch_size=mini_batch_size,
                                   nb_workers=nb_workers, baselines=use_base_line)
    time_optimizer.selection_scalar = time_scalar


    specific_action_index = None
    load_pose_params = False
    export_path = None

    time_optimizer.render_bbox = False
    time_optimizer.image_scale = 2
    time_optimizer.minibatch_size = 10
    render_default_col = not args.render_col

    all_view_render = False
    time_optimizer.render_bright = True


    time_optimizer.iterate_actions(render=render_video, worker=worker, export_path=export_path,
                                   path_to_mesh=path_to_mesh, load_prev_pose_params=load_pose_params,
                                   render_in_default_col=render_default_col, all_view_render=all_view_render,
                                   specific_action_index=specific_action_index)

if __name__ == "__main__":
    main()