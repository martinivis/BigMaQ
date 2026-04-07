from pose_reconstruction.src.Optimizers.BaseOpt import BaseOptimizer
from pose_reconstruction.src.utils.data_loader import ActionLoader
from pytorch3d.renderer import PointLights, AmbientLights
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, RasterizationSettings
from pose_reconstruction.src.utils.utils import check_create_path, smooth
from os.path import join
import json
import torch
from procrustes import orthogonal
from scipy.spatial.transform import Rotation as R, Slerp
import os
import time
from tqdm import tqdm
import numpy as np
import cv2

class SharedVertexOptimizer(BaseOptimizer):
    def __init__(self, device, mesh_model, action_loader: ActionLoader, cameras_to_retain, CROP, verbose, debug=False,
                 changes_string="", create_base_path=True, step_scheduler=True, template_stretch=False,
                 **kwargs):

        # Get rid of template stretch initial
        super().__init__(device=device, mesh_model=mesh_model, action_loader=action_loader,
                                          cameras_to_retain=cameras_to_retain, CROP=CROP, verbose=verbose,
                         step_scheduler=step_scheduler, template_stretch=template_stretch, **kwargs)

        self.debug = debug
        self.mod_iter_show = 10

        # Render Optimization
        self.epoch_render_mod = 100

        # For saving the plots more systematically
        self.changes_path = None
        self.changes_string=changes_string

        # Base path
        self.base_path = None

        if create_base_path:
            self.create_run_save_path()

        self.limit_lr_scheduler = 0.00003


    def valid_labels(self, pycams):
        """
        Check for all the cameras to retain in the loaded actions
        :return: Valid labeled keyframe or no.
        """
        for cam_key, cam_inds in pycams.items():

            if cam_key in self.cameras_to_retain:
                cam = cam_inds[self.cur_ind]

                if cam is None:
                    return False

        return True

    def generate_ind_lists(self):
        """
        Creates a dictionary for the individuals that can actually be iterated over
        :return: Dict: key indvidual, value list of action index and frame
        """
        # Depends on cameras to retain which lists can be made

        ind_iter_dict = {}

        for ind in self.action_loader.unique_individuals:

            self.cur_ind = ind

            actions_and_frames = []

            # Generate here the tuple
            for action_index in self.action_loader.entire_labeled_data.keys():

                self.action_loader.set_action_loader_by_index(action_index)

                if ind in self.action_loader.inds:

                    # Iterate the frames available
                    for frame, pycams in self.action_loader.pytorch_cameras.items():

                        # Pycams is a dictionary over viewpoints and then individuals
                        if self.valid_labels(pycams):
                            # Append
                            actions_and_frames.append((action_index, frame))



            ind_iter_dict[ind] = actions_and_frames

        return ind_iter_dict

    def create_run_save_path(self):

        # Create a base path
        base_render_path = join(self.base_path, "IndividualFits")
        check_create_path(base_render_path)

        # Create the render path
        render_path = join(base_render_path, self.changes_string)
        check_create_path(render_path)

        self.changes_path = render_path

        return render_path

    def create_render_path(self):

        render_path = self.create_run_save_path()

        # Create a new file path for the epochs
        #render_path_epoch = join(render_path, f"{self.epoch}")
        #check_create_path(render_path_epoch)
        self.render_path = render_path

    def render_parameters(self):

        #print(f"Rendering images after the stage")
        ### Plot the results of the fitting
        for action_frame_index, (action_index, frame) in enumerate(self.action_and_keyframes):
            self.action_loader.set_action_loader_by_index(action_index)
            self.action_frame_index = [action_frame_index]

            # Update the frame
            self.frame = frame
            self.bundled_frames = [frame]

            ## Pose the mesh
            self.__pose_mesh__()
            self.create_renderers()

            # Display the mesh
            self.display_mesh_on_images()



    def set_action_keyframe_dict_individual_fits(self, action_keyframe_dict_plus):


        action_and_keyframes = []

        for (_, action_index, frame) in action_keyframe_dict_plus.values():
            action_and_keyframes.append((action_index, frame))

        self.action_and_keyframes = action_and_keyframes



    def render_action(self, action_index_to_render):

        for action_frame_index, (action_index, frame) in enumerate(self.action_and_keyframes):

            if action_index != action_index_to_render:
                continue

            self.action_loader.set_action_loader_by_index(action_index)
            self.action_frame_index = action_frame_index

            self.frame = frame

            ## Pose the mesh
            self.__pose_mesh__()
            self.create_renderers()

            # Display the mesh
            self.display_mesh_on_images()


    def allign_by_procrustes(self, window=None):

        # Set/Reset to 0
        self.action_frame_index = [0]
        # Pose the mesh, to get the joints etc.
        self.__pose_mesh__()
        # 3d keypoints
        model_3d_keypoints_init_pos = self.posed_3d_keypoints.clone().detach().cpu().numpy()


        ### Iterate the keyframes of that action
        global_t = np.zeros(shape=(len(self.action_and_keyframes), 1, 3))

        init_R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        global_R = np.zeros(shape=(len(self.action_and_keyframes), 1, 3))

        for action_frame_index, (action_index, frame) in enumerate(self.action_and_keyframes):


            # If we are not in over time optimization set the action by index first
            if self.action_loader.labels_only:
                self.action_frame_index = action_frame_index
                self.action_loader.set_action_loader_by_index(action_index)

            # Get the keypoints that are non-zero
            kp_3d_frame_ind = self.action_loader.keypoints_3d[frame][self.cur_ind]
            non_zero_kp_indices = np.argwhere(kp_3d_frame_ind.sum(axis=-1) != 0).squeeze()
            valid_3d_kp_pos = kp_3d_frame_ind[non_zero_kp_indices, :]
            # Just take the first frame, it is anyway initialized all the same at the start
            model_3d_keypoints = model_3d_keypoints_init_pos[0, non_zero_kp_indices, :]
            # No keypoints? --> no procrustes
            if valid_3d_kp_pos.shape[0] == 0:
                # Skip empty entries
                continue
            # non-zero kp indices is a single number only? valid_3d_kp_pos, must be (nx3) if it is (3,) you need to enlarge it?
            # Ideally both should be at the same time reduced in the first dimension anyway, but better for debugging
            if valid_3d_kp_pos.ndim == 1:
                valid_3d_kp_pos = np.expand_dims(valid_3d_kp_pos, axis=0)
                model_3d_keypoints = np.expand_dims(model_3d_keypoints, axis=0)
                # If there is only a single keypoint
                continue
            # Get the shift
            shift_t = np.mean(valid_3d_kp_pos, axis=0) - np.mean(model_3d_keypoints, axis=0)
            shifted_model_keypoints = model_3d_keypoints + shift_t
            result = orthogonal(shifted_model_keypoints, valid_3d_kp_pos, scale=True, translate=True)
            transform = result.t
            # Add the shift to the old position
            global_t[action_frame_index, :, :] = shift_t + self.global_t[0, :, :].detach().cpu().numpy()#np.array([[0, 0, -0.75]]) #np.mean(valid_3d_kp_pos, axis=0).reshape((1, 3))
            global_R[action_frame_index, :, :] = R.from_matrix(transform.T).as_rotvec()


            # global_t[action_frame_index, :, :] = np.mean(valid_3d_kp_pos, axis=0).reshape((1, 3))

            # Do a procrustes for the remaining keypoints

            # Get the 3d points of the prediction

            # Get the 3D points for the template

            # Do the procrustes

        # TODO: Procrustes is not the problem itself if smoothed
        # todo: check that we are in time mode, check that action keyframes are long enough
        if window is not None:
            global_t = smooth(global_t, window=window)

            rots = R.from_rotvec(global_R.reshape(-1, 3))

            smoothed_rots = [rots[0]]  # initialize with first frame
            alpha = 0.2  # smoothing factor (0 = very smooth, 1 = no smoothing)

            for i in range(1, len(rots)):
                prev = smoothed_rots[-1]
                curr = rots[i]

                key_times = [0, 1]
                key_rots = R.concatenate([prev, curr])

                slerp = Slerp(key_times, key_rots)

                smoothed = slerp([alpha])[0]  # interpolate toward current
                smoothed_rots.append(smoothed)

            global_R = R.concatenate(smoothed_rots).as_rotvec().reshape(-1, 1, 3)


        self.global_t = torch.from_numpy(global_t).float().to(self.device).requires_grad_(True)
        self.global_orientation = torch.from_numpy(global_R).float().to(self.device).requires_grad_(True)


    def bundled_seq_opt(self):

        ### Generate placeholders for the parameters for that individual

        ### Create the flags optimizer etc
        self.update_learnable_flags_optimizer()


        iter_index = 0
        for epoch in tqdm(range(self.params_dict["epochs"])):

            epoch_loss = 0

            self.epoch = epoch


            # SGD like, parameters are shared but not batched over frames
            for action_frame_index, (action_index, frame) in enumerate(self.action_and_keyframes):

                self.action_loader.set_action_loader_by_index(action_index)
                self.action_frame_index = [action_frame_index]

                #self.action_index = action_index

                self.iter = iter_index
                # Update the frame
                self.frame = frame
                self.bundled_frames = [frame]

                ## Pose the mesh
                self.__pose_mesh__()
                self.create_renderers()
                self.__cam_loop__()

                # Update the parameters
                self.body_optimizer.zero_grad()
                self.loss.backward()
                self.check_gradients()
                self.body_optimizer.step()

                epoch_loss += self.loss.item()

                # Afterward gradient update
                iter_index += 1

            if self.epoch % self.epoch_render_mod == 0: #and self.epoch != 0:
                self.render_parameters()


            ### Adjust the lr scheduler
            if self.params_dict["Scheduler"]:
                # Only necessary if not multi-step
                if self.step_scheduler:
                    self.scheduler.step()
                else:
                    self.scheduler.step(epoch_loss)


            # If the learning rate is too small for all parameters --> break the loop
            if np.max(np.array([x["lr"] for x in self.body_optimizer.param_groups])) < self.limit_lr_scheduler and not self.step_scheduler:
                print("Exiting the stage because the lr was below threshold.")
                break


        # if self.stage != 'Color':
        #     # Get the colors from the images
        #     self.render_colormaps()

        # Render with pytorch cameras
        self.render_parameters()


        # Assign the values to be saved
        self.assign_model_paramaters_to_dict()


    def colormap_rasterizer(self, camera, im_h, im_w):
        raster_settings = RasterizationSettings(
            image_size=(im_h,
                        im_w),
            blur_radius=0.0,
            faces_per_pixel=1,  # K closest faces
        )

        rasterizer = MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        )

        return rasterizer
    def render_colormaps(self):
        """
                Following: https://github.com/facebookresearch/pytorch3d/issues/126
                :return:
                """

        vertex_colors_saved = {}

        with torch.no_grad():
            for action_frame_index, (action_index, frame) in enumerate(self.action_and_keyframes):
                self.action_loader.set_action_loader_by_index(action_index)
                self.action_frame_index = action_frame_index

                # Update the frame
                self.frame = frame

                ## Pose the mesh
                self.__pose_mesh__()
                self.create_renderers()



                # Iterate the cameras
                for cam_key, cam_inds in self.action_loader.pytorch_cameras[self.frame].items():

                    if cam_key not in self.cameras_to_retain:
                        continue

                    cam = cam_inds[self.cur_ind]
                    #big_cam = self.action_loader.big_cams.cam_dict[cam_key]
                    im_h, im_w = cam.image_size.squeeze()


                    # cam_loc = self.action_loader.big_cams.get_camera_in_world(cam_key).T
                    # light = PointLights(device=self.device, location=cam_loc)


                    # # This should already contain the correct light for that viewpoint that aligns with the camera origin
                    # mesh_in_image = self.display_renderer[cam_key](self.posed_mesh, cameras=cam, lights=light)
                    # mesh_img = mesh_in_image[0, ...].cpu().numpy().astype(dtype=np.uint8)


                    background = self.get_background_image_non_single_views(cam_key)


                    # Ensure both images are the same size, also because of max img size limits
                    background = cv2.resize(background, (int(im_w), int(im_h)))

                    ##
                    raster_camera = self.colormap_rasterizer(cam, int(im_h), int(im_w))
                    fragments = raster_camera(self.posed_mesh)
                    meshes = self.posed_mesh
                    # pix_to_face is of shape (N, H, W, 1)
                    pix_to_face = fragments.pix_to_face
                    # plt.figure()
                    # plt.imshow(pix_to_face[0, :, :, 0].clone().detach().cpu().numpy())
                    # plt.show()
                    # (F, 3) where F is the total number of faces across all the meshes in the batch
                    # 3 vertices are per face then, row indicates the face_idx, 3 are then the vertices
                    packed_faces = meshes.faces_packed()
                    pix_to_face_image = pix_to_face[0, :, :, 0]
                    unique_faces = pix_to_face_image.unique()

                    # Iterate over all faces that are in the rendering
                    for face_id in tqdm(unique_faces):  # 20948
                        spots = torch.where(pix_to_face_image == face_id)
                        for occurence_idx in range(spots[0].shape[0]):
                            x = spots[0][occurence_idx]
                            y = spots[1][occurence_idx]



                            # Retrieve the color of the image at that location
                            #color = self.rgbs[cam_key][x, y, :]
                            color = background[x, y, :]

                            # Iter associateed vertices for that face, and save the color
                            for vert_id in packed_faces[face_id]:
                                if vert_id in vertex_colors_saved:
                                    # Key exists, append the item to the existing list
                                    vertex_colors_saved[vert_id].append(color)
                                else:
                                    # Key does not exist, create a new list with the item
                                    vertex_colors_saved[vert_id] = [color]

        num_vertices = self.mesh_model.V.shape[1]

        # Initialize RGB array
        rgb_array = np.zeros((num_vertices, 3), dtype=np.float64)

        print("Average rgb colors for faces")
        # Iterate over the dictionary to compute the average colors
        for vertex_index, color_list in tqdm(vertex_colors_saved.items()):
            # Compute the average for each color channel
            if color_list:  # Check if there are colors to avoid division by zero
                averaged_color = np.mean(color_list, axis=0)  # .astype(int)
                rgb_array[vertex_index] = averaged_color


        # Transform rgb values so the ln is
        # +1 gets rid of the 0s
        # Larger max value of the 1s
        #rgb_array = (rgb_array + 1) / 257

        rgb_array = rgb_array / 255

        eps = 1e-6
        rgb_array = np.clip(rgb_array, eps, 1-eps)

        rgb_array = np.log(rgb_array/(1-rgb_array))

        # Put the new texture on the mesh

        self.vertex_colors = torch.from_numpy(rgb_array).to(torch.float32)[None].to(
            self.device)
        # vert rgb as (1, V, 3)
        self.mesh_model.vert_colors = torch.from_numpy(rgb_array).to(torch.float32)[None].to(
            self.device)


    def save_json_config(self):

        with open(join(self.changes_path, f"{self.stage}.json"), "w") as f:
            json.dump(self.params_dict, f)

    def sequence_optimization_mesh(self, species_path=None, sequence=["Bones", "Mesh", "Color"]):
        """
        Loads the parameter dictionaries for the respective stages and then iterates over them
        :var species_identifier: This is in species or species retargeting used
        :return:
        """

        #sequence = ["Global", "Pose1", "Bones", "Mesh", "Color"]

        #sequence = ["Bones", "Mesh", "Color"]
        #sequence = ["Mesh","Color"]

        ### Save the actions and keyframes
        actions_keyframes_dict = {}
        for iter_index, (action_index, frame) in enumerate(self.action_and_keyframes):
            actions_keyframes_dict[iter_index] = [self.action_loader.labeled_actions[action_index][0], action_index, frame]


        if species_path is not None:
            with open(join(species_path, f"{self.cur_ind}_labeled_actions_and_frames.json"),
                      'w') as f:
                json.dump(actions_keyframes_dict, f, indent=4)
        else:
            with open(join(r"../../data/Individuals", f"{self.cur_ind}_labeled_actions_and_frames.json"), 'w') as f:
                json.dump(actions_keyframes_dict, f, indent=4)


        for stage in sequence:

            #save_path_individual = join(r"../../data/Individuals", f"{self.cur_ind}_{stage}_{self.action_loader.high_poly}.pth")
            #save_path_individual =


            if species_path is not None:

                save_path_individual = join(species_path, f"{self.cur_ind}_{stage}.pth")

            else:
                if self.action_loader.high_poly:
                    save_path_individual = join(self.action_loader.hard_drive_loc,
                                                rf"Training/IndividualFitsParameters/{self.cur_ind}_Color.pth")
                else:
                    save_path_individual = join(self.action_loader.hard_drive_loc,
                                                rf"Training/IndividualFitsParameters/{self.cur_ind}_Color_{False}.pth")


            self.params_dict = self.read_json_cfg(stage)
            self.stage = stage
            # Save the current run config to a folder as backup
            self.save_json_config()

            ## Loads parameters if available and if stated so in param dict
            if self.params_dict["Load_Params"] and os.path.exists(save_path_individual):

                print(f"Loaded parameters of stage: {stage}")
                # Load them if possible
                self.model_parameters = torch.load(save_path_individual)
                self.assign_dict_to_model_parameters()

            else:
                self.update_opt_params()
                print(f"Starting optimization of stage: {stage}")
                self.bundled_seq_opt()

                #if self.stage == 'Mesh':
                # if self.stage == 'Mesh':
                #     self.render_colormaps()

                torch.save(self.model_parameters, save_path_individual)


    def iterate_individuals(self, reload=False):
        """
        Overarching function that loades labels from action loader, filters them and then iterates the individuals
        and their associated labels in the data.
        """


        self.action_loader.load_entire_labeled_frames(reload=reload)
        ind_iter_dict = self.generate_ind_lists()

        ### Load the pose parameters etc. from a certain individual for all the actions
        for ind in self.action_loader.unique_individuals:

            self.cur_ind = ind

            print("Working on individual: ", self.cur_ind)

            if self.debug and ind != 'Hyde':
                continue

            # if ind != 'Jekyll':
            #     continue

            #self.aggregate_parameters_per_individual(ind)

            ### Iterate then these actions in a loop etc.
            self.action_and_keyframes = ind_iter_dict[self.cur_ind]

            # Reset the action frame index (can be still set from the old individual)
            self.action_frame_index = 0

            ### Do this only once
            # Just put placeholders
            self.init_model_params()

            # Align by Procrustes
            self.allign_by_procrustes()

            ## iterate the different steps
            self.sequence_optimization_mesh()
