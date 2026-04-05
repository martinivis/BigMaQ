import copy
import pickle

import pandas as pd

from pose_reconstruction.src.Optimizers.VertexOpt import SharedVertexOptimizer
from pose_reconstruction.src.utils.data_loader import ActionLoader, longest_continuous_ones_indices, generate_overlapping_batches, split_into_minibatches
import numpy as np
import torch
from os.path import join
from tqdm import tqdm
import os
import json
import warnings
from pose_reconstruction.src.utils.utils import check_create_path
import pose_reconstruction.src.utils.utils as utils
import time
from pytorch3d.renderer.cameras import PerspectiveCameras
from pose_reconstruction.src.utils.LBS import MeshModel




class TimeOptimizer(SharedVertexOptimizer):
    def __init__(self, device, mesh_model, action_loader: ActionLoader, cameras_to_retain, CROP, verbose, debug,
                 nb_ind_actions, nb_cameras, use_kp_conf=True, use_bbox_conf=True, changes_string="", image_scale=4,
                 hand_weight=1, tail_weight=0, epoch_render_mod=50, cam_approach=0, mini_batch_size=1, nb_workers=1,
                 **kwargs):
        super().__init__(device=device, mesh_model=mesh_model, action_loader=action_loader,
                         cameras_to_retain=cameras_to_retain, CROP=CROP, verbose=verbose, debug=debug,
                         changes_string=changes_string, create_base_path=False, template_stretch=True,
                         **kwargs)

        self.nb_ind_actions = nb_ind_actions

        # Number of cameras for the action optimization
        self.nb_cameras = nb_cameras
        self.NUM_CAMS = nb_cameras
        self.cam_approach = cam_approach

        # Start frame
        self.start_frame = 0
        self.end_frame = -1

        # Batch size for rendering the mesh
        self.minibatch_size = mini_batch_size #40 is fine for 100 size
        #print(f"Mini batch size: {mini_batch_size}")

        # Number workers for parallelization (important for which actions the object iterates)
        self.nb_workers = nb_workers
        self.worker=0

        self.camera_selection_matrix = None


        # Individual save path
        self.ind_params_path = None
        # Action save path
        self.action_save_path = None


        # Det confidences
        self.use_kp_conf = use_kp_conf
        self.use_bbox_conf = use_bbox_conf

        # Epoch
        self.epoch = 0
        self.last_epochs = 0

        # for 2 cameras --> probably batch size 50 okey...


        # batch size and cameras is linear
        # image size probably square

        # limit = img_size**2 * batch_size * camera --> derive from this then the maximum batch size given the number of cameras


        # 40 batchsize limit with 2 cameras and 500 image size
        # 20 with 4 cameras and 500 works as well

        # with 16 cameras just 4 frames at a time? see how a reduction to 256x256 would work


        # for 4 cameras
        self.batch_size = 80


        # The overlap determines the breaks...
        self.overlap = 10 # to reduce a little bit errors?

        # List of sublist of frames to iterate, for rendering and optimization
        self.minibatches_to_iterate = []

        #self.mod_iter_show



        ## Plotting and debugging specific

        # Only the first frame of the sequence to show
        self.mod_iter_show = 10 # 100

        # Render Optimization
        self.epoch_render_mod = epoch_render_mod#50

        # Overwrite the faces per pixel, check if 20 faces per pixel works with 16 cameras and 256 and batch size 80, at least linear in faces per pixel

        self.faces_per_pixel = 20 # DIFFERENT TO VERTEX OPT AS WELL


        self.plot_skeleton = True

        # Make a check whether we can run the time optimization
        self.check_time_configuration()


        ####


        self.image_scale = image_scale




        #filtered_table = self.skip_table[self.skip_table["Locomotion"] != "skip"]
        #max_index = filtered_table["n_frames"].idxmax()
        #pass

        # Cameras that should be rendered after the action tracking is complete
        self.cameras_action_display = []

        ### Rendering
        # Video flag
        self.video = False
        # Default color rendering
        #self.render_default_col = False
        # Minibatch size to iterate

        # Eval
        self.FF=False
        self.SA=False


        self.grad_ok = True

    def check_time_configuration(self):
        """
        Checks whether it is safe to run this over the entire dataset in terms of CUDA memory usage w.r.t. to previous
        runs
        :return:
        """

        reference_values = [40, 2, 500, 500, 90]

        current_values = [self.batch_size, self.nb_cameras, self.action_loader.max_image_size,
                          self.action_loader.max_image_size, self.faces_per_pixel]


        ref_val = 1

        for index in range(reference_values.__len__()):


            ref_val *= current_values[index] / reference_values[index]


        if ref_val <= 1:
            #print(f"The values set are within testing bounds. : {ref_val}")
            pass
        else:
            warnings.warn(f"The set of parameters could lead to memory issues: {ref_val}")

    def sparse_mixed_camera_selection(self, export_path=None):
        """

        :param approach: (0) waterfall, (1) random selection per frame
        :return:
        """

        # This goes rather frame-wise and checks for available cameras until max number cameras is reached
        # If one camera is gone, try to substitute that camera until max cameras is reached

        # cur individual index
        cur_ind_idx = self.action_loader.individuals.index(self.cur_ind)

        camera_frame_matrix = self.action_loader.tracking_chk_sum[cur_ind_idx, :, :]

        num_frames, num_cameras = camera_frame_matrix.shape

        # Calculate the total detections per camera
        nb_dets = np.sum(camera_frame_matrix, axis=0)

        # Sort cameras by track length (descending order)
        sorted_cameras = np.argsort(nb_dets)[::-1]



        # Iterate the sorted cameras and check if the first frame is valid, if yes, this camera is the initial set
        # For consecutive frames, check if previous cameras are valid if not switch to a non-active camera with a large set

        # Selected cameras over frames
        camera_selection_matrix = np.zeros_like(camera_frame_matrix)


        if export_path is not None:
            # Make sure the approach is 0 continous for the comparison
            assert self.cam_approach == 0

        # Continous stream
        if self.cam_approach == 0:

            # Iterate the frames
            for frame in range(num_frames):

                if frame == 0:
                    # No previous frame available, just select the ones in order of sorted cameras until full
                    for sorted_cam in sorted_cameras:
                        if camera_frame_matrix[frame, sorted_cam]:
                            camera_selection_matrix[frame, sorted_cam] = 1

                        if camera_selection_matrix[frame, :].sum() == self.nb_cameras:
                            break

                else:
                    # Get previous active cameras
                    prev_act_cameras = np.argwhere(camera_selection_matrix[frame-1, :] == 1)

                    for prev_act_camera in prev_act_cameras:
                        if camera_frame_matrix[frame, prev_act_camera] == 1:
                            camera_selection_matrix[frame, prev_act_camera] = 1

                    # If not full set in the new frame, fill by others
                    if camera_selection_matrix[frame, :].sum() < self.nb_cameras:

                        # Break at the earliest incidence of change of cameras
                        # Fill with other cameras from sorted

                        # If exporting, then just break at the earliest incidence
                        if export_path is not None:
                            break
                        else:
                            pass

                        for sorted_cam in sorted_cameras:
                            if sorted_cam not in prev_act_cameras and camera_frame_matrix[frame, sorted_cam]:
                                camera_selection_matrix[frame, sorted_cam] = 1
                            if camera_selection_matrix[frame, :].sum() == self.nb_cameras:
                                break


        elif self.cam_approach:
            # Random selection of cameras, not a continuous stream

            for frame in range(num_frames):

                # Get the active cameras
                active_cameras = np.argwhere(camera_frame_matrix[frame, :] == 1).squeeze()

                n = min(self.nb_cameras, len(active_cameras))
                # Reproducible selections
                np.random.seed(frame)
                random_subset = np.random.choice(active_cameras, n, replace=False)

                camera_selection_matrix[frame, random_subset] = 1


        # The camera_selection_matrix contains selected cameras per frame,
        # but is not pruned to the min number of cameras, i.e. it contains max number cameras, but also less




        # Checks per frame if there are as many entries as nb_cameras
        full_set_vector = camera_selection_matrix.sum(axis=1)
        full_set_vector = (full_set_vector == self.nb_cameras).astype(int)

        if full_set_vector.sum() == 0:
            return False
        else:
            indices_series = longest_continuous_ones_indices(full_set_vector)

            self.start_frame = min(indices_series)
            self.end_frame = max(indices_series) + 1
            self.camera_selection_matrix = camera_selection_matrix

            return True

    def render_action(self, action_index_to_render):


        batch_range = 30


        for action_frame_index, (action_index, frame) in enumerate(self.action_and_keyframes):
            if action_index != action_index_to_render:
                continue

            self.action_frame_index = action_frame_index
            self.frame = frame

            if action_frame_index % batch_range == 0:
                self.action_loader.get_background_images_batched(frame, batch_range=batch_range)


            ## Pose the mesh
            self.__pose_mesh__()
            self.create_renderers()
            # Display the mesh
            self.display_mesh_on_images()

    def create_action_and_keyframes_per_action(self):

        action_index = self.action_loader.action_index

        actions_and_frames = []

        for frame in range(self.start_frame, self.end_frame):
            actions_and_frames.append((action_index, frame))

        self.action_and_keyframes = actions_and_frames

    def assign_dict_to_model_parameters(self):

        # Load only body parameters for that individual
        self.global_scale = self.model_parameters["s"]
        self.bone_length = self.model_parameters["bone_lengths"]
        self.vertex_offsets = self.model_parameters["vertex_offsets"]

        if not self.render_default_col:
            self.vertex_colors = self.model_parameters["rgb"]

        if not self.action_loader.load_undist_rgb:
            self.global_orientation = self.model_parameters["R"]
            self.global_t = self.model_parameters["t"]
            self.body_pose = self.model_parameters["body_pose"]

            self.start_frame = int(self.model_parameters["frame_sequence"][0].cpu().detach().numpy())
            self.end_frame = int(self.model_parameters["frame_sequence"][1].cpu().detach().numpy())

            self.camera_selection_matrix = self.model_parameters["camera_selection_matrix"].cpu().detach().numpy()

    def assign_partial_loaded_parameters(self):
        # Only load specific parameters from the disk into model_parameters not all
        model_parameters = torch.load(self.ind_params_path, map_location=self.device)


        self.model_parameters["s"] = model_parameters["s"]
        self.model_parameters["bone_lengths"] = model_parameters["bone_lengths"]

        self.model_parameters["vertex_offsets"] = model_parameters["vertex_offsets"]
        self.model_parameters["rgb"] = model_parameters["rgb"]


    def assign_time_fits(self):
        """
        Loads pose fits over time from (previous) optimization
        :return:
        """

        model_parameters = torch.load(self.action_save_path, map_location=self.device)

        self.model_parameters["R"] = model_parameters["R"]
        self.model_parameters["t"] = model_parameters["t"]
        self.model_parameters["body_pose"] = model_parameters["body_pose"]

        self.model_parameters["frame_sequence"] = model_parameters["frame_sequence"]
        self.model_parameters["camera_selection_matrix"] = model_parameters["camera_selection_matrix"]

        ### Call the assignment
        self.assign_dict_to_model_parameters()


    def create_render_path(self):


        base_path = self.action_loader.action_path

        # Create a base path
        base_render_path = join(base_path, "Optimization_Renderings")
        check_create_path(base_render_path)

        if self.action_loader.load_undist_rgb:
            # Create the render path
            render_path = join(base_render_path, self.changes_string)
            check_create_path(render_path)

            self.render_path_base = render_path

            # Create a new file path for the epochs
            render_path_epoch = join(render_path, f"{self.epoch}")
            check_create_path(render_path_epoch)
            self.render_path = render_path_epoch

        else:
            # Create the render path
            render_path = join(base_render_path, "Videos_last")
            check_create_path(render_path)

            self.render_path_base = render_path

            # Create a new file path for the change string
            render_path_epoch = join(render_path, self.changes_string)
            check_create_path(render_path_epoch)
            self.render_path = render_path_epoch

    def load_individual_fits_parameters(self):
        # Init the model parameters
        self.init_model_params()
        self.allign_by_procrustes()
        # Load parts of the values from previous fits for that particular individual
        self.assign_partial_loaded_parameters()
        self.assign_dict_to_model_parameters()

    def set_active_cameras_per_frames(self):

        self.cameras_to_retain = []
        for frame in self.bundled_frames:

            if self.video:
                cameras_to_retain = self.cameras_action_display
            else:
                # Get the columns which are set, and add 1 to them
                cameras_to_retain = np.argwhere(self.camera_selection_matrix[frame, :] == 1).squeeze() + 1
                cameras_to_retain = [str(x) for x in cameras_to_retain]

            self.cameras_to_retain.append(cameras_to_retain)


    def render_parameters(self):

        self.flattened_idx = 0

        for batch_idx, batch in enumerate(self.minibatches_to_iterate):
            for minibatch_idx, frames in enumerate(batch):

                self.action_frame_index = [x - self.start_frame for x in frames]
                self.bundled_frames = frames
                self.set_active_cameras_per_frames()

                if self.video:
                    self.action_loader.get_background_images_batched(frames[0],
                                                                     len(frames) - 1)

                # Pose the mesh for the batch
                self.__pose_mesh__()

                # Display the mesh
                self.display_mesh_on_images()

                self.flattened_idx += 1

    def render_parameters_action(self):

        # Similiar to render parameters
        # But before the cameras need to be updated to the new cameras
        # The active set of cameras needs to be changed as well to the ones specified, done

        batches_to_iterate = generate_overlapping_batches(self.start_frame, self.end_frame, self.batch_size,
                                                          self.overlap)

        self.minibatches_to_iterate = split_into_minibatches(batches_to_iterate, self.minibatch_size) ##changed from 20 to self.minibatch_size

        self.create_batched_cameras_and_gt_masks()
        self.render_parameters()



    def create_batched_cameras_and_gt_masks(self):

        gt_masks_list = []
        batched_cameras_list = []

        for minibatch_idx, batch in enumerate(self.minibatches_to_iterate):
            for frames in batch:

                self.bundled_frames = frames
                self.set_active_cameras_per_frames()
                self.create_renderers()

                if self.action_loader.same_size:
                    # Put the cameras all in one render
                    Rs = []
                    Ts = []
                    focals = []
                    img_sizes = []
                    princ_points = []

                    gt_masks = []

                    for frame_index, frame in enumerate(frames):

                        for cam_key in self.cameras_to_retain[frame_index]:
                            frame_cam_key = f"{frame}_{cam_key}"
                            camera = self.cameras[frame_cam_key]

                            Rs.append(camera.R)
                            Ts.append(camera.T)
                            focals.append(camera.focal_length)
                            img_sizes.append(camera.image_size)
                            princ_points.append(camera.principal_point)

                            # Do not retrieve gt_masks if we want to render videos
                            if not self.video:
                                gt_masks.append(self.action_loader.masks[frame][cam_key][self.cur_ind][None])

                    Rs = torch.cat(Rs, dim=0)
                    Ts = torch.cat(Ts, dim=0)
                    focals = torch.cat(focals, dim=0)
                    img_sizes = torch.cat(img_sizes, dim=0)
                    princ_points = torch.cat(princ_points, dim=0)

                    # Same as above
                    if not self.video:
                        gt_masks = torch.cat(gt_masks, dim=0)

                    batched_cameras = PerspectiveCameras(R=Rs, T=Ts, focal_length=focals, principal_point=princ_points,
                                                         image_size=img_sizes, device=self.device)

                    gt_masks_list.append(gt_masks)
                    batched_cameras_list.append(batched_cameras)

        self.gt_masks_list = gt_masks_list
        self.batched_cameras = batched_cameras_list

    def bundled_seq_opt(self):

        ### Create the flags etc.
        self.update_learnable_flags_optimizer()

        # Iterate gradient updates
        iter_index = 0

        # Change the overlap a bit to obscure the step in updates.
        batches_to_iterate = generate_overlapping_batches(self.start_frame, self.end_frame, self.batch_size,
                                                          self.overlap)

        self.minibatches_to_iterate = split_into_minibatches(batches_to_iterate, self.minibatch_size)
        self.create_batched_cameras_and_gt_masks()


        #print(f"Sequence will have frames from {self.start_frame} to {self.end_frame}")

        tqdm_epochs = tqdm(range(self.params_dict["epochs"]), desc=f"Worker: {self.worker}, "
                                                                   f"Action Subindex: ({self.action_iterate_index}, "
                                                                   f"{self.end_frame-self.start_frame}),"
                                                                   f"Epochs")




        #with profile(use_cuda=True, with_stack=True) as prof:
        #print(self.ANGLE_WEIGHTING)



        # Epoch means how many times over the frames of the action
        for epoch in tqdm_epochs:
            epoch_loss = 0
            self.epoch = epoch + self.last_epochs
            self.flattened_idx = 0

            #rem = epoch % 10
            #if rem == 0:




            for batch_idx, batch in enumerate(self.minibatches_to_iterate):
                self.iter = iter_index
                self.batch_loss = 0
                self.batch_idx = batch_idx
                for minibatch_idx, frames in enumerate(batch):
                    # Update the frame
                    self.bundled_frames = frames

                    # Change the cameras to retain for the respective frames
                    self.set_active_cameras_per_frames()
                    self.action_frame_index = [x - self.start_frame for x in frames]
                    # Update action index, which is here just the frame itself as index
                    #self.action_frame_index = self.frame - self.start_frame
                    self.__pose_mesh__()


                    # This is the most expensive step time-wise
                    self.__cam_loop__()
                    self.batch_loss += self.loss
                    self.flattened_idx += 1

                if self.nb_workers == 1 and self.worker == -1:
                    utils.print_cuda_memory_stats(device=self.device)
                # Update the parameters
                self.body_optimizer.zero_grad()
                self.batch_loss.backward()
                ret = self.check_gradients()
                if not ret:
                    self.grad_ok = ret
                self.body_optimizer.step()
                epoch_loss += self.batch_loss.item()
                # Afterward gradient update
                iter_index += 1
            if self.epoch % self.epoch_render_mod == 0:
                self.render_parameters()
            lr_s_cur = []
            for param in self.body_optimizer.param_groups:
                lr_s_cur.append(param["lr"])
            #print(f"Epoch loss: {epoch_loss}, lrs: {lr_s_cur}")
            # Update tqdm with loss and learning rates
            tqdm_epochs.set_postfix({
               "loss": f"{epoch_loss:.4f}",
               **{f"lr{i}": f"{lr:.6f}" for i, lr in enumerate(lr_s_cur)}
            })
            ### Adjust the lr scheduler
            if self.params_dict["Scheduler"]:
                # Only necessary if not multi-step
                if self.step_scheduler:
                    self.scheduler.step()
                else:
                    self.scheduler.step(epoch_loss)
            # If the learning rate is too small break the loop
            if self.body_optimizer.param_groups[0]["lr"] < self.limit_lr_scheduler:
                print("Exiting the stage because the lr was below threshold.")
                break

        #print("Profiling results incoming")
        # Print profiling results
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        #prof.export_chrome_trace("trace.json")

        # Render with pytorch cameras
        #self.render_parameters()

        # Assign the values to be saved
        self.assign_model_paramaters_to_dict()


        # Assign epochs to last epochs
        self.last_epochs += self.epoch



    def update_angle_weighting(self):


        HAND_ROT_WEIGHT = self.params_dict["hand_weight_joint"]  # 1 / 4  #/ 4# 8
        TAIL_ROT_WEIGHT = self.params_dict["tail_weight_joint"]  # 1 / 4 #/ 4 # Tail not as loose as before
        # HAND_ROT_WEIGHT = TAIL_ROT_WEIGHT

        self.ANGLE_WEIGHTING = torch.tensor([1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, HAND_ROT_WEIGHT, 1,
                                             1, 1, HAND_ROT_WEIGHT, 1, 1, HAND_ROT_WEIGHT,
                                             TAIL_ROT_WEIGHT, TAIL_ROT_WEIGHT, TAIL_ROT_WEIGHT, TAIL_ROT_WEIGHT,
                                             TAIL_ROT_WEIGHT, TAIL_ROT_WEIGHT,
                                             1, 1, HAND_ROT_WEIGHT], device=self.device)


    def optimization_arch(self):



        # Create the action keyframes to iterate
        self.create_action_and_keyframes_per_action()

        # Init the model parameters,and overwrite by previous fits
        self.load_individual_fits_parameters()

        self.last_epochs = 0

        #stages = ["Time", "Time2"]
        stages = ["Time"]

        for stage in stages:

            self.stage = stage
            # Load the parameter dictionary
            self.params_dict = self.read_json_cfg(stage)

            self.update_angle_weighting()

            # Update the opt params
            self.update_opt_params()

            self.create_render_path()
            self.save_configurations_used()

            # Optimize the sequence
            # Start frame, end frame, cameras are in camera selection matrix
            self.bundled_seq_opt()


    def save_configurations_used(self):

        # Save the parameters used

        param_dict_to_save = copy.deepcopy(self.params_dict)
        #param_dict_to_save["Angle_weighting"] = {'hand': self.hand_weight, 'tail':self.tail_weight}


        # Save the configuration at the render path
        with open(join(self.render_path, f"{self.stage}.json"), "w") as f:
            json.dump(param_dict_to_save, f)


    def retrieve_skip(self, action_index):
        # Retrieves information about whether to skip an action or not
        if self.skip_table.loc[action_index]["Locomotion"] in ['skip', 'denied']:
            return True
        if self.skip_table.loc[action_index]["SocialAndOthers"] in ['skip', 'denied']:
            return True


        return False


    def generate_actions_to_iterate(self, worker):
        """
        Generates a list of action indices to iterate over for a given worker. The
        generation process determines viable action indices by checking a skip table,
        excluding actions marked for skipping, and then splits the viable indices
        into sublists based on the number of workers. Depending on the number of
        workers, it either returns all indices or only the indices corresponding to
        the given worker.

        :param worker: The identifier of the current worker for which the action
            indices are generated.
        :type worker: int
        :return: A list of action indices assigned for the worker to iterate over.
        :rtype: list
        """


        viable_action_indices = []
        for action_index in self.skip_table.index:
            # Skip if that action should not be tracked
            if self.retrieve_skip(action_index):
                continue
            else:
                viable_action_indices.append(action_index)

        # if nb workers is 1, then ignore worker id
        # else take the sublist of worker id
        viable_action_indices_sublisted = utils.split_into_sublists(viable_action_indices, num_sublists=self.nb_workers)

        if self.nb_workers == 1:
            action_indices_to_iterate = viable_action_indices_sublisted[0]
        else:
            action_indices_to_iterate = viable_action_indices_sublisted[worker]

        return action_indices_to_iterate


    def split_data_in_two(self, push_to_worker_2=200):

        self.skip_table["FramesIndProduct"] = self.skip_table['n_individuals'] * self.skip_table['n_frames']
        self.skip_table_sorted = self.skip_table.sort_values(by="FramesIndProduct", ascending=True)

        viable_action_indices = []
        for action_index in self.skip_table_sorted.index:
            # Skip if that action should not be tracked
            if self.retrieve_skip(action_index):
                continue
            else:
                viable_action_indices.append(action_index)

        self.skip_table_sorted = self.skip_table_sorted.loc[viable_action_indices]


        ### Get where the half of the dataframe is at

        # Calculate the total sum of the column
        total_sum = self.skip_table_sorted['FramesIndProduct'].sum()
        half_sum = total_sum / 2

        # Compute the cumulative sum of the column
        cumulative_sum = self.skip_table_sorted['FramesIndProduct'].cumsum()

        # Find the index where cumulative sum is about half of the total sum
        row_index = cumulative_sum[cumulative_sum >= half_sum].index[0]

        # Where the mid action lies
        split_index = viable_action_indices.index(row_index)

        # Put some more actions into the second worker

        assert self.worker in [0, 1]


        split_index = split_index - push_to_worker_2

        if self.worker == 0:
            viable_action_indices = viable_action_indices[:split_index]
        else:
            viable_action_indices = viable_action_indices[split_index:]

        return viable_action_indices

    def iterate_actions(self, render=True, worker=-1, specific_action_index=None, export_path=None, path_to_mesh=None,
                        all_view_render=False, render_in_default_col = False, load_prev_pose_params=False, debug=True,
                        indep_cam=None):


        # Iterate all the actions of the dataset by individual



        action_indices_to_iterate = [0]

        for action_iterate_index, action_index in enumerate(action_indices_to_iterate):


            if specific_action_index is not None:
                if action_index != specific_action_index:
                    continue
            # # # an action that gets a nan
            # if action_index != 141:
            #    continue

            # Last indices being processed...
            #if action_index < 268:
            #    continue


            self.grad_ok = True

            self.action_iterate_index = action_iterate_index
            # Load the tracking data of that action

            nb_inds = 1
            if self.SA and nb_inds!=1:
                continue

            self.action_loader.load_action_into_entire_dict(action_index)

            print("Loading action: ", self.action_loader.action_path)



            for ind_index, ind in enumerate(self.action_loader.unique_individuals):
                self.cur_ind = ind


                if self.action_loader.high_poly:
                    self.ind_params_path = join(self.action_loader.hard_drive_loc,
                                                rf"IndividualFitsParameters/{self.cur_ind}_Color.pth")
                else:
                    self.ind_params_path = join(self.action_loader.hard_drive_loc,
                                                rf"IndividualFitsParameters/{self.cur_ind}_Color_{False}.pth")

                # Move past individuals that are not part of the action
                if not self.cur_ind in self.action_loader.individuals:
                    continue

                # Define the path to save the optimized action
                self.action_save_path = join(self.action_loader.action_path, f"action_params_{self.cur_ind}_"
                                                                             f"{str(self.nb_cameras).zfill(2)}.pth")



                # Generate a flow of cameras to use in optimization
                ret = self.sparse_mixed_camera_selection(export_path)


                if not ret:
                    print("No frame satisfies all cameras")
                    continue
                print(f"Optimizing action: ({action_index}, {self.action_loader.action_path})")


                # Change the render color
                self.render_default_col = render_in_default_col


                if not load_prev_pose_params:

                    # Run the optimization with preloaded labels that are initialized
                    self.optimization_arch()
                    # Create frame sequence, otherwise start and end of the tracked sequence are not clear
                    self.model_parameters["frame_sequence"] = torch.Tensor([self.start_frame, self.end_frame]).to(
                        self.device)
                    self.model_parameters["camera_selection_matrix"] = torch.from_numpy(self.camera_selection_matrix)

                    # Save the pose data for that action
                    torch.save(self.model_parameters, self.action_save_path)
                    # For hyper-optimization save the parameters for the configuration as well.
                    torch.save(self.model_parameters, join(self.render_path, f"action_params_{self.cur_ind}_"
                                                                         f"{str(self.nb_cameras).zfill(2)}.pth"))
                else:


                    # Ind params path is set already

                    # Create the action keyframes to iterate
                    self.create_action_and_keyframes_per_action()

                    # Init the model parameters,and overwrite by previous fits
                    self.load_individual_fits_parameters()

                    self.action_loader.load_undist_rgb = False

                    # Load the parameters usually saved into the model, body pose etc.
                    self.assign_time_fits()


                if render:
                    print("Render the fitted sequence!")
                    self.action_loader.load_undist_rgb = False
                    self.action_loader.load_scaled_cameras(action_index)

                    if all_view_render:
                        self.cameras_action_display = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",
                                                      "14", "15", "16"]
                    else:
                        self.cameras_action_display = ["5", "14"]


                    self.video = True

                    self.render_parameters_action()
                    #self.render_action(action_index_to_render=action_index)
                    utils.create_videos(self.render_path, cameras=self.cameras_action_display, ind=ind)

                    # Reset the action iteration again
                    self.video = False
                    self.action_loader.load_undist_rgb = True

                    # Reload the action data
                    # Load the tracking data of that action
                    self.action_loader.load_action_into_entire_dict(action_index)



    def export_mesh(self, export_path, path_to_mesh, root=True):
        """


        :param export_path:
        :param path_to_mesh:
        :param root: Includes a root joint, if not included then the mesh in MAMMAL has a weird inclination at the belly
        :return:
        """


        txt_path = join(export_path, "monkey_model/monkey_txt")
        txt_path_2 = join(export_path, "monkey_model")



        mesh_model_unshifted = MeshModel(mesh=path_to_mesh, device=self.device, shift_initially=False)


        # Vertices txt
        vertices = mesh_model_unshifted.V.squeeze()
        np.savetxt(join(txt_path, "vertices.txt"), vertices.cpu())

        # Faces txt, (nb_faces, 3) where the 3 is triangular mesh with ids
        faces = mesh_model_unshifted.faces
        np.savetxt(join(txt_path, "faces_vert.txt"), faces.cpu(), fmt='%i')


        # id_to_names, is simply all the joints that exist as a list, and root (add root just as (0, 0, 0) if that is where the hip resides.)
        id_to_names = []

        if root:
            id_to_names.append('root')

        for bone in mesh_model_unshifted.dd['bones']:
            id_to_names.append(bone['name'])

        with open(join(txt_path, "id_to_names.pkl"), 'wb') as file:
            pickle.dump(id_to_names, file)


        if root:
            # Names to ID, is a dictionary where name is key, and value is index in the list  add +1 because of root
            names_to_id = {'root': 0}

            for joint_name, index_in_list in mesh_model_unshifted.dd['joint_to_index'].items():
                names_to_id[joint_name] = index_in_list + 1
        else:
            names_to_id = {}

            for joint_name, index_in_list in mesh_model_unshifted.dd['joint_to_index'].items():
                names_to_id[joint_name] = index_in_list

        with open(join(txt_path, "names_to_id.pkl"), 'wb') as file:
            pickle.dump(names_to_id, file)


        if root:
            # Parents gives the parents back in the list, insert the root at 0 and add +1 for the rest
            parents = (mesh_model_unshifted.parents.cpu() + 1).tolist()
            parents.insert(0, -1)  # add the root
        else:
            parents = (mesh_model_unshifted.parents.cpu()).tolist()

        with open(join(txt_path, "parents.pkl"), 'wb') as file:
            pickle.dump(parents, file)


        # Joint shifts, a list of (3,) joint positions,
        init_joint_trans = []

        if root:
            init_joint_trans.append(np.zeros(3))

        for joint_pos in mesh_model_unshifted.dd['J']:
            init_joint_trans.append(np.array(joint_pos))


        # List of initial joint rotations as 3,3 matrix, (currently not aware of any joint rotations that need to be applied as it is already the t-pose
        init_joint_rot_mat = []
        init_joint_rotvec = []

        rot_mats = mesh_model_unshifted.dd['joint_rot_mats']

        for i in range(len(names_to_id)):

            if root:
                if i==0:
                    rot_mat = np.array(rot_mats[0])
                    rot_mat = np.eye(3) # just put no transform at all.
                else:
                    rot_mat = np.array(rot_mats[i-1])
            else:
                rot_mat = np.array(rot_mats[i])

            rot_vec = utils.get_rot_vec_from_R(rot_mat)

            init_joint_rot_mat.append(rot_mat)
            init_joint_rotvec.append(rot_vec)

        with open(join(txt_path, "init_joint_rot_mat.pkl"), 'wb') as file:
            pickle.dump(init_joint_rot_mat, file)

        with open(join(txt_path, "init_joint_rotvec.pkl"), 'wb') as file:
            pickle.dump(init_joint_rotvec, file)


        ### Local joint shifts ---------------------


        #local_shifts = self.compute_local_translations_list(init_joint_trans, init_joint_rot_mat, parents)

        local_shifts = []
        if root:
            local_shifts.append(np.zeros(3))

        for local_shift in mesh_model_unshifted.dd["joint_trans_loc"]:
            local_shifts.append(np.array(local_shift))

        with open(join(txt_path, "init_joint_trans.pkl"), 'wb') as file:
            pickle.dump(local_shifts, file)

        #### ---------------------


        ## Skinning weights but weirdly mapped, (N_W, 3), first dim: joint id, 2nd dim: vertex id, 3rd the weight
        skinning_weights = []

        for vertex_deform in mesh_model_unshifted.dd['deform_weights']:
            vertex_index = vertex_deform['vertex_index']
            joint_influences = vertex_deform['weights']

            for joint, weight in joint_influences.items():
                joint_id = names_to_id[joint]
                # Append to skinning weights
                np_ar = np.array([joint_id, vertex_index, weight])
                skinning_weights.append(np_ar)

        # Ascend across the joints
        skinning_weights = np.asarray(skinning_weights)

        # Sort by the first column
        skinning_weights = skinning_weights[skinning_weights[:, 0].argsort()]
        np.savetxt(join(txt_path, "skinning_weights.txt"), skinning_weights)


        ## Indicate which bones to scale and which to scale in the same way.
        bone_length_mapper = -1 * np.ones(len(id_to_names))

        # There is no 0, so starting with 1?, for each new scale start with a new number #
        # Go over the joints, if the joint is in joints to scale give it a new scale, if its in hands or feet give it a unit scalar
        # Currently the hands and feet are not scaled in the monkey...
        end_effector_bone_id = {"LeftHand" : -1, "RightHand": -1, "LeftFoot": -1, "RightFoot": -1}

        bone_length_id = 1
        for bone_length_to_opt in self.BONE_LENGTHS_TO_OPT:

            # Mapping end-effecotrs to the same scale
            for end_eff, end_eff_bone_id in end_effector_bone_id.items():
                # If it is a part of the hierarchy but has a longer name
                if end_eff in bone_length_to_opt and len(bone_length_to_opt) > len(end_eff):
                    print("Longer Name")
                    # Check if the bone id for that effector is already set,
                    if end_eff_bone_id > 0:
                        bone_length_mapper[names_to_id[bone_length_to_opt]] = end_eff_bone_id
                    else:
                        # Set a new id for the next round
                        bone_length_mapper[names_to_id[bone_length_to_opt]] = bone_length_id
                        end_effector_bone_id[end_eff] = bone_length_id
                        bone_length_id += 1

                    # Do not map again
                    continue
            bone_length_mapper[names_to_id[bone_length_to_opt]] = bone_length_id
            bone_length_id += 1

        #bone_length_mapper[3:] = -1
        np.savetxt(join(txt_path, "bone_length_mapper.txt"), bone_length_mapper, fmt='%i')


        ## Create the reduced faces which is simply in our case all faces
        faces = mesh_model_unshifted.faces
        np.savetxt(join(txt_path, "reduced_face_7200.txt"), faces.cpu(), fmt='%i')

        # reduced ids would be the vertices to take only (N, ) array
        vert_ids = np.arange(0, vertices.shape[0], 1)
        np.savetxt(join(txt_path, "reduced_ids_7200.txt"), vert_ids, fmt='%i')


        ## Reg weights are the weighted angle penalties
        # Wherever there is a Hand inside a high penalty from normal pose otherwise put a 1
        # ndarray (N_ids, ) (including root, but that has 0)
        reg_weights = np.ones(len(id_to_names))

        if root:
            reg_weights[0] = 0

        for id_index, id_name in enumerate(id_to_names):

            for end_eff in end_effector_bone_id.keys():

                if end_eff in id_name:
                    if len(id_name) > len(end_eff):
                        reg_weights[id_index] = 6
                        #print(id_name)
                    else:
                        reg_weights[id_index] = 0.2
                        #print("End effector: ", id_name)

        np.savetxt(join(txt_path_2, "reg_weights.txt"), reg_weights)


        ## The mapper, only goes so far as the number of keypoints that are given
        mapper_list = []

        for i in range(self.action_loader.nb_markers):
            # do the mapping

            # ids is the joint or vertex ids
            # keypoint is the keypoint index
            # type is the way of mapping it to the mesh only J now

            corresponding_joint = self.HIERARCHY_MAPPING[i]
            corresponding_id = names_to_id[corresponding_joint]

            # The corresponding_id needs to be a list!!!
            # Also for joints, even though it is sometimes used for vertices

            map_dict = {'ids': [corresponding_id], 'keypoint': i, 'type': 'J'}
            mapper_list.append(map_dict)

        mapper_dict = {'mapper': mapper_list}

        with open(join(txt_path_2, "keypoint22_mapper.json"), 'w') as f:
            json.dump(mapper_dict, f, indent=2)

        print("Done exporting")







