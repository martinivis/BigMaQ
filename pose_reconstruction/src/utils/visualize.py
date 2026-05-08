import os.path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join
import numpy as np

plt.ioff()

class Visualizer():

    def __init__(self, num_views):

        self.num_views = num_views
        self.fig, self.axs = plt.subplots(num_views, 2, figsize=(10, num_views*5))
        # Plt show is not necessary
        #plt.ion()

    def plot_mask_and_pred(self, image_pred, gt_mask, cps, cgs, iter, frame, iter_per_frame):


        marker_size = 2
        if self.num_views == 1:
            # Plot the prediction

            self.axs[0].clear()
            self.axs[0].imshow(image_pred[0])
            self.axs[0].set_title('Prediction')

            cp = cps[0]#.cpu().detach().numpy()
            cg = cgs[0]  # .cpu().detach().numpy()

            self.axs[0].plot(cp[:, 0], cp[:, 1], 'go', markersize=marker_size, label="Prediction")
            self.axs[0].plot(cg[:, 0], cg[:, 1], 'ro', markersize=marker_size, label="Ground-truth")
            self.axs[0].legend()
            # Plot the second mask
            self.axs[1].clear()
            self.axs[1].imshow(gt_mask[0])
            self.axs[1].set_title('GT Mask')



            self.axs[1].plot(cg[:, 0], cg[:, 1], 'ro', markersize=marker_size, label="Ground-truth")
            self.axs[1].plot(cp[:, 0], cp[:, 1], 'bo', markersize=marker_size, label="Prediction")
            self.axs[1].legend()

        else:

            for j in range(self.num_views):
                self.axs[j, 0].clear()

                if image_pred[j].shape.__len__() == 4:
                    im = self.axs[j, 0].imshow(image_pred[j][0, :, :, :3]/ 255)
                else:
                    im = self.axs[j, 0].imshow(image_pred[j])
                self.axs[j, 0].set_title('Prediction')

                cp = cps[j]#.cpu().detach().numpy()
                cg = cgs[j]
                self.axs[j, 0].plot(cp[:, 0], cp[:, 1], 'go', markersize=marker_size, label="Prediction")
                self.axs[j, 0].plot(cg[:, 0], cg[:, 1], 'ro', markersize=marker_size, label="Ground-truth")
                self.axs[j, 0].legend()
                self.axs[j, 0].legend()
                #self.colorbar = self.fig.colorbar(im, orientation='vertical', ax=self.axs[j, 0])


                # Plot the second mask
                self.axs[j, 1].clear()
                self.axs[j, 1].imshow(gt_mask[j])
                self.axs[j, 1].set_title('GT Mask')

                #.cpu().detach().numpy()
                self.axs[j, 1].plot(cg[:, 0], cg[:, 1], 'ro', markersize=marker_size, label="Ground-truth")
                self.axs[j, 1].plot(cp[:, 0], cp[:, 1], 'bo', markersize=marker_size, label="Prediction")
                self.axs[j, 1].legend()



        if frame != 0:
            self.fig.suptitle(f"Iterations: {iter_per_frame} for Frame: {frame}")
        else:
            self.fig.suptitle(f"Iterations: {iter} for Frame: {frame}")
        # Display the plot
        self.fig.canvas.draw()

        plt.pause(0.1)


class LossMonitor:
    def __init__(self):



        self.silhouette_loss = []
        self.keypoint_loss = []
        self.scale_loss = []
        self.scale_jt_loss = []
        self.cage_loss = []
        self.joint_angle_loss = []
        self.prev_body_pose_loss = []
        self.prev_body_kp_3d_loss = []
        self.kin_loss = []
        self.vertex_norm_ls = []
        self.vertex_smooth_ls = []
        self.max_rot_loss_ls = []
        self.normals_ref_ls = []
        self.angle_consistency_loss_ls = []
        self.hull_ls = []
        self.edge_ls = []
        self.laplace_ls = []
        self.intersection_loss_ls = []

        
        
        self.alphas = {}
        self.loss_dict_iters = {}



        self.cur_lrs = []


        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        #plt.ion()  # Interactive mode on


    def init_loss_dict_iters(self):

        for alpha_key in self.alphas.keys():
            self.loss_dict_iters[alpha_key] = []


    def update_losses(self, alphas, loss_dict_over_cams):
        """
        Loss dict for the current iterations over cameras
        :param loss_dict_over_cams:
        :return:
        """

        self.alphas = alphas

        ### If there are no keys in the loss dict iter then init it...
        if self.loss_dict_iters.keys().__len__() == 0:
            self.init_loss_dict_iters()


        for alpha_key in self.alphas.keys():
            self.loss_dict_iters[alpha_key].append(np.mean(loss_dict_over_cams[alpha_key]))

        self.plot_losses()


    def update_lr(self, param_groups_opt):

        self.cur_lrs = []

        for param_group in param_groups_opt:
            self.cur_lrs.append(param_group["lr"])


    def plot_losses(self, disable_hard_constraints=True, disable_mesh_losses=True):
        self.ax.clear()  # Clear current axes

        last_to_show = 50


        for alpha_key, alpha_val in self.alphas.items():

            if alpha_val != 0:
                self.ax.plot(self.loss_dict_iters[alpha_key][-last_to_show:],
                             label=rf'{alpha_key} = {alpha_val}', linestyle='--')



        #self.ax.set_ylim(0, 20)
        self.ax.set_xlabel(f'(Last {last_to_show}) Iterations ')
        self.ax.set_ylabel('Loss')
        self.ax.set_title(f'Loss Progression with lrs: {self.cur_lrs}')
        self.ax.legend()

        plt.draw()  # Redraw the current figure
        plt.pause(0.001)  # Pause to update the figure


class VideoCompositor():
    def __init__(self, smoothed, time_included, render_path, cameras_used):


        self.time_included = time_included
        self.path_rendered_images = render_path
        self.cameras_used = cameras_used

        # Smoothed time trajectory
        self.smoothed = smoothed

        self.fps_iters = 10


    def get_frame_size(self, img_path):
        # Get the frame size of an image that has been stored on disk already
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        return (height, width)


    def create_videos_opencv(self, image_paths, image_base_path, video_output_path, fps):

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')  # or use 'XVID'

        video_fps = fps  # Frames per second
        frame_size = self.get_frame_size(join(image_base_path, image_paths[0]))  # Frame size, ensure all images are this size

        out = cv2.VideoWriter(video_output_path, fourcc, video_fps, frame_size)

        for img_path in image_paths:
            img = cv2.imread(join(image_base_path, img_path))
            if img is not None:
                img_resized = cv2.resize(img, frame_size)
                out.write(img_resized)
            else:
                print(f"Warning: could not read file {img_path}")

        # Release everything when job is finished
        out.release()

    def create_videos(self):
        # Create videos for the rendered images

        for cam_name in self.cameras_used:
            camera_path = join(self.path_rendered_images, f"{str(cam_name).zfill(2)}")

            iterations_path = join(camera_path, "as_iters")
            if not os.path.exists(iterations_path):
                raise ValueError("There were no iteration images saved!")

            # Get all posed images
            posed_image_names = [x for x in os.listdir(iterations_path) if 'Global' in x]
            bones_image_names = [x for x in os.listdir(iterations_path) if 'Posing' in x]
            mesh_image_names = [x for x in os.listdir(iterations_path) if 'Mesh' in x]

            # Sort the image names by iteration
            posed_image_names = sorted(posed_image_names)
            bones_image_names = sorted(bones_image_names)
            mesh_image_names = sorted(mesh_image_names)

            all_image_names = posed_image_names + bones_image_names + mesh_image_names

            output_video_path = join(self.path_rendered_images,
                                     f"pose_bone_mesh_as_iters_{cam_name}_{self.fps_iters}.mp4")
            self.create_videos_opencv(all_image_names, iterations_path, output_video_path, fps=self.fps_iters)


            if self.time_included:
                time_path = join(camera_path, f"time_{self.smoothed}")
                if not os.path.exists(time_path):
                    raise ValueError("There was not time course saved!")
                time_image_names = [x for x in os.listdir(time_path) if 'Time' in x]
                time_image_names = sorted(time_image_names)

                output_video_path = join(self.path_rendered_images,
                                         f"video_{cam_name}_Time_{self.time_included}_Smooth_{self.smoothed}.mp4")
                self.create_videos_opencv(time_image_names, time_path, output_video_path, fps=40)

