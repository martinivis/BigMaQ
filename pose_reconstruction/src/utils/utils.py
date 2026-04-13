import os
import torch
import numpy as np
import cv2
from os.path import join
import json
from scipy.spatial.transform import Rotation as R


def get_rot_vec_from_R(R_mat):
    """

    :param R: (3x3) np.array
    :return: r
    """
    rot = R.from_matrix(R_mat)
    return rot.as_rotvec()


def print_cuda_memory_stats(device=0, prefix=""):
    """Utility function to print memory stats for a given CUDA device."""
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    print(f"{prefix} memory_allocated: {allocated / (1024**2):.2f} MB")
    print(f"{prefix} memory_reserved : {reserved / (1024**2):.2f} MB")

def smooth(x, window=5):
    out = np.copy(x)
    for i in range(len(x)):
        start = max(0, i - window)
        end = min(len(x), i + window + 1)
        out[i] = np.mean(x[start:end], axis=0)
    return out

def convert_tensor_to_numpy(image):
    if isinstance(image, torch.Tensor):
        # Move to CPU and convert to NumPy
        return image.cpu().numpy()
    return image  # Return as is if it's not a PyTorch tensor



def split_into_sublists(original_list, num_sublists):
    # Calculate the approximate size of each sublist
    sublist_size = len(original_list) // num_sublists
    remainder = len(original_list) % num_sublists  # Handle leftover elements

    sublists = []
    start_index = 0

    for i in range(num_sublists):
        # Determine the end index for the current sublist
        # Distribute the remainder among the first `remainder` sublists
        end_index = start_index + sublist_size + (1 if i < remainder else 0)
        sublists.append(original_list[start_index:end_index])
        start_index = end_index  # Move to the next sublist

    return sublists

def pad_image_to_bbox(image, bbox):
    """
    Pads the image so that it accommodates the bounding box.

    Args:
        image (numpy.ndarray): The input image as a 3D array (H, W, C).
        bbox (numpy ndarray): (2x2) (UL, LR) (x, y) format
    Returns:
        numpy.ndarray: The padded image.
    """
    im_h, im_w, im_c = image.shape  # Image dimensions

    # Extract bounding box coordinates (x1, y1 = top-left, x2, y2 = bottom-right)
    x1, y1 = bbox[0]  # Top-left corner (UL)
    x2, y2 = bbox[1]  # Bottom-right corner (LR)

    # Calculate padding required
    pad_right = max(0, x2 - im_w)  # Excess width
    pad_bottom = max(0, y2 - im_h)  # Excess height

    # No padding required, return original image
    if pad_right == 0 and pad_bottom == 0:
        return image

    # Pad the image
    pad_top = 0  # Assuming no top padding needed for now
    pad_left = 0  # Assuming no left padding needed for now

    padded_image = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),  # Padding for H, W, C
        mode="constant",
        constant_values=0  # Pads with black (0)
    )

    return padded_image

def check_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_json_from_path(json_path):
    with open(json_path, 'r') as f:
        d = json.load(f)
    return d

def mean_without_outliers(points, distance_threshold):
    """
    Compute the mean of 3D points, ignoring any points
    whose distance from the initial mean is above 'distance_threshold'.

    Parameters
    ----------
    points : np.ndarray
        An Nx3 array of 3D points.
    distance_threshold : float
        Maximum allowed distance from the mean to be considered an inlier.

    Returns
    -------
    np.ndarray
        The mean of the filtered inlier points, shape (3,).
    """

    # 1. Compute the initial mean of all points
    centroid = np.mean(points, axis=0)  # shape (3,)

    # 2. Compute distances of each point from that mean
    diffs = points - centroid  # shape (N, 3)
    dists = np.linalg.norm(diffs, axis=1)  # shape (N,)

    # 3. Filter points based on the distance threshold
    mask = dists < distance_threshold
    inlier_points = points[mask]

    if inlier_points.size == 0:
        # Edge case: if all points are outliers
        print("Warning: No inliers found within the threshold.")
        return centroid  # or return None, depending on your use case

    # 4. Compute the mean of the inliers
    filtered_mean = np.mean(inlier_points, axis=0)
    return filtered_mean



def images_to_video(
    images_folder: str,
    output_filename: str = "output.mp4",
    fps: int = 40,
        ind=None
) -> None:
    """
    Converts a folder of images whose filenames start with a numeric frame index,
    followed by an underscore and any additional info (e.g. '1_whatever.png',
    '2_otherstuff.png'), into a video at the specified FPS.

    :param images_folder: Path to the folder containing images.
    :param output_filename: Name (and path) of the output video file.
    :param fps: Frames per second for the output video.
    """
    # Gather images that end with .png, .jpg, or .jpeg
    file_list = [
        f for f in os.listdir(images_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    #file_list = [f for f in file_list if ind in f]
    file_list = [
        f for f in file_list
        if f"_{ind}_" in f
    ]


    if not file_list:
        raise ValueError(f"No valid images found in folder: {images_folder}")

    # Function to extract the leading integer (frame number) before the first underscore
    def extract_frame_number(filename: str) -> int:
        """
        Extracts the leading integer from filenames like '10_someInfo.png'.
        This splits on the underscore, takes the first part (e.g. '10'),
        and converts it to an integer.
        """
        # Remove the extension
        base_name, _ = os.path.splitext(filename)
        # Split on the underscore
        parts = base_name.split("_", 1)
        # The first part should be the frame number in string form
        frame_str = parts[0]  # e.g. "10" from "10_someInfo"
        return int(frame_str)

    # Sort the file list by their integer frame number
    file_list.sort(key=extract_frame_number)

    # Read the first image to get video dimensions
    first_image_path = os.path.join(images_folder, file_list[0])
    first_frame = cv2.imread(first_image_path)
    height, width, _ = first_frame.shape

    # Define the video writer (MP4 with 'mp4v' codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Move up the camera folder
    video_path = os.path.abspath(os.path.join(images_folder, '..'))

    video_writer = cv2.VideoWriter(join(video_path, output_filename), fourcc, fps, (width, height))

    # Write each image in ascending numerical order to the video
    for image_file in file_list:
        image_path = os.path.join(images_folder, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()
    cv2.destroyAllWindows()

    #print(f"Video saved as {output_filename}")

def create_videos(render_path, cameras, ind):


    for cam in cameras:

        cam_path = join(render_path, str(cam).zfill(2))

        images_to_video(cam_path, f"{ind}_{cam}.mp4", ind=ind)

def find_second_underscore(s):
    # Find the position of the first underscore
    first = s.find('_')

    # If there is no underscore at all, return -1
    if first == -1:
        return -1

    # Find the position of the second underscore, starting the search after the first underscore
    second = s.find('_', first + 1)

    return second

def get_video_names(path):
    # Get the video files in the action path folder
    video_files = [x for x in os.listdir(path) if '.mp4' and 'CoreView' in x]
    underscore_pos = find_second_underscore(video_files[0])
    video_name = video_files[0][:underscore_pos + 1]
    return video_name

def extract_frames(video_path, frame_number):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Set the position of the next frame to be read
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame.")
        cap.release()
        return
    else:
        cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_frames_from_video(video_path, start_frame, num_following=30):
    """
    Reads the frame at 'start_frame' plus up to 'num_following' subsequent frames,
    if available.

    Returns:
        frames: a list of valid frames (up to num_following+1).
    """
    cap = cv2.VideoCapture(video_path)
    # Jump to the start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    # We want the start frame + 30 frames after => total of num_following + 1
    for _ in range(num_following + 1):
        ret, frame = cap.read()
        if not ret:  # No more frames to read
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def draw_inset_bounding_box(img, detected=False):
    """
    Draw a bounding box inset from the edges of an image.
    The bounding box is green if flip_condition=False, red if flip_condition=True.

    Args:
        img: the image
      flip_condition (bool): Toggles the bounding box color (False -> green, True -> red).
    """

    # Dimensions of the image
    h, w = img.shape[:2]

    # How far from each edge to inset the bounding box
    margin = 0
    top_left = (margin, margin)
    bottom_right = (w - margin, h - margin)

    # Choose color based on flip_condition
    # (OpenCV uses BGR format)
    if not detected:
        color = (255, 0, 0)  # Red in RGB
    else:
        color = (0, 255, 0)  # Green in RGB

    # Draw the rectangle on the image
    thickness = 6  # thickness of the bounding box lines
    cv2.rectangle(img, top_left, bottom_right, color, thickness)

    return img































