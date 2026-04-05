import numpy as np
import os
from os.path import join
import cv2

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

def return_nb_frames_video(path_to_video):
    # Open the video file
    video_capture = cv2.VideoCapture(path_to_video)
    # Check if the video file was successfully opened
    if video_capture.isOpened():
        # Get the total number of frames in the video
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(f"Total number of frames: {frame_count}")
    else:
        print("Error: Could not open video file.")
    # Release the video capture object
    video_capture.release()
    return frame_count

def get_additive_frames_from_file(path):
    # Retrieve the videos of the current action
    video_names = [x for x in os.listdir(path) if '.mp4' and 'CoreView' in x]

    additive_frames_cam = {}
    video_name_mapping = {}

    for video_name in video_names:
        ### Open the video
        nb_frames = return_nb_frames_video(join(path, video_name))
        ## Extract the name as well
        cam_view = video_name[video_name.rfind('_') + 1 : -4]
        additive_frames_cam[cam_view] = nb_frames
        video_name_mapping[cam_view] = video_name

    min_frames_action = int(min(additive_frames_cam.values()))

    return additive_frames_cam, min_frames_action


def longest_common_interval(overlap_counts: np.ndarray, nb_animals: int):
    """
    Given overlap_counts[t] = number of animals present at frame t,
    find the longest contiguous run where overlap_counts[t] == nb_animals.
    Returns (start_idx, end_idx), where start_idx is inclusive and
    end_idx is exclusive. If no such run exists, returns (None, None).
    """
    # Boolean mask: True where all animals are present
    present_all = (overlap_counts == nb_animals)

    best_len    = 0
    best_range  = (None, None)
    curr_start  = None

    for i, is_present in enumerate(present_all):
        if is_present:
            if curr_start is None:
                curr_start = i
        else:
            if curr_start is not None:
                curr_len = i - curr_start
                if curr_len > best_len:
                    best_len   = curr_len
                    best_range = (curr_start, i)
                curr_start = None
    # handle case where interval runs until the last frame
    if curr_start is not None:
        curr_len = len(present_all) - curr_start
        if curr_len > best_len:
            best_range = (curr_start, len(present_all))

    return best_range