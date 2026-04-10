import numpy as np
from itertools import chain, combinations
import cv2
def get_camera_vertices(dim=2, asp=2464/2056):
    vertices_camera = np.array([[-1, -2, -1],
                                [1, -2, -1],
                                [1, -2, 1],
                                [-1, -2, 1],
                                [-1, -2, -1],
                                [-1, -1, -1],
                                [1, -1, -1],
                                [1, -1, 1],
                                [-1, -1, 1],
                                [-1, -1, -1],
                                [1, -1, -1],
                                [1, -2, -1],
                                [1, -2, 1],
                                [1, -1, 1],
                                [-1, -1, 1],
                                [-1, -2, 1],
                                [-1, -1, 1],
                                [-asp * dim, 0, dim],
                                [-asp * dim, 0, -dim],
                                [asp * dim, 0, -dim],
                                [asp * dim, 0, dim],
                                [-asp * dim, 0, dim],
                                [-asp * dim, 0, dim],
                                [-1, -1, 1],
                                [1, -1, 1],
                                [asp * dim, 0, dim],
                                [asp * dim, 0, -dim],
                                [1, -1, -1],
                                [-1, -1, -1],
                                [-asp * dim, 0, -dim]])

    # Rotate vertices by 90 degrees around x so that camera back is in -z
    z_rot = np.array([[1, 0, 0],
                      [0, 0, -1],
                      [0, 1, 0]])

    # changing the orientation
    vertices_camera = (z_rot @ vertices_camera.T).T

    return vertices_camera



def powerset(iterable, min_nb_triangulation = 2, max_set = 0):

    s = list(iterable)

    if not max_set:
        max_set = s.__len__()

    list_subsets = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
    # Convert tuples to lists
    list_subsets = [list(x) for x in list_subsets]
    # Reduce list by min points for triangulation
    list_subsets = [x for x in list_subsets if max_set >= x.__len__() >= min_nb_triangulation]

    return list_subsets

def triangulate_simple(points, camera_mats):
    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0]
        A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d

def rep_error(p1, p2):
    """ Reprojection error of one 'marker'
    :param p1: 1x2 np array
    :param p2: 1x2 np array
    :return:
    """
    return np.linalg.norm(p1-p2)


def return_nb_frames_video(path_to_video):

    # Open the video file
    video_capture = cv2.VideoCapture(path_to_video)

    # Check if the video file was successfully opened
    if video_capture.isOpened():
        # Get the total number of frames in the video
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        #print(f"Total number of frames: {frame_count}")
    else:
        print("Error: Could not open video file.")

    # Release the video capture object
    video_capture.release()

    return frame_count
