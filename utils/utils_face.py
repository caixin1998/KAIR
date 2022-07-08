
import random
import cv2
import math
import numpy as np
 
def get_components_bbox(lm):
    item_dict = {}
    map_left_eye = list(range(36, 42))
    map_right_eye = list(range(42, 48))
    map_mouth = list(range(48, 68))

    mean_left_eye = np.mean(lm[map_left_eye], 0)  # (x, y)
    half_len_left_eye = np.max((np.max(np.max(lm[map_left_eye], 0) - np.min(lm[map_left_eye], 0)) / 2, 16))
    item_dict['left_eye'] = [mean_left_eye[0], mean_left_eye[1], half_len_left_eye]
    # mean_left_eye[0] = 512 - mean_left_eye[0]  # for testing flip
    half_len_left_eye *= 1.4

    # eye_right
    mean_right_eye = np.mean(lm[map_right_eye], 0)
    half_len_right_eye = np.max((np.max(np.max(lm[map_right_eye], 0) - np.min(lm[map_right_eye], 0)) / 2, 16))
    item_dict['right_eye'] = [mean_right_eye[0], mean_right_eye[1], half_len_right_eye]
    # mean_right_eye[0] = 512 - mean_right_eye[0]  # # for testing flip
    half_len_right_eye *= 1.4

    # mouth
    mean_mouth = np.mean(lm[map_mouth], 0)
    half_len_mouth = np.max((np.max(np.max(lm[map_mouth], 0) - np.min(lm[map_mouth], 0)) / 2, 16))
    item_dict['mouth'] = [mean_mouth[0], mean_mouth[1], half_len_mouth]

    return item_dict