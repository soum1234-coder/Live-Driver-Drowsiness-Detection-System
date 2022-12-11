import os
import time
import sys
from stat import S_ISREG, ST_CTIME, ST_MODE
import numpy as np
import itertools

from PIL import Image, ImageDraw
import face_recognition

def load_image(file_path):
    face = Image.open(file_path)
    face = face.convert(mode="RGB")
    return face

# create a variable for the facial feature coordinates
def face_landmarks(file_path):
    loaded_image = face_recognition.load_image_file(file_path)
    facial_features_list = face_recognition.face_landmarks(loaded_image)
    return facial_features_list

"""
create a placeholder list for the eye coordinates
and append coordinates for eyes to list unless eyes
weren't found by facial recognition
"""

def both_eye_coors(face_features):

    left_eyes = []
    right_eyes = []
    try:
        left_eyes.append(face_features[0]['left_eye'])
        right_eyes.append(face_features[0]['right_eye'])
    except:
        pass

    left_eyes = list(itertools.chain(*left_eyes))
    right_eyes = list(itertools.chain(*right_eyes))

    return (left_eyes, right_eyes)

def max_min_points(eye_list):
    # establish the max x and y coordinates of the eye
    x_max = max([coordinate[0] for coordinate in eye_list])
    x_min = min([coordinate[0] for coordinate in eye_list])
    y_max = max([coordinate[1] for coordinate in eye_list])
    y_min = min([coordinate[1] for coordinate in eye_list])

    return (x_max, x_min, y_max, y_min)

# establish the range of x and y coordinates
def coor_range(x_max, x_min, y_max, y_min):

    x_range = x_max - x_min
    y_range = y_max - y_min

    return (x_range, y_range)

    """
    in order to make sure the full eye is captured,
    calculate the coordinates of a square that has a
    50% cushion added to the axis with a larger range and
    then match the smaller range to the cushioned larger range
    """
def define_buffer(eye_coors):

    x_max, x_min, y_max, y_min = max_min_points(eye_coors)
    x_range, y_range = coor_range(x_max, x_min, y_max, y_min)

    right = round(.5*x_range) + x_max
    left = x_min - round(.5*x_range)
    bottom = round(((right-left) - y_range))/2 + y_max
    top = y_min - round(((right-left) - y_range))/2

    return (left, top, right, bottom)

def process_face_img(filepath):
    face_image = load_image(filepath)

    face_features = face_landmarks(filepath)

    left_eyes, right_eyes = both_eye_coors(face_features)

    left_eye_coors = define_buffer(left_eyes)
    left_eye_im = face_image.crop(left_eye_coors).resize((24,24))

    right_eye_coors = define_buffer(right_eyes)
    right_eye_im = face_image.crop(right_eye_coors).resize((24,24))

    return left_eye_im, right_eye_im

def save_cropped_eyes(filename, L_eye, R_eye, L_pred, R_pred):
    L_eye.save('upload/crop/'+filename+'_left-'+L_pred+'.jpg')
    R_eye.save('upload/crop/'+filename+'_right-'+R_pred+'.jpg')

def convert_words(list):
    new_list = []
    for pred in list:
        if pred<0.5:
            new_list.append('closed')
        elif pred>0.5:
            new_list.append('open')
        else:
            new_list.append('error')
    return new_list

def get_file_logs(dir_path):

    entries = (os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path))

    entries = ((os.stat(path), path) for path in entries)

    entries = ((stat[ST_CTIME], path)
               for stat, path in entries if S_ISREG(stat[ST_MODE]))

    logs = sorted(list(entries))[::-1]
    logs_clean = []

    for time, filename in logs:
        filename_edit = filename.split('/')[-1]

        if filename_edit != '.DS_Store':
            logs_clean.append(filename_edit)

    return logs_clean