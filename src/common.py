import cv2
import logging
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import os
import yaml
import glob
from pathlib import Path

from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from PIL import Image
from typing import Dict, List, Optional


FORMAT = '%(asctime)s %(clientip)-16s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()


def find_model_name(name: str, l: List[Dict[str, str]]) -> Optional[str]:
    for small_dict in l:
        if name in small_dict:
            return small_dict[name]
    return None


def read_caption(caption_path: str) -> List[str]:
    with open(caption_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def find_common_prefix(str_list: List[str]):
    return os.path.commonprefix(str_list)


def find_common_suffix(str_list: List[str]):
    str_list_inv = [x[::-1] for x in str_list]
    return find_common_prefix(str_list_inv)


def draw_landmarks_on_image(rgb_image, detection_result, mode:str = 'default'):
    if mode not in ('default', 'binary'):
        raise Exception(f"Unkown mode: {mode}")

    face_landmarks_list = detection_result.face_landmarks
    if mode == 'binary':
        rgb_image = Image.new("RGB", (rgb_image.shape[0], rgb_image.shape[1]), (0, 0, 0))
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for face_landmarks in face_landmarks_list:
        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names, face_blendshapes_scores = [], []
    for face_blendshapes_category in face_blendshapes:
        face_blendshapes_names.append(face_blendshapes_category.category_name)
        face_blendshapes_scores.append(face_blendshapes_category.category_score)

    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    _, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores,
                  label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(),
                  patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def normalizer(image: Image) -> Image:
    """Normalize an image pixel values between [0 - 255]"""

    img = np.array(image)
    return cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)


def contains_only_one_substring(input_string, substring_list):
    count = 0

    for substring in substring_list:
        if substring in input_string:
            count += 1

    return count == 1

def contains_word(string, words):
    for word in words:
        if word.lower() in string.lower():
            return True


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (np.array): Array representing the first bounding box in YOLOv5 format (x_center, y_center, width, height).
        box2 (np.array): Array representing the second bounding box in YOLOv5 format (x_center, y_center, width, height).

    Returns:
        float: IoU score between the two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate coordinates of the intersection rectangle
    x_left = max(x1 - w1 / 2, x2 - w2 / 2)
    y_top = max(y1 - h1 / 2, y2 - h2 / 2)
    x_right = min(x1 + w1 / 2, x2 + w2 / 2)
    y_bottom = min(y1 + h1 / 2, y2 + h2 / 2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate IoU
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou


def bbox_min_max_to_center_dims(x_min, x_max, y_min, y_max, image_width, image_height):
    x_center = (x_min + x_max) / 2.0 / image_width
    y_center = (y_min + y_max) / 2.0 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height


def create_files_list(image_files, txt_file_path):
    with open(txt_file_path, 'w') as f:
        f.write('\n'.join(image_files))


def list_images(images_path: Path, formats: List[str], limit:int = None):
    images = []
    for format in formats:
        images += [
            *glob.glob(str(images_path.absolute()) + f'/*.{format}')
        ]
    return images[:limit]


def create_yaml_file(save_path: Path, train: Path, val: Path, test: Path):
    """
    Construct the yaml file

    :param pathlib.Path txt_dir: path used to create the txt files
    :param pathlib.Path yaml_dir: path used to create the yaml file

    :return: None
    :rtype: NoneType
    """

    yaml_file = {
        'train': str(train.absolute()),
        'val': str(val.absolute()),
        'test': str(test.absolute()),
        'names': {0: 'person'}
    }

    with open(save_path, 'w') as file:
        yaml.dump(yaml_file, file)
