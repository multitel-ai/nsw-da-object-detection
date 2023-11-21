import cv2
import logging
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import os

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
