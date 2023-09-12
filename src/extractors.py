import cv2
import numpy as np
from PIL import Image
from typing import Tuple

from controlnet_aux import OpenposeDetector

# Mediapipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from common import draw_landmarks_on_image


AVAILABLE_EXTRACTORS = ('openpose', 'canny', 'mediapipe_face')


def extract_model_from_name(raw_name: str) -> str:
    if 'openpose' in raw_name:
        return 'openpose'
    elif 'canny' in raw_name:
        return 'canny'
    elif 'mediapipe' in raw_name:
        return 'mediapipe_face'
    else:
        raise Exception(f'Unkown model: {raw_name}')


class Extractor:
    def __new__(cls, control_model: str, **kwargs):
        if control_model not in AVAILABLE_EXTRACTORS:
            raise Exception(f'Unknown control model: {control_model}')

        if 'openpose' in control_model:
            return OpenPose(**kwargs)
        elif 'canny' in control_model:
            return Canny(**kwargs)
        elif 'mediapipe_face' in control_model:
            return MediaPipeFace(**kwargs)


class Canny:
    def __init__(self, auto_threshold: bool = False, low_threshold: int = 100, high_threshold: int = 200, **kwargs):
        self.auto_threshold = auto_threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def canny_get_thresholds(self, image: np.array) -> Tuple[float, float]:

        """
        Args: image; numpy array of RGB image
        Returns: low; float; the lower threshold value for canny edge detector
                 high; float; the upper threshold value for canny edge detector
        """

        img_median = np.median(image)
        img_std = np.std(image)
        low = int(max(0, img_median - 0.5*img_std))
        high = int(min(255, img_median + 0.5*img_std))
        return low, high

    def extract(self, image: Image) -> Image:

        """
        Arg: image; Image; image in pillow format
        Returns: canny_image; Image; image with edges marked
        """

        image = np.array(image)

        if self.auto_threshold:
            low_threshold, high_threshold = self.canny_get_thresholds(image)
        else:
            low_threshold, high_threshold = self.low_threshold, self.high_threshold

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        return canny_image

    def __str__(self) -> str:
        return 'Extractor(canny)'


class OpenPose:
    def __init__(self, model: str = 'lllyasviel/ControlNet', **kwargs):
        self.model = OpenposeDetector.from_pretrained(model)

    def extract(self, image: Image) -> Image:
        image = np.array(image)
        pose = self.model(image)
        return pose

    def __str__(self) -> str:
        return 'Extractor(openpose)'


class MediaPipeFace:
    def __init__(self, model: str = 'ressources/mediapipe/face_landmarker_v2_with_blendshapes.task', **kwargs):
        self.base_options = python.BaseOptions(model_asset_path = model)
        self.options = vision.FaceLandmarkerOptions(base_options = self.base_options,
                                            output_face_blendshapes = True,
                                            output_facial_transformation_matrixes = True,
                                            num_faces = 1)
        self.detector = vision.FaceLandmarker.create_from_options(self.options)

    def extract(self, image: Image) -> Image:
        image = np.array(image)
        image_mp = mp.Image(image_format = mp.ImageFormat.SRGB, data = image)
        detection_result = self.detector.detect(image_mp)

        annotated_image = draw_landmarks_on_image(image, detection_result)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        annotated_image = Image.fromarray(annotated_image)

        return annotated_image

    def __str__(self) -> str:
        return 'Extractor(mediapipe_face)'
