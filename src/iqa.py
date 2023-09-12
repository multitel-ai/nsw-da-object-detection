import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import os
from pathlib import Path
import re
from typing import List, Optional, Tuple
from tqdm import tqdm

from pyiqa import create_metric
from pyiqa.models.inference_model import InferenceModel

from common import logger


# In this file the approach to measure quality will be the extensive library
# IQA-Pytorch: https://github.com/chaofengc/IQA-PyTorch
# Read also the paper: https://arxiv.org/pdf/2208.14818.pdf

# There are basically two approaches to measure image quality
# - full reference: compare againts a real pristine image
# - no reference: compute metrics following a learned opinion

# Because images are generated there is no reference image to compare to. We
# will be using with the no-reference metrics

# Note that methods using here are agnostic to the content of the image, no
# subjective or conceptual score is given.
# Measures generated here only give an idea of how 'good looking' the images
# are.

# Methods used:
# - brisque: https://www.sciencedirect.com/science/article/abs/pii/S0730725X17301340
# - cliipiqa: https://arxiv.org/pdf/2207.12396.pdf
# - dbccn: https://arxiv.org/pdf/1907.02665v1.pdf
# - niqe: https://live.ece.utexas.edu/research/quality/nrqa.html


# Note that all score measure do not have the same range. Before plotting we
# normalize.
# Methods with an infinite range are of course not normalized.
def normalize(metric, scores, avg_score):
    if metric == 'brisque':
        return scores, avg_score
    elif metric == 'clipiqa' or ('clipiqa' in metric):
        return [score * 100 for score in scores], avg_score * 100
    return scores, avg_score


def measure_several_images(metric: InferenceModel,
                           image_paths: List[str],
                           ref_image_paths: Optional[List[str]] = None
                           ) -> Tuple[float, float]:
    number_of_images = len(image_paths)
    scores = []
    avg_score = 0

    for i, image_path in enumerate(tqdm(image_paths, unit='image')):
        ref_image_path = ref_image_paths and ref_image_paths[i]

        score = metric(image_path, ref_image_path)
        score = score.item()  # This should be adapted if using cpu as device,
                              # here because of cuda we get a 1-dim tensor

        scores.append(score)
        avg_score += score

    avg_score = avg_score / number_of_images
    return scores, avg_score


def is_generated_image(image_path: str) -> bool:
    # You should change the regex in this function to match whatever
    # naming convention you follow for you experience.
    regex = '^[0-9]+_[0-9]+.(jpg|png)'

    image_wo_path = os.path.basename(image_path)
    return re.match(regex, image_wo_path)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # BASE PATHS, please used these when specifying paths
    data_path = cfg['data_path']
    # keep track of what feature was used for generation too in the name
    base_path = os.path.join(*data_path['base'])
    GEN_DATA_PATH =  Path(base_path) / data_path['generated'] / cfg['model']['cn_use']

    logger.info(f'Reading images from {GEN_DATA_PATH}')

    image_paths = [
        str(GEN_DATA_PATH / image_path)
        for image_path in os.listdir(str(GEN_DATA_PATH)) if is_generated_image(image_path)
    ]
    image_paths.sort()
    image_names = [os.path.basename(image_path)[:4] for image_path in image_paths]

   # We are hard-coding the No-Reference methods for the moment.
   # See reasonment above.
    METRIC_MODE = 'NR'

    metrics = [metric.lower() for metric in cfg['iqa']['metrics']]
    if not metrics:
        metrics = ['brisque']
    device = cfg['iqa']['device']

    logger.info(f'Using a {METRIC_MODE} approach, metrics: {metrics} and device: {device}')

    overall_scores = {}
    for metric_name in tqdm(metrics):
        logger.info(f'Measure using {metric_name} metric.')

        iqa_model = create_metric(metric_name, device=device, metric_mode=METRIC_MODE)
        scores, avg_score = measure_several_images(iqa_model, image_paths)
        scores, avg_score = normalize(metric_name, scores, avg_score)

        overall_scores[metric_name] = scores, avg_score

    global_avg_score = 0
    for metric_name in overall_scores:
        scores, avg_score = overall_scores[metric_name]
        global_avg_score += avg_score
        plt.plot(image_names, scores, label = f'Avg score of {metric_name}: {avg_score}')
    global_avg_score = global_avg_score / len(metrics)

    plt.title(f'Dataset: {os.path.basename(str(GEN_DATA_PATH))}\nGlobal avg score: {global_avg_score}', loc='left')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
