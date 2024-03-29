# © - 2024 Université de Mons, Multitel, Université Libre de Bruxelles, Université Catholique de Louvain

# CIA is free software. You can redistribute it and/or modify it 
# under the terms of the GNU Affero General Public License 
# as published by the Free Software Foundation, either version 3 
# of the License, or any later version. This program is distributed 
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License 
# for more details. You should have received a copy of the Lesser GNU 
# General Public License along with this program.  
# If not, see <http://www.gnu.org/licenses/>.

import hydra
import matplotlib.pyplot as plt
import os
import re

from omegaconf import DictConfig
from pathlib import Path
from pyiqa import create_metric
from pyiqa.models.inference_model import InferenceModel
from tqdm import tqdm
from typing import List, Optional, Tuple

from common import logger, find_common_prefix, find_common_suffix
import numpy as np
import seaborn as sns


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

# TODO: some metrics already exists on pytorch, see if we can bypass the pytorch-iqa module

# Note that all score measure do not have the same range. Before plotting we
# normalize.
# Methods with an infinite range are of course not normalized.
def adapt_metric(metric, scores, avg_score):
    if metric == 'brisque':
        return scores, avg_score
    elif metric == 'clipiqa' or ('clipiqa' in metric):
        return [(1 - score)*100 for score in scores], avg_score * 100
    return scores, avg_score


def normalize(values):
    min_value = min(values)
    values = [x - min_value for x in values]

    max_value = max(values)
    values = [x / max_value for x in values]

    return values, (sum(values) / len(values))


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


@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # BASE PATHS, please used these when specifying paths
    data_path = cfg['data']
    # keep track of what feature was used for generation too in the name
    base_path = os.path.join(*data_path['base']) if isinstance(data_path['base'], list) else data_path['base']
    GEN_DATA_PATH =  Path(base_path) / data_path['generated'] / cfg['model']['cn_use']
    LOG_PATH = Path(base_path) / data_path['real'] 

    logger.info(f'Reading images from {GEN_DATA_PATH}')

    image_paths = [
        str(GEN_DATA_PATH / image_path)
        for image_path in os.listdir(str(GEN_DATA_PATH)) if is_generated_image(image_path)
    ]
    image_paths.sort()

    prefix_len = len(find_common_prefix(image_paths))
    suffix_len = len(find_common_suffix(image_paths))
    image_names = [image_path[prefix_len:-suffix_len] for image_path in image_paths]

   # We are hard-coding the No-Reference methods for the moment.
   # See reasonment above.
    METRIC_MODE = 'NR'

    metrics = [metric.lower() for metric in cfg['iqa']['metrics']]
    if not metrics:
        metrics = ['brisque']
    device = cfg['iqa']['device']

    logger.info(f'Using a {METRIC_MODE} approach, metrics: {metrics} and device: {device}')

    overall_scores = {"score":[], "metric":[]}
    for metric_name in tqdm(metrics):
        logger.info(f'Measure using {metric_name} metric.')

        iqa_model = create_metric(metric_name, device=device, metric_mode=METRIC_MODE)
        scores, avg_score = measure_several_images(iqa_model, image_paths)
        scores, avg_score = adapt_metric(metric_name, scores, avg_score)

        scores, avg_score = normalize(scores)

        overall_scores["score"] = [*overall_scores["score"], *scores]
        overall_scores["metric"] = [*overall_scores["metric"], *[metric_name for _ in range(len(scores))] ]

    overall_scores["score"] = np.array(overall_scores["score"])
    overall_scores["metric"] = np.array(overall_scores["metric"])

    global_avg_score = 0
    for metric_name in overall_scores:
        scores = overall_scores["score"][overall_scores["metric"]==metric_name]
        avg_score = scores.mean()
        global_avg_score += avg_score
        # plt.plot(image_names, scores, label = f'Avg score of {metric_name}: {avg_score}')
    global_avg_score = global_avg_score / len(metrics)
    
    fig, ax = plt.subplots(1,1)
    ax = sns.boxplot(data=overall_scores, x="metric", y="score", ax=ax)
    plt.title(f'Dataset: {os.path.basename(str(GEN_DATA_PATH))}\nGlobal avg score: {global_avg_score}',loc='left')
    plt.legend()
    file_name = "metrics" + cfg['model']['cn_use'] + ".png"
    plt.savefig(LOG_PATH / file_name)


if __name__ == '__main__':
    main()
