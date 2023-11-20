
import hydra
import matplotlib.pyplot as plt
import os
import re
import sys
import uuid
import yaml

from omegaconf import DictConfig
from pathlib import Path
from pyiqa import create_metric
from pyiqa.models.inference_model import InferenceModel
from tqdm import tqdm
from typing import List, Optional, Tuple

from common import logger, find_common_prefix, find_common_suffix
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd

# TODO: fix this, I do not like this at all
sys.path.append(os.path.join(os.getcwd(), "ultralytics")) 
from ultralytics import YOLO

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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # BASE PATHS, please used these when specifying paths
    data_path = cfg['data']
    # keep track of what feature was used for generation too in the name
    base_path = os.path.join(*data_path['base']) if isinstance(data_path['base'], list) else data_path['base']
    GEN_DATA_PATH =  Path(base_path) / data_path['generated'] / cfg['model']['cn_use']
    LOG_PATH = Path(base_path) / data_path['real']
    print("LOG PATH", LOG_PATH)
    cns = ['lllyasviel_canny', 'lllyasviel_openpose', 'crucible_mediapipe_face', 'controlnet_segmentation']
    all_paths = [Path(base_path) / data_path['real'] / "images", Path(base_path) / data_path['real'] / "coco/Coco_1FullPerson"] + [Path(base_path) / data_path['generated'] / cn for cn in cns]

    _cns = ["real", "real_full"] + cns
    overall_scores = {"score":[], "metric":[], "cn":[], "path":[]}
    for path in tqdm(all_paths, unit="folder"):
        logger.info(f'Reading images from {path}')

        image_paths = [
            str(path / image_path)
            for image_path in os.listdir(str(path)) if True # is_generated_image(image_path)
        ]
        image_paths.sort()

        print(path, len(image_paths), len(os.listdir(str(path))) )

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

        cn = _cns[0] ; _cns = _cns[1:]
        for metric_name in tqdm(metrics, unit="metric"): 
            logger.info(f'Measure using {metric_name} metric.')

            iqa_model = create_metric(metric_name, device=device, metric_mode=METRIC_MODE)
            scores, avg_score = measure_several_images(iqa_model, image_paths)
            scores_, avg_score = adapt_metric(metric_name, scores, avg_score)

            scores, avg_score = normalize(scores_)

            overall_scores["score"] = [*overall_scores["score"], *scores_]
            overall_scores["metric"] = [*overall_scores["metric"], *[metric_name for _ in range(len(scores))]]
            overall_scores["cn"] = [*overall_scores["cn"], *[cn for _ in range(len(scores))]]
            overall_scores["path"] = [*overall_scores["path"], *os.listdir(str(path))] # *image_names]
        overall_scores["score"] = np.array(overall_scores["score"])
        overall_scores["metric"] = np.array(overall_scores["metric"])
        overall_scores["cn"] = np.array(overall_scores["cn"])
        overall_scores["path"] = np.array(overall_scores["path"])
       # with open('/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/data/iqa_values.pkl', 'wb') as fp:
       #     pickle.dump(overall_scores, fp)
       #     print("dictionary saved successfully")
        global_avg_score = np.zeros(len(cns))
        for metric_name in metrics:
            for i, cn in enumerate(cns):
                scores = overall_scores["score"][overall_scores["metric"]==metric_name][overall_scores["cn"][overall_scores["metric"]==metric_name]==cn]
                avg_score = scores.mean(); print("Avg score: metric, cn", metric_name, cn, scores)
                global_avg_score[i] += avg_score
            # plt.plot(image_names, scores, label = f'Avg score of {metric_name}: {avg_score}')
        global_avg_score = global_avg_score / len(metrics)

    # Figure
    fig, ax = plt.subplots(1,1)
    ax = sns.boxplot(data=overall_scores, x="metric", y="score", ax=ax, hue='cn')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    #plt.title(f'Dataset: {os.path.basename(str(GEN_DATA_PATH))}\nGlobal avg score: {global_avg_score}',loc='left')
    plt.title("Quality comparison between ControlNet models used to generate images")
    plt.legend()
    #file_name = "metrics" + cfg['model']['cn_use'] + ".png"
    file_name = "comparison_metrics_all_models.png"
    plt.savefig(LOG_PATH / file_name)
    df = pd.DataFrame.from_dict(overall_scores)
    path = LOG_PATH / 'iqa_values.csv'
    df.to_csv(path)

@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main_2(cfg: DictConfig) -> None:
    data = cfg['data'] ; active = cfg['active'] ;  iqa = cfg['iqa']
    base_path = Path(data['base']) 
    GEN_DATA_PATH =  Path(base_path) / data['generated'] / cfg['model']['cn_use'] 
    REAL_DATA = Path(base_path) / data['real']
    
    run = "" if cfg['ml']['augmentation_percent']==0 else cfg['model']['cn_use'] + str(cfg['ml']['augmentation_percent'])
    data_yaml_path = "/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/" / REAL_DATA / 'data.yaml'
    model_path = "/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/" / Path(base_path) / str( cfg['model']['cn_use'] + str(cfg['ml']['augmentation_percent']) + ".pt")
    print("model_path", model_path)
    cn_use = cfg['model']['cn_use']
    aug_percent = cfg['model']['augmentation_percent']
    name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}_iqa"
    #name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}_{AL}" if active["abled"] else name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}"
    #name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}_{iqa}" if iqa["abled"] else name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}" 

    #data_yaml_path = str(data_yaml_path.absolute())
    if iqa["abled"]:
        image_paths = ["/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/" + str(GEN_DATA_PATH / im) for im in os.listdir(GEN_DATA_PATH)]
        image_paths.sort()
        mean = False
        if mean: 
           iqa_value_path = "/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/data/real/iqa_values.csv" 
           scores = pd.read_csv(iqa_value_path)
           scores = scores[scores['cn']==cn_use] 
           scores_reshape = scores.pivot(index=scores.columns.tolist()[-1], columns='metric', values='score')
           scores_reshape = scores_reshape.reset_index()
           scale = StandardScaler()
           all_scores_scaled = scale.fit_transform(scores_reshape)
           mean_scores = np.mean(all_scores_scaled, axis=1)
           sel = int(cfg['ml']['train_nb'] * cfg['model']['augmentation_percent'])
           selected_idx = np.argsort(mean_scores)[-sel:]
           selected_images = list(np.array(image_paths)[selected_idx]) 
        else: 
           metric = iqa['metric']
           device = iqa['device']
           iqa_model = create_metric(metric, device=device, metric_mode='NR')
           scores, avg_score = measure_several_images(iqa_model, image_paths)
           #scores, avg_score = adapt_metric(metric, scores, avg_score)
           #scores, avg_score = normalize(scores)
           sel = int(cfg['ml']['train_nb'] * cfg['model']['augmentation_percent'])
           if metric == 'brisque':
              selected_idx = np.argsort(scores)[:sel]
           else:
              selected_idx = np.argsort(scores)[-sel:] #-sel:, :sel 
           selected_images = list(np.array(image_paths)[selected_idx])
        with open(data_yaml_path, 'r') as _file:
            used_data_ = yaml.safe_load(_file) 
        train = used_data_["train"]
        with open(train, 'r') as _file: 
            used_data = _file.readlines()
        U = [u.replace('\n','') for u in used_data]
        #used_data = [u.split("/")[-1] for u in used_data]
        print("U", U); all_data = U + selected_images
        print("all_data", all_data)

        fold = '/'.join(str(data_yaml_path).split('/')[:-1]) + f"/iqa_{cfg['model']['cn_use']}_{cfg['model']['augmentation_percent']}" #f"active_$
        if not os.path.isdir(fold): 
            os.makedirs(fold)
        data_yaml_path2 = f"{fold}/data_iqa_{cfg['model']['cn_use']}_{cfg['model']['augmentation_percent']}.yaml"
        os.system(f"cp {data_yaml_path} {data_yaml_path2}")
        # _data_yaml_path2 = fold + "/" + str(_data_yaml_path2).split('/')[-1] 
        new_train = data_yaml_path2.replace("data_", "train_").replace("yaml", "txt")
        os.system(f"cp {used_data_['train']} {new_train}")
        print("new train", new_train)
        used_data_["train"] = new_train
        with open(data_yaml_path2, 'w') as _file:
            yaml.dump(used_data_, _file)
        with open(new_train, 'w') as _file: 
            _file.write("\n".join(all_data))

        model = YOLO("yolov8n.yaml")
        model.model.query = False
        model.train(
            data = str(Path(data_yaml_path2).absolute()),
            epochs = cfg['ml']['epochs'],
            project = 'sdcn-coco',
            control_net=cn_use,
            ALsampling=cfg['logs']['sampling'], 
            experiment=cfg['logs']['experiment'],
            # entity = 'sdcn-nantes',
            name = name
        )

if __name__ == '__main__':
    main_2()
