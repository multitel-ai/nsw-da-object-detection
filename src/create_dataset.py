import sys
sys.path.append('/home/ucl/elen/tgodelai/.local/lib/python3.9/site-packages') ; sys.path.pop(5) 

import glob
import hydra
import os
import random
import yaml

from omegaconf import DictConfig
from pathlib import Path
from typing import List

def create_mixte_dataset(real_images_dir: str,
                         synth_images_dir: str,
                         txt_dir: str,
                         per_synth_data: float,
                         formats: List[str] = ['jpg', 'png', 'jpeg']):
    """
    Construct the txt file containing a percentage of real and synthetic data

    :param str real_images_dir: path to the folder containing real images
    :param str synth_images_dir: path to the folder containing synthetic images
    :param str txt_dir: path used to create the txt file
    :param float per_synth_data: percentage of synthetic data compared to real (ranges in [0, 1])

    :return: None
    :rtype: NoneType
    """

    train_txt_path = txt_dir / 'train.txt'
    val_txt_path = txt_dir / 'val.txt'
    test_txt_path = txt_dir / 'test.txt'
    data_yaml_path = txt_dir / 'data.yaml'

    real_images_path = real_images_dir / 'images'
    val_images_path = Path(str(real_images_path.absolute()).replace('/real/', '/val/'))
    test_images_path = Path(str(real_images_path.absolute()).replace('/real/', '/test/'))

    real_images = list_images(real_images_path, formats, 250)
    synth_images = list_images(synth_images_dir, formats)
    val_images = list_images(val_images_path, formats)
    test_images = list_images(test_images_path, formats)

    # shuffle images
    random.Random(42).shuffle(real_images)
    random.Random(42).shuffle(synth_images)

    # nb_real_images = int(len(real_images) * (1 - per_synth_data))
    nb_synth_images = int(len(real_images) * per_synth_data)

    real_images = real_images
    synth_images = synth_images[:nb_synth_images]

    train_images = real_images + synth_images

    with open(train_txt_path, 'w') as f:
        f.write('\n'.join(train_images))

    with open(val_txt_path, 'w') as f:
        f.write('\n'.join(val_images))

    with open(test_txt_path, 'w') as f:
        f.write('\n'.join(test_images))

    create_yaml_file(data_yaml_path, train_txt_path, val_txt_path, test_txt_path)


def list_images(images_path: Path, formats: List[str], max_ = None):
    images = []
    for format in formats:
        images += [
            *glob.glob(str(images_path.absolute()) + f'/*.{format}')
        ]
    if max_ is None:
        return images
    else:
        return images[:250]

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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    data = cfg['data']
    base_path = Path(data['base'])
    GEN_DATA_PATH =  Path(base_path) / data['generated'] / cfg['model']['cn_use']
    REAL_DATA_PATH = Path(base_path) / data['real'] 
    
    if cfg['ml']['augmentation_percent']==0:  
        fold = REAL_DATA_PATH
    else:
        fold = cfg['model']['cn_use'] + str(cfg['ml']['augmentation_percent']) 
        fold = REAL_DATA_PATH / fold
        
    if not os.path.isdir(fold): 
        os.makedirs(fold)

    create_mixte_dataset(
        REAL_DATA_PATH, GEN_DATA_PATH, fold, cfg['ml']['augmentation_percent']
    )


if __name__ == '__main__':
    main()
