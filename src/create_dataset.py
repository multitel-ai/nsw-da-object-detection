import os
import argparse
import random
import yaml

from common import logger

def create_mixte_dataset(real_images_dir: str, syn_images_dir: str, txt_dir: str, per_syn_data: float, n_file: int):
    """
    Construct the txt file containing a percentage of real and synthetic data
    :param real_images_dir: path to the folder containing real images
    :param syn_images_dir: path to the folder containing synthetic images
    :param txt_dir: path used to create the txt file
    :param per_syn_data: float, [0, 1], percentage of synthetic data compared to real ones
    :param n_file: int, number of the file used for the txt and yaml names
    :return: /
    """
    set = 'train'
    txt_dir_path = os.path.join(txt_dir, set + '.txt')

    real_images_path = os.path.join(real_images_dir, 'images/')
    real_images_path = os.path.abspath(real_images_path)
    real_images = sorted(os.listdir(real_images_path))
    if real_images and real_images[0] == '.DS_Store': real_images = real_images[1:]
    random.Random(42).shuffle(real_images)

    nb_real_images = int(len(real_images) * (1 - per_syn_data))
    nb_syn_images = int(len(real_images) * per_syn_data)
    aorw = 'w'
    if per_syn_data != 0:
        #syn_images_path = os.path.join(syn_images_dir, set, 'images/')
        syn_images = sorted(os.listdir(syn_images_dir))#syn_images_path))
        if syn_images and syn_images[0] == '.DS_Store': syn_images = syn_images[1:]
        random.Random(42).shuffle(syn_images)
        syn_images = syn_images[:nb_syn_images]
        with open(txt_dir_path, 'w') as f:
            for image in syn_images:
                name = syn_images_dir + str(image) + '\n'
                f.write(name)
        aorw = 'a'

    real_images = real_images[:nb_real_images]
    with open(txt_dir_path, aorw) as f:
        for image in real_images:
            name = str(os.path.join(real_images_path, image)) + '\n'
            f.write(name)

    sets = ['val', 'test']
    for set in sets:
        real_images_path = os.path.join(real_images_dir.split('real')[0], set, 'images/')
        real_images_path = os.path.abspath(real_images_path)
        real_images = sorted(os.listdir(real_images_path))
        if real_images and real_images[0] == '.DS_Store': real_images = real_images[1:]
        random.Random(42).shuffle(real_images)

        txt_dir_path = os.path.join(txt_dir, set  + '.txt')
        with open(txt_dir_path, 'w') as f:
            for image in real_images:
                name = str(os.path.join(real_images_path, image)) + '\n'
                f.write(name)

    yaml_dir = os.path.join(os.path.join(txt_dir, 'coco' +'.yaml'))
    create_yaml_file(txt_dir, yaml_dir)


def create_yaml_file(txt_dir, yaml_dir):
    """
    Construct the yaml file
    :param txt_dir: path used to create the txt files
    :param yaml_dir: path used to create the yaml file
    :param n_file: int, number of the file used for the txt and yaml names
    :return: /
    """
    yaml_file = {
                    'path': '',
                    'train': os.path.abspath(os.path.join(txt_dir, 'train.txt')),
                    'val': os.path.abspath(os.path.join(txt_dir, 'val.txt')),
                    'test': os.path.abspath(os.path.join(txt_dir, 'test.txt')),
                    'names': {0: 'person'}
                }
    yaml_file['path'] = os.path.abspath(txt_dir)
    with open(yaml_dir, 'w') as file:
        yaml.dump(yaml_file, file)

def run():
    parser = argparse.ArgumentParser(description="Create custom train.txt, valid.txt and test.txt.")
    parser.add_argument("--real_images_dir", type=str,
                        help="Path to the folder that contains the real images")
    parser.add_argument("--syn_images_dir", type=str,
                        help="Path to the folder that contains the synthetic images")
    parser.add_argument("--txt_dir", type=str,
                        help='Path used to create the txt file')
    parser.add_argument("--per_syn_data", type=float,
                        help='Percentage of synthetic data compared to real ones used for pre-training')
    parser.add_argument("--n_file", type=int, help='Number of the file')
    args = parser.parse_args()
    logger.info(f"Command line arguments: {args}")

    create_mixte_dataset(args.real_images_dir, args.syn_images_dir, args.txt_dir, args.per_syn_data, args.n_file)

if __name__ == '__main__':
    run()