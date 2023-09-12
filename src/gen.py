import glob
import hydra
from omegaconf import DictConfig
import os
from pathlib import Path

from diffusers.utils import load_image
import torch

from common import *
from extractors import *
from generators import SDCN

# Do not let torch decide on best algorithm (we know better!)
torch.backends.cudnn.benchmark=False


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:

    # BASE PATHS, please used these when specifying paths
    data_path = cfg['data_path']
    base_path = os.path.join(*data_path['base'])
    REAL_DATA_PATH = Path(base_path) / data_path['real']
    # keep track of what feature was used for generation too in the name
    GEN_DATA_PATH =  Path(base_path) / data_path['generated'] / cfg['model']['cn_use']

    # Specify the results path
    (GEN_DATA_PATH).mkdir(parents=True, exist_ok=True)

    formats = cfg['image_formats']
    images = []
    for format in formats:
        images += [
            *glob.glob(str(REAL_DATA_PATH.absolute()) + f'/*.{format}')
        ]
    images = [load_image(image_path) for image_path in images]

    prompt = cfg['prompt']
    if isinstance(prompt['base'], str):
        positive_prompt = [prompt['base'] + ' ' + prompt['modifier'] + ' ' + prompt['quality']]
        negative_prompt = [''.join(prompt['negative_simple'])]
    else:
        positive_prompt = [
            pb + ' ' + prompt['modifier'] + ' ' + prompt['quality']
            for pb in prompt['base']
        ]
        negative_prompt = [''.join(prompt['negative_simple'])] * len(positive_prompt)

    # Specify the model and feature extractor. Be aware that ideally both extractor and
    # generator should be using the same feature.
    model_data = cfg['model']
    sd_model = model_data['sd']

    cn_model = find_model_name(model_data['cn_use'], model_data['cn'])
    cn_model = cn_model if cn_model is not None else 'fusing/stable-diffusion-v1-5-controlnet-openpose'

    seed = model_data['seed']
    device = model_data['device']

    if model_data['cn_use'] in model_data['cn_extra_settings']:
        cn_extra_settings = model_data['cn_extra_settings'][model_data['cn_use']]
    else:
        cn_extra_settings = {}

    generator = SDCN(
        sd_model,
        cn_model,
        seed,
        device = device,
        cn_extra_settings = cn_extra_settings
    )
    extractor = Extractor(extract_model_from_name(model_data['cn_use']))
    logger.info(f'Using extractor: {extractor} and generator: {generator}')

    logger.info(f'Results will be saved to {GEN_DATA_PATH}')
    # Generate from each image several synthetic images following the different prompts.
    for i, image in enumerate(images):
        try:
            # Copy the original image to the same directory to ease the quality testing after.
            image.save(GEN_DATA_PATH / f'b_{i+1}.png')

            # Feature extraction, save also the features.
            feature = extractor.extract(image)
            feature.save(GEN_DATA_PATH / f"f_{i+1}.png")

            # Generate with stable diffusion
            output = generator.gen(feature, positive_prompt, negative_prompt)

            # save images
            for j, img in enumerate(output.images):
                img.save(GEN_DATA_PATH / f'{i+1}_{j+1}.png')
        except Exception as e:
            logger.info('Image {i}: Exception during Extraction/SDCN', e)


if __name__ == '__main__':
    main()
