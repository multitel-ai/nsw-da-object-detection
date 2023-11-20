import glob
import hydra
import os
import random
import torch

from diffusers.utils import load_image
from omegaconf import DictConfig
from pathlib import Path

from common import *
from extractors import *
from generators import SDCN, PromptGenerator


# Do not let torch decide on best algorithm (we know better!)
torch.backends.cudnn.benchmark=False 

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    data = cfg['data']
    base_path = os.path.join(*data['base']) if isinstance(data['base'], list) else data['base']
    REAL_data = Path(base_path) / data['real']

    real_images_path = REAL_data / 'images'
    real_captions_path = REAL_data / 'captions'
    real_labels_path = REAL_data / 'labels'

    # Keep track of what feature was used for generation too in the name.
    GEN_data =  Path(base_path) / data['generated'] / cfg['model']['cn_use']

    # Create the generated directory if necessary.
    (GEN_data).mkdir(parents=True, exist_ok=True)

    formats = data['image_formats']
    real_images = []
    for format in formats:
        real_images += [
            *glob.glob(str(real_images_path.absolute()) + f'/*.{format}')
        ]
    real_images.sort()
    real_images_path = real_images
    real_dataset_size = len(real_images_path)

    prompt = cfg['prompt']
    modify_captions = bool(prompt['modify_captions'])
    prompt_generation_size = prompt['generation_size']
    promp_generator = PromptGenerator(prompt['template']) if modify_captions else None

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

    use_captions = bool(model_data['use_captions'])
    use_labels = bool(model_data['use_labels'])

    if use_captions:
        real_captions_path = glob.glob(str(real_captions_path.absolute()) + f'/*')
        real_captions_path.sort()
        if len(real_captions_path) != real_dataset_size:
            raise Exception("Cannot use a captions dataset of different size!")
        logger.info("Using captions")
    else:
        real_captions_path = []

    if use_labels:
        real_labels_path = glob.glob(str(real_labels_path.absolute()) + f'/*')
        real_labels_path.sort()
        if len(real_labels_path) != real_dataset_size:
            raise Exception("Cannot use a labels dataset of different size!")
        logger.info("Using labels")
    else:
        real_labels_path = []

    cn_model = find_model_name(model_data['cn_use'], model_data['cn'])
    cn_model = (cn_model
                if cn_model is not None
                else 'fusing/stable-diffusion-v1-5-controlnet-openpose')

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

    logger.info(f'Results will be saved to {GEN_data}')
    # Generate from each image several synthetic images following the different prompts.

    for idx in range(real_dataset_size):
        image_path = real_images_path[idx]

        caption_path = real_captions_path[idx] if use_captions else None
        label_path = real_labels_path[idx] if use_labels else None

        try:
            image = load_image(image_path)
            image_name = image_path.split(f'{os.sep}')[-1].split('.')[0]

            # Copy the original image to the same directory to ease the quality testing after.
            # image.save(GEN_data / f'b_{i}.png')

            # Feature extraction, save also the features.
            feature = extractor.extract(image)

            # this is feature debugging
            # feature.save(GEN_data / f"f_{i+1}.png")

            # Here we use captions if necessary and modify them if necessary.
            if use_captions:
                positive_prompt = [p.lower() for p in read_caption(caption_path)]
                negative_prompt = [''.join(prompt['negative_simple'])] * len(positive_prompt)

                if modify_captions:
                    def modify_prompt(p):
                        modified_list = promp_generator.prompts(prompt_generation_size, p)
                        cleaned_list = list(filter(lambda new_p: new_p != p , modified_list))
                        if not cleaned_list:
                            logger.info(
                                f"No new prompt created for caption \"{p}\", using the same.")
                            return p
                        return random.choice(cleaned_list)

                    positive_prompt = [modify_prompt(p) for p in positive_prompt]

            # Generate with stable diffusion
            # Clean a little the gpu memory between generations
            torch.cuda.empty_cache()
            output = generator.gen(feature, positive_prompt, negative_prompt)

            # save images
            for j, img in enumerate(output.images):
                img.save(GEN_data / f'{image_name}_{j + 1}.png')

        except Exception as e:
            logger.info(f'Image {image_path}: Exception during Extraction/SDCN', e)

        if (idx + 1) % 50 == 0:
            logger.info(f"Treated {idx + 1} images ({(idx)/real_dataset_size * 100}%).")



if __name__ == '__main__':
    main()
