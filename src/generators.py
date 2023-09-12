import numpy as np
from typing import List

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

from common import logger


class SDCN:
    def __init__(self, sd_model: str, control_model: str, seed: int, device='cpu',  cn_extra_settings = {}):

        logger.info(f'Initializing SDCN with {sd_model} and {control_model}, seed ={seed}, device={device}')

        self.seed = seed
        self.device = device
        self.control_model = control_model

        self.controlnet = ControlNetModel.from_pretrained(
            self.control_model,
            torch_dtype = torch.float16,
            **cn_extra_settings
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model,
            controlnet = self.controlnet,
            torch_dtype = torch.float16,
            safety_checker = None,
        )

        # The default config seems to work best for the moment, we would need to tweak a lot to know
        # what to use.
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        # Docs of this scheduler: https://huggingface.co/docs/diffusers/main/en/api/schedulers/unipc
        # Example of scheduler with full parameters customization, does not seem to work fine...
        self.pipe.scheduler2 = UniPCMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            trained_betas=None, # this is a list of floats or the equivalen 1-dim np.array
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            sample_max_value=1.0,
            predict_x0=True,
            solver_type="bh2",
            lower_order_final=True,
            disable_corrector=[],
            solver_p=None,
            # The arguments below are listed in the documentation but are not part of the current source code
            # use_karras_sigmas=False,
            # timestep_spacing="linspace",
            # steps_offset=0,
        )

        # Move the whole pipe to the designed device to avoid malformed cuda/cpu instructions
        # self.pipe.to(device)

        # The line below is explained in https://huggingface.co/blog/controlnet but sometimes
        # it will throw an error later in the pipeline about having or not instructions for half
        # floats if you try to use 'cuda'.
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()

    def __str__(self) -> str:
        return f'SDCN({self.control_model})'

    def gen(self,
            condition: np.array | List[np.array],
            positive_prompts: List[str],
            negative_prompts: List[str],
            quality: str = 30,
            guidance_scale: float = 7.0,
        ):
        generator = [
            torch.Generator(device=self.device).manual_seed(self.seed) for i in range(len(positive_prompts))
        ]

        output = self.pipe(
            positive_prompts,
            condition,
            negative_prompt = negative_prompts,
            generator = generator,
            num_inference_steps = quality,
            guidance_scale = guidance_scale,
        )

        return output
