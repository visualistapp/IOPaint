import PIL.Image
import cv2
import torch
from loguru import logger

from lama_cleaner.model.base import DiffusionInpaintModel
from lama_cleaner.model.helper.cpu_text_encoder import CPUTextEncoderWrapper
from lama_cleaner.model.utils import handle_from_pretrained_exceptions
from lama_cleaner.schema import InpaintRequest, ModelType


class SD(DiffusionInpaintModel):
    pad_mod = 8
    min_size = 512
    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

    def init_model(self, device: torch.device, **kwargs):
        from diffusers.pipelines.stable_diffusion import StableDiffusionInpaintPipeline

        fp16 = not kwargs.get("no_half", False)

        model_kwargs = {}
        if kwargs["disable_nsfw"] or kwargs.get("cpu_offload", False):
            logger.info("Disable Stable Diffusion Model NSFW checker")
            model_kwargs.update(
                dict(
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                )
            )

        use_gpu = device == torch.device("cuda") and torch.cuda.is_available()
        torch_dtype = torch.float16 if use_gpu and fp16 else torch.float32

        if self.model_info.is_single_file_diffusers:
            if self.model_info.model_type == ModelType.DIFFUSERS_SD:
                model_kwargs["num_in_channels"] = 4
            else:
                model_kwargs["num_in_channels"] = 9

            self.model = StableDiffusionInpaintPipeline.from_single_file(
                self.model_id_or_path, dtype=torch_dtype, **model_kwargs
            )
        else:
            self.model = handle_from_pretrained_exceptions(
                StableDiffusionInpaintPipeline.from_pretrained,
                pretrained_model_name_or_path=self.model_id_or_path,
                variant="fp16",
                dtype=torch_dtype,
                **model_kwargs,
            )

        if kwargs.get("cpu_offload", False) and use_gpu:
            logger.info("Enable sequential cpu offload")
            self.model.enable_sequential_cpu_offload(gpu_id=0)
        else:
            self.model = self.model.to(device)
            if kwargs["sd_cpu_textencoder"]:
                logger.info("Run Stable Diffusion TextEncoder on CPU")
                self.model.text_encoder = CPUTextEncoderWrapper(
                    self.model.text_encoder, torch_dtype
                )

        self.callback = kwargs.pop("callback", None)

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """
        self.set_scheduler(config)

        img_h, img_w = image.shape[:2]

        output = self.model(
            image=PIL.Image.fromarray(image),
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            mask_image=PIL.Image.fromarray(mask[:, :, -1], mode="L"),
            num_inference_steps=config.sd_steps,
            strength=config.sd_strength,
            guidance_scale=config.sd_guidance_scale,
            output_type="np",
            callback_on_step_end=self.callback,
            height=img_h,
            width=img_w,
            generator=torch.manual_seed(config.sd_seed),
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output


class SD15(SD):
    name = "runwayml/stable-diffusion-inpainting"
    model_id_or_path = "runwayml/stable-diffusion-inpainting"


class Anything4(SD):
    name = "Sanster/anything-4.0-inpainting"
    model_id_or_path = "Sanster/anything-4.0-inpainting"


class RealisticVision14(SD):
    name = "Sanster/Realistic_Vision_V1.4-inpainting"
    model_id_or_path = "Sanster/Realistic_Vision_V1.4-inpainting"


class SD2(SD):
    name = "stabilityai/stable-diffusion-2-inpainting"
    model_id_or_path = "stabilityai/stable-diffusion-2-inpainting"
