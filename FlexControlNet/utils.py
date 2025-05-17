import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image
import gc
from controlnet_aux import CannyDetector
from constants import *
from models import (
    SimilarStructureControlNetModel, 
    FlexibleStructureControlNetModel, 
    FlexibleModulatedControlNetModel
)

def generate_validation_image(
    vae, text_encoder, tokenizer, unet, checkpoint_dir, accelerator, weight_dtype, model_str="FlexibleModulatedControlNetModel", high_threshold=200, low_threshold=70
):
    models = {
       "SimilarStructureControlNetModel": SimilarStructureControlNetModel,
       "FlexibleStructureControlNetModel": FlexibleStructureControlNetModel,
       "FlexibleModulatedControlNetModel": FlexibleModulatedControlNetModel
    }
    controlnet = models[model_str].from_pretrained(checkpoint_dir, torch_dtype=weight_dtype) #jangan lupa diganti

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_PATH,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=REVISION,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    pipeline.enable_xformers_memory_efficient_attention()

    generator = torch.Generator(device=accelerator.device).manual_seed(42)

    validation_image, validation_prompt = VALIDATION_PATH, VALIDATION_PROMPT

    validation_image = Image.open(validation_image).convert("RGB")

    # jadiin validation image -> canny
    apply_canny = CannyDetector()
    validation_image = apply_canny(validation_image, low_threshold, high_threshold, detect_resolution=512, image_resolution=512)

    with torch.autocast("cuda"):
        image = pipeline(
            validation_prompt, validation_image, num_inference_steps=20, generator=generator
        ).images[0]

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return image

def controlnet_mse(stu, teach, reduction="mean"):
  mse_list = []
  for st, te in zip(stu, teach):
    te_mean, te_std = te.mean(), te.std()
    nte = (te - te_mean) / te_std
    nst = (st - te_mean) / te_std
    mse = F.mse_loss(nst.float(), nte.float(), reduction="mean")
    mse_list.append(mse)
  mse_list = torch.stack(mse_list).to("cuda")
  if reduction == "mean":
    return mse_list.mean(), mse_list
  if reduction == "sum":
    return mse_list.sum(), mse_list