import torch
from constants import EVAL_DIR, EVAL_DS_PATH, MODEL_PATH, CONTROLNET_PATH
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from metrics import *
from tqdm import tqdm
from datasets import load_dataset
from models import (
    SimilarStructureControlNetModel, 
    FlexibleStructureControlNetModel, 
    FlexibleModulatedControlNetModel
)

if __name__ == '__main__':
    #please adjust this
    controlnet = FlexibleModulatedControlNetModel.from_pretrained(EVAL_DIR, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    #controlnet = SimilarStructureControlNetModel.from_pretrained(EVAL_DIR, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    #buat load teacher controlnet
    teacher_controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float16).to("cuda")
    pipe_teacher = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_PATH, controlnet=teacher_controlnet, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")

    pipe_teacher.scheduler = UniPCMultistepScheduler.from_config(pipe_teacher.scheduler.config)
    pipe_teacher.enable_model_cpu_offload()
    pipe_teacher.set_progress_bar_config(disable=True)

    #load dataset
    eval_ds = load_dataset(EVAL_DS_PATH)

    #Metrics main code
    fid_metric = FID_metric()
    ssim_metric = SSIM_metric()
    psnr_metric = PSNR_metric()
    lpips_metric = LPIPS_metric()

    fid_metric_wrt_teacher = FID_metric()
    ssim_metric_wrt_teacher = SSIM_metric()
    psnr_metric_wrt_teacher = PSNR_metric()
    lpips_metric_wrt_teacher = LPIPS_metric()

    prompts = []

    for row in tqdm(eval_ds['train']):
        gt, can, prompt = row['image'].convert('RGB'), row['canny_image'], row['prompt']
        generator = torch.Generator(device=device).manual_seed(42)
        torch.cuda.empty_cache()

        output = pipe(
            prompt, image=can, generator=generator, num_inference_steps=20
        ).images[0]

        output_teacher = pipe_teacher(
            prompt, image=can, generator=generator, num_inference_steps=20
        ).images[0]

    #w.r.t ground truth
    fid_metric.update(gt, real=True)
    fid_metric.update(output)
    ssim_metric.update(gt, output)
    psnr_metric.update(gt, output)
    lpips_metric.update(gt, output)

    #w.r.t teacher model
    fid_metric_wrt_teacher.update(output_teacher, real=True)
    fid_metric_wrt_teacher.update(output)
    ssim_metric_wrt_teacher.update(output_teacher, output)
    psnr_metric_wrt_teacher.update(output_teacher, output)
    lpips_metric_wrt_teacher.update(output_teacher, output)

    print("Metrics With Respect to Ground Truth : ")
    fid_metric.print()
    ssim_metric.print()
    psnr_metric.print()
    lpips_metric.print()
    print("")
    print("Metrics With Respect to Teacher : ")
    fid_metric_wrt_teacher.print()
    ssim_metric_wrt_teacher.print()
    psnr_metric_wrt_teacher.print()
    lpips_metric_wrt_teacher.print()