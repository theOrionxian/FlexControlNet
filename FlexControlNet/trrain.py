import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL import PngImagePlugin
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from einops import rearrange
import pickle


import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, deprecate
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import load_image, make_image_grid
from diffusers.models.controlnet import ControlNetOutput, ControlNetConditioningEmbedding
from diffusers.models.embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)
from diffusers.models.resnet import Downsample2D
import cv2

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from transformers import CLIPTextModel

import bitsandbytes as bnb
import xformers

from IPython.display import display
from controlnet_aux import CannyDetector
from constants import *
from utils import *
from models import (
    SimilarStructureControlNetModel, 
    FlexibleStructureControlNetModel, 
    FlexibleModulatedControlNetModel
)
from datasets_util import collate_fn, make_train_dataset

from huggingface_hub import login

#Acknowledgement: This Training loop is inspired by the diffusers library

if __name__ == '__main__':
    login("")

    PngImagePlugin.MAX_TEXT_CHUNK = 20485760
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    weight_dtype = torch.float16
    logging_dir = Path(OUTPUT_DIR, LOGGING_DIR)
    accelerator_project_config = ProjectConfiguration(project_dir=OUTPUT_DIR, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        project_config=accelerator_project_config,
        mixed_precision="fp16"
    )

    #IMPORTANT: Choose One of These 3 Models
    #initialization for SimilarStructure
    student_controlnet = SimilarStructureControlNetModel(
        target_block_out_channels = [ 320, 640, 1280, 1280],
        attention_head_dim=[
            5,
            10,
            20,
            20
        ], 
        block_out_channels = [160, 320, 640, 1280], 
        controlnet_conditioning_channel_order="rgb",
        cross_attention_dim = 1024,
        down_block_types = [
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D"
        ],
        downsample_padding = 1,
        flip_sin_to_cos = True,
        freq_shift = 0,
        in_channels = 4,
        layers_per_block = 2,
        mid_block_scale_factor = 1,
        norm_eps = 1e-05,
        norm_num_groups = 32,
        num_class_embeds = None,
        only_cross_attention = False,
        projection_class_embeddings_input_dim = None,
        resnet_time_scale_shift = "default",
        upcast_attention = True,
        use_linear_projection = True
    )

    #Flexible structure
    new_structure = [
        ('ResnetBlock2D+Transformer2DModel', 160, 5),
        ('Downsample2D', 160, 5),
        ('ResnetBlock2D+Transformer2DModel', 320, 10),
        ('Downsample2D', 320, 10),
        ('ResnetBlock2D+Transformer2DModel', 480, 15),
        ('Downsample2D', 640, 20),
        ('ResnetBlock2D', 640, 20),
        ('ResnetBlock2D', 1280, 20),
    ]
        
    zero_conv_mapping = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7]

    student_controlnet = FlexibleStructureControlNetModel(
        structure=new_structure,
        zero_conv_mapping=zero_conv_mapping,
        controlnet_conditioning_channel_order="rgb",
        cross_attention_dim = 1024,
        downsample_padding = 1,
        flip_sin_to_cos = True,
        freq_shift = 0,
        in_channels = 4,
        mid_block_scale_factor = 1,
        norm_eps = 1e-05,
        norm_num_groups = 32,
        num_class_embeds = None,
        only_cross_attention = False,
        projection_class_embeddings_input_dim = None,
        resnet_time_scale_shift = "default",
        upcast_attention = True,
        use_linear_projection = True
    ).to('cuda')

    #Flexible Modulated
    new_structure = [
        ('ResnetBlock2D+Transformer2DModel', 160, 5),
        ('DownsampleLoCon2D', 160, 5),
        ('ResnetBlock2D+Transformer2DModel', 320, 10),
        ('DownsampleLoCon2D', 320, 10),
        ('ResnetBlock2D+Transformer2DModel', 480, 15),
        ('DownsampleLoCon2D', 640, 20),
        ('ResnetBlock2D', 640, 20),
        ('ResnetBlock2D', 640, 20),
    ]
    zero_conv_mapping = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7]

    student_controlnet = FlexibleModulatedControlNetModel(
        structure=new_structure,
        zero_conv_mapping=zero_conv_mapping,
        controlnet_conditioning_channel_order="rgb",
        cross_attention_dim = 1024,
        downsample_padding = 1,
        flip_sin_to_cos = True,
        freq_shift = 0,
        in_channels = 4,
        mid_block_scale_factor = 1,
        norm_eps = 1e-05,
        norm_num_groups = 32,
        num_class_embeds = None,
        only_cross_attention = False,
        projection_class_embeddings_input_dim = None,
        resnet_time_scale_shift = "default",
        upcast_attention = True,
        use_linear_projection = True,
        modulation_scale=200
    ).to('cuda')

    student_controlnet.enable_xformers_memory_efficient_attention()

    #initialize other models
    tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
            revision=REVISION,
            use_fast=False,
        )

    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder", revision=REVISION)
    vae = AutoencoderKL.from_pretrained(
                MODEL_PATH, subfolder="vae", revision=REVISION
        )
    unet = UNet2DConditionModel.from_pretrained(
            MODEL_PATH, subfolder="unet", revision=REVISION
        )

    if CONTROLNET_PATH != "":
        teacher_controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH)
    else:
        teacher_controlnet = ControlNetModel.from_unet(unet)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_controlnet.requires_grad_(False)
    student_controlnet.train()

    unet.enable_xformers_memory_efficient_attention()
    teacher_controlnet.enable_xformers_memory_efficient_attention()
    student_controlnet.enable_xformers_memory_efficient_attention()

    params_to_optimize = student_controlnet.parameters()
    optimizer = bnb.optim.AdamW8bit(
        params_to_optimize,
        lr=LR,
        betas=(.9, .999),
        weight_decay=1e-2,
        eps=1e-8
    )

    train_dataset = make_train_dataset(
        tokenizer,
        accelerator,
        size=48000,
        image_col_name="image",
        prompt_col_name="prompt",
        guide_col_name="canny_image"
    )

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=BATCH_SIZE,
        )
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    max_train_steps = EPOCH * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=max_train_steps,
        num_cycles=1,
    )
    # Prepare everything with our `accelerator`.
    student_controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        student_controlnet, optimizer, train_dataloader, lr_scheduler
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    teacher_controlnet.to(accelerator.device, dtype=weight_dtype)

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    global_step = 0
    losses = []

    image_logs = None
    for epoch in range(0, EPOCH):
        for step, batch in enumerate(train_dataloader):
            if step==0: print(batch["pixel_values"].shape)
            with accelerator.accumulate(student_controlnet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                with torch.no_grad():
                    down_block_res_samples_tea, mid_block_res_sample_tea = teacher_controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )

                down_block_res_samples, mid_block_res_sample = student_controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                with torch.no_grad():
                    model_pred_teacher = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples_tea
                        ],
                        mid_block_additional_residual=mid_block_res_sample_tea.to(dtype=weight_dtype),
                        return_dict=False,
                    )[0]

                # Get the target for loss depending on the prediction type
                target = noise
                diff_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                noise_dist_loss = F.mse_loss(model_pred.float(), model_pred_teacher.float(), reduction="mean")

                stu_res = [*down_block_res_samples, mid_block_res_sample]
                tea_res = [*down_block_res_samples_tea, mid_block_res_sample_tea]
                kd_loss, mse_list = controlnet_mse(stu_res, tea_res, reduction="mean")

                loss = diff_loss + noise_dist_loss + 0.05*kd_loss
                with torch.no_grad():
                    losses.append({"diff_loss":diff_loss.detach().cpu(), "kd_loss":kd_loss.detach().cpu(), "noise_dist_loss":noise_dist_loss.detach().cpu(), "mse_list":mse_list.detach().cpu()})

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = student_controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                progress_bar.set_description(f"diff_loss : {float(diff_loss.detach())}, kd_loss : {float(kd_loss.detach())}, noise_loss ; {float(noise_dist_loss.detach())}")
                global_step += 1

                if (global_step % VALIDATION_STEP == 0 or global_step==1) and VALIDATION_PATH != "" and VALIDATION_PROMPT != "":
                    checkpoint_dir = f"/content/checkpoint-{global_step}"
                    student_controlnet.save_pretrained(checkpoint_dir)
                    image = generate_validation_image(vae, text_encoder, tokenizer, unet, checkpoint_dir, accelerator, torch.float16)
                    display(image.resize((256, 256)))

    with open('losses.pkl', 'wb') as f:
        pickle.dump(losses, f)