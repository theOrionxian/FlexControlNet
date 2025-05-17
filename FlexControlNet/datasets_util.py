import torch
import random
from datasets import load_dataset
from torchvision import transforms
import numpy as np
from constants import *

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }

def make_train_dataset(tokenizer, accelerator, size=25000, image_col_name="image", prompt_col_name="text", guide_col_name="guide"):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if DATASET_PATH is not "" and DATASET_PATH is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            DATASET_PATH,
            cache_dir=CACHE_DIR,
        )
    else:
        if TRAIN_DATA_DIR is not "" and TRAIN_DATA_DIR is not None:
            dataset = load_dataset(
                TRAIN_DATA_DIR,
                cache_dir=CACHE_DIR,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    temp = dataset["train"].select(range(size))
    dataset['train'] = temp
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    image_column = image_col_name
    if image_column not in column_names:
        raise ValueError(
            f"`--image_column` value 'image' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
        )


    caption_column = prompt_col_name #tadinya guide
    if caption_column not in column_names:
        raise ValueError(
            f"`--caption_column` value 'text' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
        )

    conditioning_image_column = guide_col_name #diganti dari guide (krn tes pke dataset Erio) ke canny_image
    if conditioning_image_column not in column_names:
        raise ValueError(
            f"`--conditioning_image_column` value 'conditioning_image' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
        )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < 0:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset