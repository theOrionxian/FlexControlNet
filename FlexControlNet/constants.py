MODEL_PATH = "stabilityai/sd-turbo" #semisal masih error ganti jadi ``sd 2.1
CONTROLNET_PATH = "thibaud/controlnet-sd21-canny-diffusers"
DATASET_PATH = "ErioLatte/laion_60k"

REVISION = None
OUTPUT_DIR = "/content/drive/MyDrive/Deep_Learning/ControlNet/Train14/Output"
LOGGING_DIR = "/content/drive/MyDrive/Deep_Learning/ControlNet/Train14/Log"
CACHE_DIR = "/content/Cache"
TRAIN_DATA_DIR = ""
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
EPOCH = 3
WARMUP_STEPS = 0
LR = 1e-5

VALIDATION_PATH = "/content/drive/MyDrive/Deep_Learning/ControlNet/TestImages/7.png"
VALIDATION_PROMPT = "blue person, purple and blue stars, beautiful majestic"
VALIDATION_STEP = 200

INITIAL_STRUCTURE = [
    ('ResnetBlock2D+Transformer2DModel', 320, 5),
    ('ResnetBlock2D+Transformer2DModel', 320, 5),
    ('Downsample2D', 320, 5),
    ('ResnetBlock2D+Transformer2DModel', 640, 10),
    ('ResnetBlock2D+Transformer2DModel', 640, 10),
    ('Downsample2D', 640, 10),
    ('ResnetBlock2D+Transformer2DModel', 1280, 20),
    ('ResnetBlock2D+Transformer2DModel', 1280, 20),
    ('Downsample2D', 1280, 20),
    ('ResnetBlock2D', 1280, 20),
    ('ResnetBlock2D', 1280, 20),
]
TARGET_CHANNELS = [l[1] for l in INITIAL_STRUCTURE]

#Evaluation
EVAL_DIR = "/content/drive/MyDrive/Deep_Learning/ControlNet/Train14"
EVAL_DS_PATH = "ErioLatte/laion_100"