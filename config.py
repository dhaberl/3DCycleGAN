import torch
from monai.transforms import AddChannel, Compose, EnsureType, LoadImage, ScaleIntensity

# CUDA device and CPU stuff
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 32
CUDNNBENCHMARK = True

# Continue training
LOAD_MODEL = False
CONTINUE_FROM_EPOCH = 9

# Hyperparameters
EPOCHS = 1000
BATCH_SIZE = 1

# Loss function constants for identity and cycle loss
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10

# Learning rate
LEARNING_RATE = 0.0002
DECAY_EPOCHS = 500
DECAY_OFFSET = 0
SEED = 0

# Verbose
PRINT_FREQUENCY = 1
CHECKPOINT_FREQUENCY = 25
SAVE_REAL_IMG = False

# Set data directories and experiment name
EXPERIMENT_ID = "001_Test_Experiment"
DATA_DIR = "Dataset/"
OUTPUT_DIR = "output_dir/"

# Transforms
transforms = Compose(
    [
        LoadImage(image_only=True),
        AddChannel(),
        ScaleIntensity(minv=-1, maxv=1),
        EnsureType(),
    ]
)
