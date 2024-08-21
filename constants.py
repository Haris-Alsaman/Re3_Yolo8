# Network Constants

USE_SMALL_NET = True

CROP_SIZE = 227
CROP_PAD = 2
MAX_TRACK_LENGTH = 32

import os.path

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs/")
DATA_DIR = os.path.join(
    os.path.dirname(__file__), "training/", "datasets"
)

GPU_ID = "cpu" # for CPU

# GPU_ID = 0 # for GPU

# Drawing constants
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480
PADDING = 2
