import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')

# Model Settings
MISTRAL_MODEL = os.getenv('MISTRAL_MODEL', 'mistral-tiny')
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', 'models/local_model')
STABILITY_MODEL = os.getenv('STABILITY_MODEL', 'stable-diffusion-v1-5')
STABILITY_API_HOST = os.getenv('STABILITY_API_HOST', 'api.stability.ai')

# Directories
TEXT_OUTPUT_DIR = os.getenv('TEXT_OUTPUT_DIR', 'output/text')
IMAGE_OUTPUT_DIR = os.getenv('IMAGE_OUTPUT_DIR', 'output/images')
CHECK_RESULTS_DIR = os.getenv('CHECK_RESULTS_DIR', 'output/results')
GENERATED_DATASET_DIR = os.getenv('GENERATED_DATASET_DIR', 'output/datasets')
LOCAL_MODEL_DIR = os.getenv('LOCAL_MODEL_DIR', 'models')

# Generation Settings
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 10))
NUM_SAMPLES = int(os.getenv('NUM_SAMPLES', 100))
IMAGE_SIZE = tuple(map(int, os.getenv('IMAGE_SIZE', '512,512').split(',')))
NUM_INFERENCE_STEPS = int(os.getenv('NUM_INFERENCE_STEPS', 50))
GUIDANCE_SCALE = float(os.getenv('GUIDANCE_SCALE', 7.5))
IMAGE_FORMAT = os.getenv('IMAGE_FORMAT', 'PNG')
IMAGE_QUALITY = int(os.getenv('IMAGE_QUALITY', 95))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')

# Model Settings
MODEL_SETTINGS = {
    'max_length': 1000,
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 50
}

# Dataset generation settings
MAX_SAMPLES = 100
MIN_SAMPLES = 1

# Quality check thresholds
QUALITY_THRESHOLD = 0.8
MISSING_THRESHOLD = 0.1
OUTLIER_THRESHOLD = 0.05
DUPLICATE_THRESHOLD = 0.02

# Dataset Configuration
IMAGE_OUTPUT_DIR = os.path.join(GENERATED_DATASET_DIR, "images")
TEXT_OUTPUT_DIR = os.path.join(GENERATED_DATASET_DIR, "text")

# Generation Configuration
MAX_RETRIES = 3

# Image Generation Configuration
IMAGE_SIZE = (512, 512)
IMAGE_FORMAT = "PNG"
IMAGE_QUALITY = 95
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

# Local Model Configuration
LOCAL_MODEL_TYPE = "stable-diffusion"  # or other model types

# Model Configuration
DEFAULT_MODEL_TYPE = 'mistral'
OLLAMA_DEFAULT_ENDPOINT = 'http://localhost:11434'

# Dataset Checker Configuration
CHECKER_CONFIG = {
    "min_image_size": (256, 256),
    "max_image_size": (1024, 1024),
    "allowed_formats": ["PNG", "JPEG"],
    "text_min_length": 10,
    "text_max_length": 500,
    "min_image_quality": 0.5,  # Minimum image quality score (0-1)
    "max_file_size": 10 * 1024 * 1024  # 10MB
} 