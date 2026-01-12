"""
Configuration settings for the sycophancy steering project.
"""

# Model configuration
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
MODEL_SIZE = "7b"
LOAD_IN_8BIT = True

# Data configuration
TRAIN_DATA_PATH = "data/generate_dataset.json"
TEST_DATA_PATH = "data/test_dataset_ab.json"
TRAIN_DATA_URL = "https://raw.githubusercontent.com/nrimsky/CAA/main/datasets/generate/sycophancy/generate_dataset.json"
TEST_DATA_URL = "https://raw.githubusercontent.com/nrimsky/CAA/main/datasets/test/sycophancy/test_dataset_ab.json"

# Layer selection configuration
LAYER_RANGE_START = 8
LAYER_RANGE_END = 20
DEFAULT_BEST_LAYER = 14  # Based on Llama-2-7B research

# Training configuration
TRAIN_VAL_SPLIT = 0.3
RANDOM_SEED = 42

# Evaluation configuration
BATCH_SIZE = 8
MAX_LENGTH = 512

# Steering configuration
DEFAULT_MULTIPLIERS = [-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0]

# Output paths
RESULTS_DIR = "results"
STEERING_VECTOR_DIR = "steering_vectors"
PLOTS_DIR = "results/plots"

# Llama-2 prompt template
B_INST = "[INST]"
E_INST = "[/INST]"
SYS_PROMPT = "You are a helpful, honest assistant."
B_SYS = "<<SYS>>"
E_SYS = "<</SYS>>"
BOS_TOKEN = "<s>"
PROMPT_TEMPLATE = f"{BOS_TOKEN} {B_INST} \n{B_SYS}{SYS_PROMPT}{E_SYS}\n\n {{question}} {E_INST} {{answer}}"