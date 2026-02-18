# src/config.py

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # None for auto detection
LOAD_IN_4BIT = True
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# The standard prompt format
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

SYSTEM_INSTRUCTION = "Extract all entity relationships from the following text and output them as a JSON list of triples."
