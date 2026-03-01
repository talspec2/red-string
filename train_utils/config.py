# src/config.py
"""
Configuration module for model training and inference.
Defines model parameters, LoRA configuration, and prompt templates.
"""

MODEL_NAME = "unsloth/Llama-3.2-3B"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # None for auto detection
LOAD_IN_4BIT = False
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Standard Alpaca instruction-tuning prompt format
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

SYSTEM_INSTRUCTION = "Extract all entity relationships from the following text and output them as a JSON list of triples."
