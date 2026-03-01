# src/model_utils.py
"""
Model initialization and configuration utilities.
Handles the loading of the base model and application of LoRA adapters.
"""

from unsloth import FastLanguageModel
from .config import MAX_SEQ_LENGTH, DTYPE, LOAD_IN_4BIT, MODEL_NAME


def load_model() -> tuple:
    """
    Loads the specified language model and configures it with PEFT (Parameter-Efficient Fine-Tuning).

    Returns:
        tuple: A tuple containing the configured PEFT model and its tokenizer.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    peft = FastLanguageModel.get_peft_model(
        model,
        r=32,  # LoRA rank dimension
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0,  # 0 is optimized
        bias="none",  # none is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return peft, tokenizer
