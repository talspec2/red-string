# src/data_utils.py
"""
Data processing utilities for the REBEL dataset.
Handles data extraction, formatting, and prompt generation for instruction tuning.
"""

import json
from tqdm import trange
from datasets import load_dataset
from .config import ALPACA_PROMPT


def format_triplet(triplet) -> list[dict]:
    """
    Parses a raw triplet string into a structured dictionary format.

    Args:
        triplet (str): Raw string containing subject, object, and relation tags.

    Returns:
        list[dict]: A list of dictionaries, each containing 'head', 'type', and 'tail' keys.
    """
    triplets = []
    raw = triplet.strip().split("<triplet>")

    for s in raw:
        s = s.strip()
        if not s:
            continue

        try:
            first = s.split("<subj>")
            if len(first) != 2:
                continue

            subj = first[0].strip()
            rest = first[1].strip()

            second = rest.split("<obj>")
            if len(second) != 2:
                continue

            obj = second[0].strip()
            rel_type = second[1].strip()

            triplets.append({"head": subj, "type": rel_type, "tail": obj})
        except (IndexError, ValueError):
            continue

    return triplets

def process(instruction, amount=20000, split="train") -> list[dict]:
    """
    Loads and processes the REBEL dataset into an instruction-tuning format.

    Args:
        instruction (str): The system instruction to prepend to the examples.
        amount (int): The number of records to process from the dataset. Default is 20000.

    Returns:
        list[dict]: A list of formatted examples containing 'instruction', 'input', and 'output'.
    """
    # Load from the auto-converted parquet branch to bypass the deprecated script
    ds = load_dataset(
        "Babelscape/rebel-dataset",
        name="default",
        split=split,
        revision="refs/convert/parquet",
    )

    processed = []
    for i in trange(amount):
        triplet = format_triplet(ds["triplets"][i])
        context = ds["context"][i].strip()
        if not triplet or not context or len(triplet) > 5:
            continue
        processed.append(
            {
                "instruction": instruction,
                "input": context,
                "output": json.dumps(triplet),
            }
        )
    return processed


def prompt(examples, tokenizer) -> dict:
    """
    Applies the Alpaca prompt template to the processed examples.

    Args:
        examples (dict): A dictionary of lists containing 'instruction', 'input', and 'output'.
        tokenizer (PreTrainedTokenizer): The tokenizer used to append the EOS token.

    Returns:
        dict: A dictionary containing the formatted text strings under the 'text' key.
    """
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    EOS_TOKEN = tokenizer.eos_token
    for instruction, text, output in zip(instructions, inputs, outputs):
        text = ALPACA_PROMPT.format(instruction, text, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }
