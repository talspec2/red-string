# src/data_utils.py
"""
Data processing utilities for the REBEL dataset.
Handles data extraction, formatting, and prompt generation for instruction tuning.
"""

import json
import torch
from tqdm import trange
from datasets import load_dataset
from .config import ALPACA_PROMPT

try:
    from sentence_transformers import SentenceTransformer, util
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    st_model = None


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

###
### TODO: INTEGRATE DEDUPLICATION INTO FRONT END
###
# def deduplicate_triplets(triplets, threshold=0.85):
#     """
#     Removes semantically redundant triplets from the model's output.
#     """
#     if not st_model:
#         print("Warning: sentence-transformers not installed. Skipping deduplication.")
#         return triplets

#     if not triplets:
#         return []

#     valid_triplets = [t for t in triplets if isinstance(t, dict)]

#     # Format triplets into strings for embedding
#     triplet_strs = [
#         f"{t.get('head', '')} {t.get('type', '')} {t.get('tail', '')}"
#         for t in valid_triplets
#     ]

#     if not triplet_strs:
#         return []

#     # Generate embeddings
#     embeddings = st_model.encode(triplet_strs, convert_to_tensor=True)

#     # Compute cosine similarities
#     cosine_scores = util.cos_sim(embeddings, embeddings)

#     unique_triplets = []
#     seen_indices = set()

#     for i in range(len(triplet_strs)):
#         if i in seen_indices:
#             continue

#         unique_triplets.append(valid_triplets[i])
#         seen_indices.add(i)

#         # Find all subsequent triplets that are semantically identical
#         for j in range(i + 1, len(triplet_strs)):
#             if cosine_scores[i][j].item() >= threshold:
#                 seen_indices.add(j)

#     return unique_triplets


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
