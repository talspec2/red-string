# src/data_utils.py
import json
from tqdm import trange
from datasets import load_dataset
from .config import ALPACA_PROMPT


def format(triplet):
    """
    Converts: "<triplet> Subject <subj> Object <obj> Relation"
    Into: [{"head": "Subject", "tail": "Object", "type": "Relation"}]
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
        except Exception:
            continue

    return triplets


def process(prompt, amount=20000):
    # Load from the auto-converted parquet branch to bypass the deprecated script
    ds = load_dataset(
        "Babelscape/rebel-dataset",
        name="default",
        split="train",
        revision="refs/convert/parquet",
    )

    processed = []
    for i in trange(amount):
        triplet = format_triplets(ds["triplets"][i])
        context = ds["context"][i].strip()
        if not triplet or not context or len(triplet) > 5:
            continue
        processed.append(
            {"instruction": prompt, "input": context, "output": json.dumps(triplet)}
        )
    return processed


def prompt(examples, tokenizer):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    EOS_TOKEN = tokenizer.eos_token
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = ALPACA_PROMPT.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }
