# src/inference.py
from .config import ALPACA_PROMPT


def extract_graph(
    model,
    tokenizer,
    text,
    instruction="Extract all entity relationships from the following text and output them as a JSON list of triples.",
):

    # Format the prompt exactly like training
    prompt = ALPACA_PROMPT.format(instruction, text, "")

    # tokenize
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Generate!
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,  # Allow enough space for JSON
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the output (skip the prompt part)
    decoded = tokenizer.batch_decode(outputs)[0]

    # Extract just the JSON part (after "### Response:")
    try:
        response = decoded.split("### Response:")[1].strip()
        # Remove the EOS token if present
        response = response.replace(tokenizer.eos_token, "")
        return response
    except:
        return decoded  # Fallback if something weird happens
