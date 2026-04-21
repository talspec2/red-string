import json
import requests
from tqdm import tqdm
from datasets import load_dataset
from train_utils.data_utils import format_triplet
from train_utils.config import SYSTEM_INSTRUCTION, ALPACA_PROMPT

# Import sentence_transformers for semantic matching
try:
    from sentence_transformers import SentenceTransformer, util

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    print(
        "Error: sentence-transformers is required for semantic similarity evaluation."
    )
    print("Please run: pip install sentence-transformers")
    exit()

# --- Configuration ---
SERVER_URL = (
    "https://printing-bookmarks-literary-measure.trycloudflare.com/v1/completions"
)
EVAL_SAMPLES = 500


def load_eval_data(num_samples):
    ds = load_dataset(
        "Babelscape/rebel-dataset",
        name="default",
        split="test",
        revision="refs/convert/parquet",
    )

    eval_data = []
    idx = 0

    with tqdm(total=num_samples, desc="Loading evaluation data") as pbar:
        while len(eval_data) < num_samples and idx < len(ds["triplets"]):
            triplet = format_triplet(ds["triplets"][idx])
            context = ds["context"][idx].strip()

            if triplet and context and len(triplet) <= 5:
                eval_data.append({"input": context, "ground_truth": triplet})
                pbar.update(1)
            idx += 1

    return eval_data


def query_server(prompt_text):
    payload = {
        "prompt": prompt_text,
        "max_tokens": 512,
        "temperature": 0.1,
        "stop": ["</s>", "<|eot_id|>", "###"],
    }

    try:
        response = requests.post(SERVER_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return "[]"


# --- Evaluation Alignment Functions ---


def evaluate_string_alignment(pred_list, truth_list, match_type="exact"):
    """Calculates TP, FP, FN using either strict exact matching or partial substring matching."""
    matched_truth = set()
    tp = 0

    valid_preds = [p for p in pred_list if isinstance(p, dict)]
    valid_truths = [t for t in truth_list if isinstance(t, dict)]

    for p in valid_preds:
        best_match_idx = -1
        h1, r1, t1 = (
            str(p.get("head", "")).lower(),
            str(p.get("type", "")).lower(),
            str(p.get("tail", "")).lower(),
        )

        for i, t in enumerate(valid_truths):
            if i in matched_truth:
                continue

            h2, r2, t2 = (
                str(t.get("head", "")).lower(),
                str(t.get("type", "")).lower(),
                str(t.get("tail", "")).lower(),
            )

            if match_type == "exact":
                if (h1 == h2) and (r1 == r2) and (t1 == t2):
                    best_match_idx = i
                    break
            elif match_type == "partial":
                # For partial, relation must overlap, and entities must be substrings of each other
                if not (h1 and r1 and t1 and h2 and r2 and t2):
                    continue
                rel_match = (r1 in r2) or (r2 in r1)
                head_match = (h1 in h2) or (h2 in h1)
                tail_match = (t1 in t2) or (t2 in t1)

                if rel_match and head_match and tail_match:
                    best_match_idx = i
                    break

        if best_match_idx != -1:
            tp += 1
            matched_truth.add(best_match_idx)

    fp = len(valid_preds) - tp
    fn = len(valid_truths) - tp
    return tp, fp, fn


def evaluate_semantic_alignment(pred_list, truth_list, threshold=0.80):
    """Calculates TP, FP, FN using cosine similarity of sentence transformer embeddings."""
    valid_preds = [p for p in pred_list if isinstance(p, dict)]
    valid_truths = [t for t in truth_list if isinstance(t, dict)]

    if not valid_preds or not valid_truths:
        return 0, len(valid_preds), len(valid_truths)

    # Format triples into single strings for semantic encoding
    pred_strs = [
        f"{p.get('head', '')} {p.get('type', '')} {p.get('tail', '')}"
        for p in valid_preds
    ]
    truth_strs = [
        f"{t.get('head', '')} {t.get('type', '')} {t.get('tail', '')}"
        for t in valid_truths
    ]

    pred_embs = st_model.encode(pred_strs, convert_to_tensor=True)
    truth_embs = st_model.encode(truth_strs, convert_to_tensor=True)

    # Generate an NxM matrix of similarity scores
    cosine_scores = util.cos_sim(pred_embs, truth_embs)

    tp = 0
    matched_truth = set()

    # Greedy match: lock in the highest scoring pairs above the threshold
    for i in range(len(valid_preds)):
        best_score = -1
        best_idx = -1
        for j in range(len(valid_truths)):
            if j in matched_truth:
                continue
            score = cosine_scores[i][j].item()
            if score > best_score and score >= threshold:
                best_score = score
                best_idx = j

        if best_idx != -1:
            tp += 1
            matched_truth.add(best_idx)

    fp = len(valid_preds) - tp
    fn = len(valid_truths) - tp
    return tp, fp, fn


def calc_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1


def evaluate():
    test_data = load_eval_data(EVAL_SAMPLES)

    metrics = {
        "exact": {"tp": 0, "fp": 0, "fn": 0},
        "partial": {"tp": 0, "fp": 0, "fn": 0},
        "semantic": {"tp": 0, "fp": 0, "fn": 0},
    }

    total_entities_generated = 0
    hallucinated_entities = 0

    for example in tqdm(test_data, desc="Evaluating"):
        full_prompt = ALPACA_PROMPT.format(
            SYSTEM_INSTRUCTION, 
            example["input"],
            ""
        )
        response_text = query_server(full_prompt)

        try:
            pred_list = json.loads(response_text)
            if not isinstance(pred_list, list):
                pred_list = []
        except json.JSONDecodeError:
            pred_list = []

        truth_list = example["ground_truth"]

        # exact match
        e_tp, e_fp, e_fn = evaluate_string_alignment(
            pred_list, truth_list, match_type="exact"
        )
        metrics["exact"]["tp"] += e_tp
        metrics["exact"]["fp"] += e_fp
        metrics["exact"]["fn"] += e_fn

        # partial string matching
        p_tp, p_fp, p_fn = evaluate_string_alignment(
            pred_list, truth_list, match_type="partial"
        )
        metrics["partial"]["tp"] += p_tp
        metrics["partial"]["fp"] += p_fp
        metrics["partial"]["fn"] += p_fn

        # semantic similarity match (same threshold as frontend)
        s_tp, s_fp, s_fn = evaluate_semantic_alignment(
            pred_list, truth_list, threshold=0.85
        )
        metrics["semantic"]["tp"] += s_tp
        metrics["semantic"]["fp"] += s_fp
        metrics["semantic"]["fn"] += s_fn

        # calculate hallucinations
        input_text_lower = example["input"].lower()
        for triplet in pred_list:
            if not isinstance(triplet, dict):
                continue

            head = str(triplet.get("head", "")).lower()
            tail = str(triplet.get("tail", "")).lower()

            if head:
                total_entities_generated += 1
                if head not in input_text_lower:
                    hallucinated_entities += 1
            if tail:
                total_entities_generated += 1
                if tail not in input_text_lower:
                    hallucinated_entities += 1

    hallucination_rate = (
        hallucinated_entities / total_entities_generated
        if total_entities_generated > 0
        else 0
    )

    # Format Results
    output_lines = []
    output_lines.append("=" * 40)
    for eval_type in ["exact", "partial", "semantic"]:
        p, r, f1 = calc_metrics(
            metrics[eval_type]["tp"], metrics[eval_type]["fp"], metrics[eval_type]["fn"]
        )
        output_lines.append(f"--- {eval_type.capitalize()} Match Results ---")
        output_lines.append(f"Precision:         {p:.4f}")
        output_lines.append(f"Recall:            {r:.4f}")
        output_lines.append(f"F1-Score:          {f1:.4f}\n")

    output_lines.append("--- Grounding ---")
    output_lines.append(f"Hallucination Rate: {hallucination_rate:.2%}")
    output_lines.append("=" * 40)
    
    final_output = "\n".join(output_lines)
    
    # Print to console
    print("\n" + final_output)
    
    # Export to text file
    with open("results1.txt", "w", encoding="utf-8") as f:
        f.write(final_output)
    print("\nResults successfully exported to results.txt")


if __name__ == "__main__":
    evaluate()