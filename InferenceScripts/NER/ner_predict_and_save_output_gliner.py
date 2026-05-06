import json
from transformers import AutoTokenizer
from gliner import GLiNER
import os
import torch
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)


def _resolve_path(path_value: str) -> str:
    """Resolve relative paths against repository root for stable pipeline runs."""
    return path_value if os.path.isabs(path_value) else os.path.abspath(os.path.join(repo_root, path_value))


default_model_candidates = [
    os.path.join(repo_root, "gliner_biomed_funnel", "gold", "final_model"),
    os.path.join(script_dir, "gliner_biomed_funnel", "gold", "final_model"),
]
default_model_path = next((p for p in default_model_candidates if os.path.isdir(p)), default_model_candidates[0])
default_input_json_path = os.path.join(repo_root, "Annotations", "Dev", "json_format", "dev.json")
default_output_json_path = os.path.join(repo_root, "dev_predictions.json")
default_output_nerd_json_path = os.path.join(repo_root, "dev_predictions_nerd.json")

model_path = _resolve_path(os.getenv("GETGUT_NER_MODEL_PATH", default_model_path))
input_json_path = _resolve_path(os.getenv("GETGUT_NER_INPUT_JSON", default_input_json_path))
output_json_path = _resolve_path(os.getenv("GETGUT_NER_OUTPUT_JSON", default_output_json_path))
output_nerd_json_path = _resolve_path(os.getenv("GETGUT_NERD_OUTPUT_JSON", default_output_nerd_json_path))

if not os.path.isdir(model_path):
    raise FileNotFoundError(
        "NER model directory not found. "
        f"Looked at: {model_path}. "
        "Set GETGUT_NER_MODEL_PATH to a valid checkpoint directory."
    )

if not os.path.isfile(input_json_path):
    raise FileNotFoundError(
        "NER input JSON not found. "
        f"Looked at: {input_json_path}. "
        "Set GETGUT_NER_INPUT_JSON to a valid file."
    )

os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
os.makedirs(os.path.dirname(output_nerd_json_path), exist_ok=True)

ENTITY_TYPES = [
    "ANATOMICAL LOCATION", "ANIMAL", "BIOMEDICAL TECHNIQUE", "BACTERIA",
    "CHEMICAL", "DIETARY SUPPLEMENT", "DDF", "DRUG", "FOOD",
    "GENE", "HUMAN", "MICROBIOME", "STATISTICAL TECHNIQUE"
]

# --- SLIDING WINDOW HELPER ---
def process_long_text_with_windows(text, ner_model, tokenizer, labels, window_size=450, overlap=50):
    """
    Process text longer than max_seq_length using sliding windows.
    Returns entities with correct indices relative to the original text.
    Optimized: Pre-calculate character offsets to avoid O(n²) decode operations.
    """
    all_entities = []
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_seq_length:
        predictions = ner_model.predict_entities(text, labels)
        for pred in predictions:
            text_span = text[pred["start"]:pred["end"]].strip()
            all_entities.append({
                "start_idx": pred["start"],
                "end_idx": pred["end"] - 1,
                "text_span": text_span,
                "label": pred["label"].lower().replace("_", " ").replace("ddf", "DDF")
            })
        return all_entities

    step = window_size - overlap

    # Pre-calculate character offsets for all windows (O(n) instead of O(n²))
    char_offsets = []
    for i in range(0, len(tokens), step):
        if i == 0:
            char_offsets.append(0)
        else:
            # Only decode once per window position, incrementally
            window_tokens = tokens[:i]
            char_offset = len(tokenizer.decode(window_tokens, skip_special_tokens=True))
            char_offsets.append(char_offset)

    for idx, i in enumerate(range(0, len(tokens), step)):
        window_tokens = tokens[i:i + window_size]
        window_text = tokenizer.decode(window_tokens, skip_special_tokens=True)

        # Use pre-calculated offset instead of recalculating
        char_offset = char_offsets[idx]

        predictions = ner_model.predict_entities(window_text, labels)

        for pred in predictions:
            start_idx = pred["start"] + char_offset
            end_idx = pred["end"] - 1 + char_offset
            text_span = text[start_idx:end_idx + 1].strip()
            all_entities.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "text_span": text_span,
                "label": pred["label"].lower().replace("_", " ").replace("ddf", "DDF")
            })

        if i + window_size >= len(tokens):
            break

    return all_entities

print("Loading model and tokenizer...")
model = GLiNER.from_pretrained(model_path)
tokenizer = getattr(model, "tokenizer", None)
if tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

# Determine device with CUDA fallback
device = "cpu"
if torch.cuda.is_available():
    try:
        print("Attempting to use CUDA GPU...")
        device = "cuda"
        model.to(device)
        print("Successfully loaded model on CUDA GPU")
    except RuntimeError as e:
        if "uncorrectable ECC error" in str(e) or "CUDA error" in str(e):
            print(f"CUDA error detected: {e}")
            print("Falling back to CPU device...")
            device = "cpu"
            model.to(device)
            # Clear CUDA cache to prevent memory issues
            torch.cuda.empty_cache()
        else:
            raise
else:
    print("CUDA not available, using CPU device...")

print(f"Using device: {device}")

# Get max sequence length from tokenizer (cap at 512 for stability)
max_seq_length = min(getattr(tokenizer, "model_max_length", 512), 512)

print("Loading input data...\n")
with open(input_json_path, 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

print(f"Processing {len(dev_data)} documents...")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

## OUTPUT STRUCTURE
output_data = {}
output_nerd_data = {}

start_time = time.time()

for doc_idx, (doc_id, doc_content) in enumerate(dev_data.items(), 1): ## FOR EACH DOCUMENT
    doc_start_time = time.time()
    metadata = doc_content.get("metadata", {}) ## GET METADATA
    
    # Extract title and abstract
    title = metadata.get("title", "").strip()
    abstract = metadata.get("abstract", "").strip()

    fields_to_process = ["title", "abstract"]  #GRAB TITLE AND ABSTRACT FOR ENTITY SEARCH
    entities = []
    
    for field_name in fields_to_process:
        field_text = metadata.get(field_name, "").strip()
        
        if not field_text:
            continue

        field_start_time = time.time()
        print(f"[{doc_idx}/{len(dev_data)}] Processing {doc_id} - {field_name}: {field_text[:60]}...", end=" ", flush=True)

        # Use sliding window for long texts, direct processing for short texts
        predictions = process_long_text_with_windows(field_text, model, tokenizer, ENTITY_TYPES)

        field_elapsed = time.time() - field_start_time
        print(f"({len(predictions)} entities, {field_elapsed:.2f}s)")

        if predictions:
            for pred in predictions:
                entity = {
                    "start_idx": pred["start_idx"],
                    "end_idx": pred["end_idx"],
                    "location": field_name,
                    "text_span": pred["text_span"],
                    "label": pred["label"]
                }
                entities.append(entity)
    
    doc_elapsed = time.time() - doc_start_time
    print(f"  → Document complete in {doc_elapsed:.2f}s\n")

    ## POPULATE OUTPUT STRUCTURE
    output_data[doc_id] = {
        "entities": entities
    }
    
    ## POPULATE NERD OUTPUT STRUCTURE (with title, abstract, and entities)
    output_nerd_data[doc_id] = {
        "metadata": {
            "title": title,
            "abstract": abstract
        },
        "entities": entities
    }

total_elapsed = time.time() - start_time
print(f"\nProcessing complete!")
print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)")
print(f"Average per document: {total_elapsed/len(dev_data):.2f}s")

print(f"\nSaving predictions to {output_json_path}...")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Saving NERD predictions to {output_nerd_json_path}...")
with open(output_nerd_json_path, 'w', encoding='utf-8') as f:
    json.dump(output_nerd_data, f, indent=2, ensure_ascii=False)
