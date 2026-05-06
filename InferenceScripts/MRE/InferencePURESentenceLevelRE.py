import json
import os
import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

import nltk
from nltk.tokenize import sent_tokenize


#Setting roots
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))

#Helper function for resolving paths with environment variables or defaults
def _resolve_path(path_value: str) -> str:
    return path_value if os.path.isabs(path_value) else os.path.abspath(os.path.join(repo_root, path_value))

# Define paths
MODEL_PATH = _resolve_path(os.getenv("GETGUT_RE_MODEL_PATH", os.path.join(repo_root, "Models", "RE")))
DEV_JSON = _resolve_path(os.getenv("GETGUT_RE_INPUT_JSON", os.path.join(repo_root, "Predictions", "NER", "predictions_for_NERD_MRE.json")))
OUTPUT_FILE = _resolve_path(os.getenv("GETGUT_RE_OUTPUT_JSON", os.path.join(repo_root, "Predictions", "MRE", "predictions_for_CRE.json")))
SUBMISSION_OUTPUT_FILE = _resolve_path(
    os.getenv("GETGUT_RE_SUBMISSION_OUTPUT_JSON", os.path.join(repo_root, "Predictions", "MRE", "GetGut@AAU_T621_runID.json"))
)

print(f"Loading model from {MODEL_PATH}.")
# Error handling
if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        "RE model directory not found. "
        f"Looked at: {MODEL_PATH}. "
        "Set GETGUT_RE_MODEL_PATH to a valid best_model directory."
    )

if not os.path.isfile(DEV_JSON):
    raise FileNotFoundError(
        "RE input JSON not found. "
        f"Looked at: {DEV_JSON}. "
        "Set GETGUT_RE_INPUT_JSON to a valid file."
    )

# Tokenizer and model loading
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Load label mappings
with open(os.path.join(repo_root, "Utils", "processed_re_data",  "label2id.json"), "r") as f:
    label2id = json.load(f)

with open(os.path.join(repo_root, "Utils", "processed_re_data", "id2label.json"), "r") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

# Move model to device and set to eval mode
num_labels = len(label2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded")
print(f"label2id: {label2id}")
print(f"Device: {device}")

# Ensure sentence tokenization works in restricted HPC environments.
SENT_TOKENIZER_READY = False
for resource_name in ["punkt_tab", "punkt"]:
    try:
        nltk.download(resource_name, quiet=True)
    except Exception:
        pass

try:
    _ = sent_tokenize("Tokenizer check.")
    SENT_TOKENIZER_READY = True
except Exception as e:
    print(
        "[WARNING] NLTK sentence tokenizer resources are unavailable. "
        "Falling back to regex sentence splitting. "
        f"Reason: {e}"
    )

# HELPER FUNCTIONS FOR SENTENCE SPLITTING AND ENTITY ALIGNMENT
def get_sentence_boundaries(text: str) -> List[Tuple[int, int, str]]:
    """Split text into sentences with character offsets."""
    if SENT_TOKENIZER_READY:
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = []
    else:
        # Fallback splitter that keeps punctuation-boundary sentences.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    if not sentences:
        return [(0, len(text), text.strip())] if text.strip() else []
    
    sentence_bounds = []
    search_start = 0
    
    for sentence in sentences:
        start_pos = text.find(sentence, search_start)
        if start_pos == -1:
            continue
        
        end_pos = start_pos + len(sentence)
        sentence_text = sentence.strip()
        
        if sentence_text:
            sentence_bounds.append((start_pos, end_pos, sentence_text))
        
        search_start = end_pos
    
    if not sentence_bounds and text.strip():
        sentence_bounds.append((0, len(text), text.strip()))
    
    return sentence_bounds

# HELPER FUNCTION TO FIND WHICH SENTENCE AN ENTITY BELONGS TO
def find_entity_sentence(entity_start: int, entity_end: int, sentences: List[Tuple[int, int, str]]) -> Optional[int]:
    for sent_idx, (sent_start, sent_end, _) in enumerate(sentences):
        if sent_start <= entity_start and entity_end <= sent_end:
            return sent_idx
    
    # Fallback: assign to sentence containing start
    for sent_idx, (sent_start, sent_end, _) in enumerate(sentences):
        if sent_start <= entity_start < sent_end:
            return sent_idx
    
    return None

# HELPER FUNCTION TO REBASE ENTITY OFFSETS TO SENTENCE-RELATIVE OFFSETS
def rebase_offset_to_sentence(char_pos: int, sentence_start: int) -> int:
    return char_pos - sentence_start

# HELPER FUNCTION TO EXTRACT SPAN+SENTENCE PAIRS FROM PRE-PARSED INSTANCES
def inject_markers(text: str, sub_start: int, sub_end: int, obj_start: int, obj_end: int) -> str:
    if sub_start < obj_start:
        first = (sub_start, sub_end, "[E1]", "[/E1]")
        second = (obj_start, obj_end, "[E2]", "[/E2]")
    else:
        first = (obj_start, obj_end, "[E2]", "[/E2]")
        second = (sub_start, sub_end, "[E1]", "[/E1]")

    marked = (
        text[:first[0]] + 
        f"{first[2]} {text[first[0]:first[1]]} {first[3]}" + 
        text[first[1]:second[0]] + 
        f"{second[2]} {text[second[0]:second[1]]} {second[3]}" + 
        text[second[1]:]
    )
    
    return marked

# Extract span+sentence pairs from pre-parsed instances for inference
def predict_relation(marked_text: str) -> Tuple[str, float]:
    # Tokenize
    inputs = tokenizer(
        marked_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions and probabilities
    logits = outputs.logits.cpu().numpy()[0]
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps)
    pred_id = np.argmax(logits)
    pred_prob = probs[pred_id]
    pred_label = id2label[int(pred_id)]
    
    return pred_label, float(pred_prob)

# MAIN INFERENCE FUNCTION
def run_inference(dev_json_path: str) -> Dict:
    
    with open(dev_json_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    
    if isinstance(dev_data, dict):
        documents = dev_data
    else:
        documents = {str(i): doc for i, doc in enumerate(dev_data)}
    
    
    predictions = {}
    total_pairs = 0
    predicted_relations = 0
    no_relation_predictions = 0
    
    print("Running inference on sentence-level entity pairs.")
    
    for doc_idx, (doc_id, document) in enumerate(documents.items()):
        if not isinstance(document, dict):
            continue

        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        # Accept either nested metadata or top-level title/abstract.
        title_text = metadata.get("title", document.get("title", ""))
        abstract_text = metadata.get("abstract", document.get("abstract", document.get("text", "")))

        title_text = title_text if isinstance(title_text, str) else ""
        abstract_text = abstract_text if isinstance(abstract_text, str) else ""

        entities = document.get("entities", [])
        
        doc_predictions = {
            "title": title_text,
            "abstract": abstract_text,
            "mention_level_relations": []
        }
        
        # Process each location (title, abstract)
        for location in ["title", "abstract"]:
            section_text = title_text if location == "title" else abstract_text
            if not isinstance(section_text, str) or not section_text.strip():
                continue
            
            # Get sentence boundaries
            sentences = get_sentence_boundaries(section_text)
            
            # For each sentence, get entities and generate pairs
            for sent_idx, (sent_start, sent_end, sent_text) in enumerate(sentences):
                # Find entities in this sentence
                sent_entities = []
                for entity in entities:
                    if entity.get("location") != location:
                        continue
                    
                    ent_start = entity.get("start_idx")
                    ent_end = entity.get("end_idx")
                    
                    if sent_start <= ent_start and ent_end <= sent_end:
                        sent_offset_start = rebase_offset_to_sentence(ent_start, sent_start)
                        sent_offset_end = rebase_offset_to_sentence(ent_end, sent_start)
                        
                        sent_entities.append({
                            "entity": entity,
                            "doc_start": ent_start,
                            "doc_end": ent_end,
                            "sent_start_offset": sent_offset_start,
                            "sent_end_offset": sent_offset_end
                        })
                
                if len(sent_entities) < 2:
                    continue  # Need at least 2 entities for a pair
                
                # Generate ordered pairs
                for i, e1_info in enumerate(sent_entities):
                    for j, e2_info in enumerate(sent_entities):
                        if i == j:
                            continue
                        
                        total_pairs += 1
                        
                        e1 = e1_info["entity"]
                        e2 = e2_info["entity"]
                        
                        # Inject markers (using sentence-relative offsets)
                        marked_text = inject_markers(
                            sent_text,
                            e1_info["sent_start_offset"],
                            e1_info["sent_end_offset"],
                            e2_info["sent_start_offset"],
                            e2_info["sent_end_offset"]
                        )
                        
                        # Predict
                        pred_label, _confidence = predict_relation(marked_text)
                        
                        # Filter out "no_relation" (we only want predicted relations)
                        if pred_label != "NO_RELATION":
                            predicted_relations += 1
                            
                            relation = {
                                "subject_text_span": e1.get("text_span"),
                                "subject_label": e1.get("label"),
                                "predicate": pred_label.lower().replace("_", " "),
                                "object_text_span": e2.get("text_span"),
                                "object_label": e2.get("label")
                            }
                            
                            doc_predictions["mention_level_relations"].append(relation)
                        else:
                            no_relation_predictions += 1
        
        predictions[doc_id] = doc_predictions
        
        if (doc_idx + 1) % 5 == 0:
            print(f"Processed {doc_idx + 1}/{len(documents)} documents")
    
    print("\nInference Summary:")
    print(f"Total candidate pairs: {total_pairs}")
    print(f"Predicted relations (non-no_relation): {predicted_relations}")
    print(f"No-relation predictions: {no_relation_predictions}")
    
    return predictions


if __name__ == "__main__":
    # Run inference
    predictions = run_inference(DEV_JSON)
    
    # Save predictions
    print(f"\nSaving predictions to {OUTPUT_FILE}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    #
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"Input predictions saved")

    # Save submission-format output (only mention_level_relations per document)
    print(f"Saving submission predictions to {SUBMISSION_OUTPUT_FILE}...")

    submission_predictions = {
        doc_id: {
            "mention_level_relations": doc_data.get("mention_level_relations", [])
        }
        for doc_id, doc_data in predictions.items()
    }

    os.makedirs(os.path.dirname(SUBMISSION_OUTPUT_FILE), exist_ok=True)
    with open(SUBMISSION_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(submission_predictions, f, indent=2, ensure_ascii=False)

    print(f"Submission predictions saved!")
    
    
    print(f"\nInference complete!")
