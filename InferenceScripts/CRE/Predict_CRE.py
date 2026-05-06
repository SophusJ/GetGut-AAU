import json
import torch
import os
import nltk
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer, util

# Ensure NLTK is ready for sentence extraction
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# --- 1. LOAD MODEL ---
models_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
model_path = os.path.join(models_root, "SapBERT-from-PubMedBERT-fulltext")

print("Loading SapBERT model...")
model = SentenceTransformer(model_path)
model.max_seq_length = 128

# --- 2. LOAD SAVED CONTEXTUAL DICTIONARY ---
print("Loading saved contextual vectors and metadata...")
repo_root = Path(__file__).parent.parent
dictionary_vectors = torch.load(repo_root / "dictionary_vectors" / "sapbert_contextual_vectors.pt", weights_only=True)

with open(repo_root / "dictionary_vectors" / "sapbert_contextual_mapping.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
    dict_contexts = metadata["contexts"] 
    dict_uris = metadata["uris"]

print(f"Loaded {len(dict_contexts)} dictionary entries.")

# --- 2b. EXTRACT AND ENCODE SPAN-ONLY VECTORS ---
print("\nExtracting span-only portions from dictionary contexts...")
dict_spans = []
for context in dict_contexts:
    # Split on [SEP] and take only the part before it
    if "[SEP]" in context:
        span = context.split("[SEP]")[0].strip()
    else:
        span = context
    dict_spans.append(span)

print("Encoding span-only dictionary vectors...")
dict_span_vectors = model.encode(dict_spans, convert_to_tensor=True, show_progress_bar=True)

# --- 3. LOAD INPUT JSON ---
def resolve_existing_path(env_name: str, candidates: list[Path]) -> Optional[Path]:
    env_val = os.getenv(env_name)
    if env_val:
        env_path = Path(env_val)
        if not env_path.is_absolute():
            env_path = repo_root / env_path
        return env_path if env_path.exists() else None

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


input_file = resolve_existing_path(
    "GETGUT_CRE_INPUT_JSON",
    [
        repo_root / "dev_predictions_CRE.json",
        repo_root / "predicted_relations_dev.json",
    ],
)

if input_file is None:
    raise FileNotFoundError(
        "Could not find CRE input JSON. "
        "Set GETGUT_CRE_INPUT_JSON or provide one of: "
        f"{repo_root / 'dev_predictions_CRE.json'}, {repo_root / 'predicted_relations_dev.json'}"
    )

print(f"\nLoading {input_file}...")

with open(input_file, "r", encoding="utf-8") as f:
    input_data = json.load(f)

# --- 4. HELPER FUNCTION TO PREDICT URI FOR A TEXT SPAN ---
def predict_uri_for_span(span, abstract_text, sentences):
    """Predict URI for a given text span using hybrid scoring."""
    # Find the exact host sentence
    host_sentence = abstract_text
    for sent in sentences:
        if span.lower() in sent.lower():
            host_sentence = sent
            break
    
    # Build the contextual query
    test_query = f"{span.lower()} [SEP] {host_sentence}"
    
    # Encode both span and context
    span_vector = model.encode(span.lower(), convert_to_tensor=True)
    context_vector = model.encode(test_query, convert_to_tensor=True)
    
    # Compute scores against dictionary
    span_scores = util.cos_sim(span_vector, dict_span_vectors)[0]
    context_scores = util.cos_sim(context_vector, dictionary_vectors)[0]
    
    # Hybrid scoring (weighted combination)
    weight_context = 0.15
    weight_span = 0.85
    hybrid_scores = weight_context * context_scores + weight_span * span_scores
    
    # Get top-1 match
    best_score, best_idx = hybrid_scores.topk(1)
    predicted_uri = dict_uris[best_idx[0].item()]
    
    return predicted_uri

# --- 5. PREDICT URIS FOR EACH RELATION ---
predictions = {}

print("Predicting concept-level relations for C-RE...")
total_relations = 0

for doc_id, document in input_data.items():
    abstract_text = document.get("text", document.get("abstract", ""))
    sentences = sent_tokenize(abstract_text) if abstract_text else []
    
    # Initialize document in predictions
    predictions[doc_id] = {
        "concept_level_relations": []
    }
    
    relation_candidates = document.get("mention_level_relations") or document.get("predicted_relations") or []

    if relation_candidates:
        for relation in relation_candidates:
            total_relations += 1
            
            # Predict URIs for subject and object
            subject_span = relation.get("subject_text_span", "")
            object_span = relation.get("object_text_span", "")

            if not subject_span or not object_span:
                continue
            
            subject_uri = predict_uri_for_span(subject_span, abstract_text, sentences)
            object_uri = predict_uri_for_span(object_span, abstract_text, sentences)
            
            # Create concept-level relation with predicted URIs
            concept_relation = {
                "subject_uri": subject_uri,
                "subject_label": relation.get("subject_label"),
                "predicate": relation.get("predicate"),
                "object_uri": object_uri,
                "object_label": relation.get("object_label")
            }
            
            predictions[doc_id]["concept_level_relations"].append(concept_relation)

# --- 6. SAVE PREDICTIONS TO JSON ---
output_env = os.getenv("GETGUT_CRE_OUTPUT_JSON")
if output_env:
    output_file = Path(output_env)
    if not output_file.is_absolute():
        output_file = repo_root / output_file
else:
    output_file = repo_root / "CRE_Predictions.json"

output_file.parent.mkdir(parents=True, exist_ok=True)
print(f"\nSaving concept-level relations to {output_file}...")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2, ensure_ascii=False)

print(f"✓ Done! Predicted concept-level relations for {total_relations} relations.")
print(f"Output saved to: {output_file}")
