import json
import torch
import os
import nltk
from pathlib import Path
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
dictionary_dir = repo_root / "dictionary_vectors"
vector_candidates = [
    dictionary_dir / "sapbert_contextual_vectors_best.pt",
    repo_root / "sapbert_contextual_vectors_best.pt",
    Path(__file__).parent / "sapbert_contextual_vectors_best.pt",
]
mapping_candidates = [
    dictionary_dir / "sapbert_contextual_mapping_best.json",
    repo_root / "sapbert_contextual_mapping_best.json",
    Path(__file__).parent / "sapbert_contextual_mapping_best.json",
]

vectors_path = next((p for p in vector_candidates if p.exists()), None)
mapping_path = next((p for p in mapping_candidates if p.exists()), None)

if vectors_path is None or mapping_path is None:
    raise FileNotFoundError(
        "Could not find contextual dictionary files. "
        f"Checked vectors: {[str(p) for p in vector_candidates]} | "
        f"mapping: {[str(p) for p in mapping_candidates]}"
    )

dictionary_vectors = torch.load(vectors_path, weights_only=True)

with open(mapping_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)
    dict_contexts = metadata["contexts"]
    dict_uris = metadata["uris"]

print(f"Using vectors: {vectors_path}")
print(f"Using mapping: {mapping_path}")

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
input_file = repo_root / "dev_predictions_nerd.json"
print(f"\nLoading {input_file}...")

with open(input_file, "r", encoding="utf-8") as f:
    input_data = json.load(f)

# --- 4. PREDICT URIS FOR EACH ENTITY ---
predictions = {}

print("Predicting URIs for entities...")
total_entities = 0

for doc_id, document in input_data.items():
    abstract_text = document.get("text", document.get("abstract", ""))
    sentences = sent_tokenize(abstract_text) if abstract_text else []
    
    # Initialize document in predictions
    predictions[doc_id] = {
        "entities": []
    }
    
    if "entities" in document:
        for entity in document["entities"]:
            if "text_span" in entity:
                span = entity["text_span"]
                total_entities += 1
                
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
                
                # Create entity entry with only essential fields (no metadata)
                entity_with_prediction = {
                    "start_idx": entity.get("start_idx"),
                    "end_idx": entity.get("end_idx"),
                    "location": entity.get("location"),
                    "text_span": entity["text_span"],
                    "label": entity["label"],
                    "uri": predicted_uri
                }
                
                predictions[doc_id]["entities"].append(entity_with_prediction)

# --- 5. SAVE PREDICTIONS TO JSON ---
output_file = repo_root / "NERD_Predictions.json"
print(f"\nSaving predictions to {output_file}...")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2, ensure_ascii=False)

print(f"✓ Done! Predicted URIs for {total_entities} entities.")
print(f"Output saved to: {output_file}")
