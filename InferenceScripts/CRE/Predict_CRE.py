import json
import torch
import os
import nltk
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer, util

# Preparing NLTK for sentence tokenization
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# LOAD THE MODELS FROM ROOT.
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
models_root = os.path.join(repo_root, "Models", "NERD_CRE")
model_path = os.path.join(models_root, "SapBERT-from-PubMedBERT-fulltext")


model = SentenceTransformer(model_path)
model.max_seq_length = 128

vector_candidates = os.path.join(repo_root, "Utils", "sapbert_contextual_vectors_best.pt")
mapping_candidates = os.path.join(repo_root, "Utils", "sapbert_contextual_mapping_best.json")

# ERROR HANDLING FOR DICTIONARY FILES
vectors_path = vector_candidates if os.path.isfile(vector_candidates) else None
mapping_path = mapping_candidates if os.path.isfile(mapping_candidates) else None

# LOAD THE DICTIONARIES
print("Loading dictionaries")
dictionary_vectors = torch.load(vectors_path, weights_only=True)

with open(mapping_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)
    dict_contexts = metadata["contexts"] 
    dict_uris = metadata["uris"]

# CHECK PRINT FOR LOADED DICTIONARY
print(f"Loaded {len(dict_contexts)} dictionary entries.")

# EXTRACT SPANS BY [SEP] IN SENTENCE SPLITTING FOR 2ND SPAN DICT.
print("\nExtracting span for dict.")
dict_spans = []
for context in dict_contexts:
    if "[SEP]" in context:
        span = context.split("[SEP]")[0].strip()
    else:
        span = context
    dict_spans.append(span)

print("Encoding dictionary spans...")
dict_span_vectors = model.encode(dict_spans, convert_to_tensor=True, show_progress_bar=True)

# HELPER FUNCTION FOR LOADING THE JSON FROM NER MODEL
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

# CALL OF HELPER FUNCTION
input_file = os.path.join(repo_root, "Predictions", "MRE", "predictions_for_CRE.json")

# ERROR HANDLING
if input_file is None:
    raise FileNotFoundError(
        "Could not find CRE input JSON. "
        "Set GETGUT_CRE_INPUT_JSON or provide one of: "
        f"{os.path.join(repo_root, 'Predictions', 'MRE', 'predictions_for_CRE.json')}"
    )

print(f"\nLoading {input_file}")

with open(input_file, "r", encoding="utf-8") as f:
    input_data = json.load(f)

# HELPER FUNCTION TO PREDICT URI FOR A GIVEN SPAN
def predict_uri_for_span(span, abstract_text, sentences):
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
    
    # Get the best matching URI
    best_score, best_idx = hybrid_scores.topk(1)
    predicted_uri = dict_uris[best_idx[0].item()]
    
    return predicted_uri

# PREDICT CONCEPT-LEVEL RELATIONS
predictions = {}

print("Predicting concept-level relations for C-RE.")
total_relations = 0

# Iterate through documents and their mention-level relations to predict concept-level relations
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

# OUTPUT THE PREDICTIONS TO JSON

output_file = os.path.join(repo_root, "Predictions", "CRE", "GetGut@AAU_T622_runID.json")
print(f"\nSaving concept-level relations to {output_file}.")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2, ensure_ascii=False)

print(f"Predicted concept-level relations for {total_relations} relations.")
print(f"Output saved to: {output_file}")
