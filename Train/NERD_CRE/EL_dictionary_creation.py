import json
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from ...Utils.build_sentence_instances import build_sentence_instances
import os

models_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
train_data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Annotations/Train/json_format")
model_path = os.path.join(models_root, "SapBERT-from-PubMedBERT-fulltext")
repo_root = os.path.dirname(os.path.dirname(__file__))
dictionary_dir = os.path.join(repo_root, "dictionary_vectors")
os.makedirs(dictionary_dir, exist_ok=True)

print("Loading base SapBERT model.")
model = SentenceTransformer(model_path)


json_files = [
    #os.path.join(train_data_root, "train_bronze.json"), #OMMITTED THIS AS BETTER RESULTS ARE SEEN WITHOUT IT FOR THE DICTIONARIES
    os.path.join(train_data_root, "train_silver.json"),
    os.path.join(train_data_root, "train_gold.json"),
    os.path.join(train_data_root, "train_silver_2025.json"),
]

## START BY COMBINING JSON FILES FOR SENTENCE SPLITTER.
combined_documents = {}

for file_path in json_files:
    file_prefix = os.path.splitext(os.path.basename(file_path))[0]
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    if isinstance(raw_data, dict):
        for doc_id, doc in raw_data.items():
            safe_id = f"{file_prefix}_{doc_id}"
            combined_documents[safe_id] = doc
            
    elif isinstance(raw_data, list):
        for i, doc in enumerate(raw_data):
            safe_id = f"{file_prefix}_{i}"
            combined_documents[safe_id] = doc
            
    else:
        raise ValueError(f"Unsupported JSON structure in file: {file_path}")

## SPLIT THE DOCUMENTS INTO SENTENCE-LEVEL INSTANCES WITH ENTITY ANNOTATIONS
instances = build_sentence_instances(combined_documents)

extracted_pairs = []

print("Extracting Span+Sentence pairs.")

# Loop through each document
for doc_id, doc_data in instances.items():

    # Loop through sections (title, abstract, etc.)
    for section_name, sentences_list in doc_data.get("sections", {}).items():
        
        # Loop through each sentence
        for sentence_obj in sentences_list:
            sent_text = sentence_obj.get("sent_text", "")
            
            # Loop through entities in this sentence
            for entity in sentence_obj.get("entities", []):
                span = entity.get("text_span", "").lower()
                uri = entity.get("uri", "")
                
                # If entity has no span or URI, skip it, failsafe.
                if not span or not uri:
                    continue
                
                # Create a context string that combines the entity span and the sentence text
                context_string = f"{span} [SEP] {sent_text}"
                extracted_pairs.append({"context_string": context_string, "uri": uri})

## REMOVE DUPLICATES TO AVOID SENTENCES WITH THE SAME ENTITY MULTIPLE TIMES
print("Removing duplicates to avoid sentences with the same entity multiple times")
df = pd.DataFrame(extracted_pairs)

# Count how many times each (context_string, uri) combo appears
freq_df = df.groupby(["context_string", "uri"]).size().reset_index(name="count")

# Sort them so the highest count is at the top
freq_df = freq_df.sort_values(by=["context_string", "count"], ascending=[True, False])

# Drop duplicates, keeping ONLY the most frequent URI for that specific context
unique_dictionary = freq_df.drop_duplicates(subset=["context_string"], keep="first")

dict_contexts = unique_dictionary["context_string"].tolist()
dict_uris = unique_dictionary["uri"].tolist()

# ENCODE THE CONTEXT STRINGS INTO VECTORS
print("Encoding Contextual Strings into vectors.")
dictionary_vectors = model.encode(dict_contexts, convert_to_tensor=True, show_progress_bar=True)

print("Saving Contextual Vector Dictionary to disk.")
vectors_path = os.path.join(dictionary_dir, "sapbert_contextual_vectors_best.pt")
mapping_path = os.path.join(dictionary_dir, "sapbert_contextual_mapping_best.json")
torch.save(dictionary_vectors, vectors_path)

import json
mapping_data = {
    "uris": dict_uris,
    "contexts": dict_contexts
}
with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(mapping_data, f)

print(f"Saved vectors: {vectors_path}")
print(f"Saved mapping: {mapping_path}")