import json
import os
from typing import Any, Dict, List, Tuple

from datasets import Dataset, concatenate_datasets

from BuildPURESentenceLevelRE import (
    build_relation_lookup,
    build_re_dataset,
    build_sentence_instances,
)


ROOT_DIR = os.path.dirname(os.path.dirname(str(__file__)))
TRAIN_JSON_DIR = os.path.join(ROOT_DIR, "Annotations", "Train", "json_format")
DEV_JSON_DIR = os.path.join(ROOT_DIR, "Annotations", "Dev", "json_format")
MODELS_ROOT = os.path.join(ROOT_DIR, "models")

BASE_MODEL_PATH = os.path.join(MODELS_ROOT, "BiomedNLP-BiomedBERT-large-uncased-abstract")
SPECIAL_MARKER_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]

DEFAULT_DATASET_SPECS = [
    (os.path.join(TRAIN_JSON_DIR, "train_bronze.json"), 0.75),
    (os.path.join(TRAIN_JSON_DIR, "train_silver.json"), 1.0),
    (os.path.join(TRAIN_JSON_DIR, "train_silver_2025.json"), 1.0),
    (os.path.join(TRAIN_JSON_DIR, "train_gold.json"), 1.25),
]
DEFAULT_EVAL_JSON_PATH = os.path.join(DEV_JSON_DIR, "dev.json")


def normalize_label(raw_label: str) -> str:
    return str(raw_label).upper().replace(" ", "_")


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_documents(json_file_path: str) -> Any:
    raw_data = read_json(json_file_path)
    if isinstance(raw_data, dict):
        return raw_data
    if isinstance(raw_data, list):
        return {str(i): doc for i, doc in enumerate(raw_data)}
    raise ValueError(f"Unsupported JSON structure in {json_file_path}")


def extract_predicates(json_file_path: str) -> List[str]:
    documents = load_documents(json_file_path)
    labels = set()

    for doc in documents.values():
        if not isinstance(doc, dict):
            continue
        for relation in doc.get("relations", []):
            if not isinstance(relation, dict):
                continue
            predicate = relation.get("predicate")
            if predicate:
                labels.add(normalize_label(predicate))

    return sorted(labels)


def build_shared_re_label_mapping(json_paths: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    all_labels = {"NO_RELATION"}

    for path in json_paths:
        for predicate in extract_predicates(path):
            all_labels.add(predicate)

    sorted_labels = ["NO_RELATION"] + sorted([l for l in all_labels if l != "NO_RELATION"])
    label2id = {label: idx for idx, label in enumerate(sorted_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def build_re_dataset_from_json(
    json_file_path: str,
    label2id: Dict[str, int],
    sample_weight: float,
    positive_pairs_only: bool = False,
    negative_to_positive_ratio: float = 3.0,
    max_negatives_when_no_positive: int = 3,
    sampling_seed: int = 42,
) -> Dataset:
    documents = load_documents(json_file_path)
    instances = build_sentence_instances(documents)
    relation_lookup = build_relation_lookup(documents)

    marked_sentences, labels_numeric, local_label2id, _ = build_re_dataset(
        instances,
        relation_lookup,
        positive_pairs_only=positive_pairs_only,
        negative_to_positive_ratio=negative_to_positive_ratio,
        max_negatives_when_no_positive=max_negatives_when_no_positive,
        sampling_seed=sampling_seed,
    )

    local_id2label = {idx: label for label, idx in local_label2id.items()}

    shared_labels = []
    for local_label_id in labels_numeric:
        normalized_label = normalize_label(local_id2label[local_label_id])
        shared_labels.append(label2id[normalized_label])

    return Dataset.from_dict(
        {
            "text": marked_sentences,
            "label": shared_labels,
            "sample_weight": [float(sample_weight)] * len(marked_sentences),
        }
    )


def build_weighted_train_dataset(
    dataset_specs: List[Tuple[str, float]],
    label2id: Dict[str, int],
    positive_pairs_only: bool = False,
    negative_to_positive_ratio: float = 3.0,
    max_negatives_when_no_positive: int = 3,
    sampling_seed: int = 42,
):
    datasets = []
    for json_path, weight in dataset_specs:
        datasets.append(
            build_re_dataset_from_json(
                json_path,
                label2id=label2id,
                sample_weight=weight,
                positive_pairs_only=positive_pairs_only,
                negative_to_positive_ratio=negative_to_positive_ratio,
                max_negatives_when_no_positive=max_negatives_when_no_positive,
                sampling_seed=sampling_seed,
            )
        )

    return concatenate_datasets(datasets).shuffle(seed=sampling_seed)
