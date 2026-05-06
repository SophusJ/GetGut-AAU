import json
import os
import re
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from unidecode import unidecode

SAT_MODEL = None
try:
    from wtpsplit import SaT
    SAT_MODEL = SaT("sat-3l-sm")
    # Try to use GPU if available
    try:
        SAT_MODEL.half().to("cuda")
        print("wtpsplit SaT model loaded on GPU")
    except Exception:
        print("wtpsplit SaT model loaded on CPU (GPU not available)")
except ImportError as e:
    print(f"Failed to import wtpsplit SaT: {e}")
    SAT_MODEL = None
except Exception as e:
    print(f"Failed to load wtpsplit model: {e}")
    SAT_MODEL = None


def build_relation_lookup(documents: List[Dict]) -> Dict[Tuple, str]:
    relation_lookup = {}
    same_sent_count = 0
    cross_sent_count = 0
    cross_loc_count = 0
    boundary_span_count = 0

    for doc_id, document in documents.items():
        if not isinstance(document, dict):
            continue
        
        metadata = document.get("metadata", {})
        entities = document.get("entities", [])
        sections = {
            "title": metadata.get("title", ""),
            "abstract": metadata.get("abstract", "")
        }
        
        # Build entity location map
        entity_by_offset = {}
        for entity in entities:
            start = entity.get("start_idx")
            end = entity.get("end_idx")
            location = entity.get("location")
            if isinstance(start, int) and isinstance(end, int):
                entity_by_offset[(start, end, location)] = entity
        
        # Process each relation
        for relation in document.get("relations", []):
            if not isinstance(relation, dict):
                continue
            
            predicate = relation.get("predicate")
            if not predicate:
                continue
            
            sub_start = relation.get("subject_start_idx")
            sub_end = relation.get("subject_end_idx")
            sub_location = relation.get("subject_location")
            
            obj_start = relation.get("object_start_idx")
            obj_end = relation.get("object_end_idx")
            obj_location = relation.get("object_location")
            
            if not (isinstance(sub_start, int) and isinstance(sub_end, int) and
                    isinstance(obj_start, int) and isinstance(obj_end, int)):
                continue
            
            # Check: same location?
            if sub_location != obj_location:
                cross_loc_count += 1
                continue
            
            # Check: same sentence? (entities must appear in at least one common sentence)
            section_text = sections.get(sub_location, "")
            if not section_text:
                continue
            
            sentences = get_sentence_boundaries(section_text)
            sub_sentences = set(find_entity_sentences(sub_start, sub_end, sentences))
            obj_sentences = set(find_entity_sentences(obj_start, obj_end, sentences))
            
            # Check if entities have no sentences
            if not sub_sentences or not obj_sentences:
                cross_sent_count += 1
                continue
            
            # Find common sentences
            common_sentences = sub_sentences & obj_sentences
            
            if not common_sentences:
                cross_sent_count += 1
                continue
            
            # Check if either entity spans multiple sentences (boundary crossing)
            if len(sub_sentences) > 1 or len(obj_sentences) > 1:
                boundary_span_count += 1
            
            # Keep this relation (it appears in at least one common sentence)
            key = (sub_start, sub_end, obj_start, obj_end)
            relation_lookup[key] = predicate
            same_sent_count += 1

    
    return relation_lookup

def inject_markers(
    text: str,
    sub_start: int,
    sub_end: int,
    obj_start: int,
    obj_end: int
) -> str:

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

def generate_candidate_pairs(
    entities: List[Dict],
    relation_lookup: Dict[Tuple, str],
    positive_pairs_only: bool = False
) -> List[Tuple[Dict, str, Dict]]:

    pairs = []
    
    for i, e1 in enumerate(entities):
        for j, e2 in enumerate(entities):
            if i == j:
                continue
            
            # Look up relation
            key = (e1["start_idx"], e1["end_idx"], e2["start_idx"], e2["end_idx"])
            predicate = relation_lookup.get(key, "no_relation")

            if positive_pairs_only and predicate == "no_relation":
                continue
            
            pairs.append((e1, predicate, e2))
    
    return pairs

def build_re_dataset(
    instances: Dict,
    relation_lookup: Dict[Tuple, str],
    positive_pairs_only: bool = False,
    negative_to_positive_ratio: Optional[float] = None,
    max_negatives_when_no_positive: Optional[int] = None,
    sampling_seed: int = 42
) -> Tuple[List[str], List[str], Dict[str, int], int]:

    marked_sentences = []
    labels_str = []
    label_set = {"no_relation"} 
    rng = random.Random(sampling_seed)
    
    pair_count = 0
    positive_pairs = 0
    sentence_pair_counts = []  # Track pairs per sentence
    dropped_negative_pairs = 0

    for doc_id, doc_instance in instances.items():
        sections = doc_instance.get("sections", {})
        
        for location, section_sentences in sections.items():
            for sentence_instance in section_sentences:
                entities = sentence_instance.get("entities", [])
                
                if not entities:
                    continue
                
                # Generate candidate pairs
                pairs = generate_candidate_pairs(
                    entities,
                    relation_lookup,
                    positive_pairs_only=positive_pairs_only
                )

                if (
                    not positive_pairs_only
                    and negative_to_positive_ratio is not None
                    and negative_to_positive_ratio >= 0
                ):
                    pos_pairs = [p for p in pairs if p[1] != "no_relation"]
                    neg_pairs = [p for p in pairs if p[1] == "no_relation"]

                    if pos_pairs:
                        max_negatives = int(len(pos_pairs) * negative_to_positive_ratio)
                    else:
                        max_negatives = max_negatives_when_no_positive if max_negatives_when_no_positive is not None else len(neg_pairs)

                    if max_negatives < len(neg_pairs):
                        sampled_neg_pairs = rng.sample(neg_pairs, max_negatives)
                        dropped_negative_pairs += len(neg_pairs) - max_negatives
                    else:
                        sampled_neg_pairs = neg_pairs

                    pairs = pos_pairs + sampled_neg_pairs

                sentence_pair_counts.append(len(pairs))
                
                sent_text = sentence_instance["sent_text"]
                sent_start = sentence_instance["sent_start"]
                
                for subject, predicate, obj in pairs:
                    pair_count += 1
                    
                    # Get sentence-relative offsets
                    sub_offset_start = subject["sent_offset_start"]
                    sub_offset_end = subject["sent_offset_end"]
                    obj_offset_start = obj["sent_offset_start"]
                    obj_offset_end = obj["sent_offset_end"]
                    
                    # Inject markers into sentence (using sentence-relative offsets)
                    marked_text = inject_markers(
                        sent_text,
                        sub_offset_start,
                        sub_offset_end,
                        obj_offset_start,
                        obj_offset_end
                    )
                    
                    marked_sentences.append(marked_text)
                    labels_str.append(predicate)
                    label_set.add(predicate)
                    
                    if predicate != "no_relation":
                        positive_pairs += 1
    
    # Build label2id mapping (assign ID 0 to "no_relation")
    label_list = ["no_relation"] + sorted([l for l in label_set if l != "no_relation"])
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Assign numeric IDs to labels
    labels_numeric = [label2id[label] for label in labels_str]

    return marked_sentences, labels_numeric, label2id, id2label

def split_by_regex(text: str) -> List[str]:

    # Split on: sentence-ending punctuation + space + capital, OR paragraph breaks, OR newlines before capitals
    pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?:\n\s*\n)+|\n(?=[A-Z])'
    sentences = re.split(pattern, text)
    # Filter out empty strings
    return [s.strip() for s in sentences if s.strip()]

def get_sentence_boundaries(text: str) -> List[Tuple[int, int, str]]:
    if not text.strip():
        return []
    
    sentences = None
    fallback_reason = None
    
    # Try neural model first
    if SAT_MODEL is not None:
        try:
            sentences = SAT_MODEL.split(text)
            
            if len(sentences) == 1 and len(text) > 500:
                fallback_reason = "Neural model returned 1 sentence for long text (>500 chars)"
                sentences = None
        except Exception as e:
            fallback_reason = f"Neural model exception: {str(e)[:50]}"
            sentences = None
    
    # Use regex fallback if neural model unavailable or failed
    if sentences is None:
        if fallback_reason:
            print(f"[DEBUG] Using regex fallback: {fallback_reason}")
        sentences = split_by_regex(text)
    
    # Reconstruct character positions by finding each sentence in the original text
    sentence_bounds = []
    search_start = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        # Find this sentence starting from our current position
        start_pos = text.find(sentence, search_start)
        
        if start_pos == -1:
            # Sentence not found, try finding without leading/trailing whitespace
            sentence_trimmed = sentence.strip()
            start_pos = text.find(sentence_trimmed, search_start)
            if start_pos == -1:
                continue
            end_pos = start_pos + len(sentence_trimmed)
        else:
            end_pos = start_pos + len(sentence)
        
        sentence_text = sentence.strip()
        
        if sentence_text:
            sentence_bounds.append((start_pos, end_pos, sentence_text))
            search_start = end_pos
    
    # If no sentences were found, return entire text as single sentence
    if not sentence_bounds:
        return [(0, len(text), text.strip())]
    
    return sentence_bounds


def find_entity_sentences(entity_start: int, entity_end: int, sentences: List[Tuple[int, int, str]]) -> List[int]:
    matching_sentences = []
    for sent_idx, (sent_start, sent_end, _) in enumerate(sentences):
        if sent_start < entity_end and entity_start < sent_end:
            matching_sentences.append(sent_idx)
    return matching_sentences


def build_sentence_instances(documents: List[Dict]) -> Dict:

    instances = {}
    boundary_entity_count = 0
    contained_entity_count = 0
    sentence_count = 0
    total_entities_in_sentences = 0
    max_entities_per_sentence = 0
    sentences_with_multiple_entities = 0
    
    for doc_id, document in documents.items():
        if not isinstance(document, dict):
            continue
        
        metadata = document.get("metadata", {})
        entities = document.get("entities", [])
        
        # Build mapping: (location, start, end) -> entity
        entity_map = {}
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            location = entity.get("location")
            start = entity.get("start_idx")
            end = entity.get("end_idx")
            if location and isinstance(start, int) and isinstance(end, int):
                entity_map[(location, start, end)] = entity
        
        sections = {}
        
        # Process each location (title, abstract)
        for location in ["title", "abstract"]:
            section_text = metadata.get(location)
            if not isinstance(section_text, str) or not section_text.strip():
                continue
            
            # Sentence-split the section
            sentences = get_sentence_boundaries(section_text)
            
            # Validation: warn if entire section is one sentence (likely wtpsplit failure)
            if len(sentences) == 1 and len(section_text) > 500:
                print(f"[WARNING] Doc {doc_id} {location}: {len(section_text)} chars in 1 sentence (wtpsplit may have failed)")
            
            section_instances = []
            
            for sent_idx, (sent_start, sent_end, sent_text) in enumerate(sentences):
                # Find all entities that touch this sentence
                sent_entities = []
                for entity in entities:
                    if entity.get("location") != location:
                        continue
                    
                    ent_start = entity.get("start_idx")
                    ent_end = entity.get("end_idx")
                    
                    # Check if entity overlaps with this sentence
                    if sent_start < ent_end and ent_start < sent_end:
                        # Compute sentence-relative offsets
                        # Clamp to sentence boundaries for boundary-crossing entities
                        sent_offset_start = max(0, ent_start - sent_start)
                        sent_offset_end = min(sent_end - sent_start, ent_end - sent_start)
                        
                        # Check if entity fully contained or spans boundary
                        is_fully_contained = (sent_start <= ent_start and ent_end <= sent_end)
                        entity_sentence_list = find_entity_sentences(ent_start, ent_end, sentences)
                        spans_boundary = len(entity_sentence_list) > 1
                        
                        if spans_boundary:
                            boundary_entity_count += 1
                        else:
                            contained_entity_count += 1
                        
                        sent_entities.append({
                            "start_idx": ent_start,  # Keep document-level for reference
                            "end_idx": ent_end,
                            "label": entity.get("label"),
                            "text_span": entity.get("text_span"),
                            "uri": entity.get("uri"),
                            "sent_offset_start": sent_offset_start,  # Sentence-relative (clamped)
                            "sent_offset_end": sent_offset_end,
                            "spans_sentences": spans_boundary,
                            "entity_sentences": entity_sentence_list  # All sentences it appears in
                        })
                
                # Track statistics
                if len(sent_entities) > 0:
                    sentence_count += 1
                    total_entities_in_sentences += len(sent_entities)
                    max_entities_per_sentence = max(max_entities_per_sentence, len(sent_entities))
                    if len(sent_entities) > 1:
                        sentences_with_multiple_entities += 1
                
                section_instances.append({
                    "sent_idx": sent_idx,
                    "sent_start": sent_start,
                    "sent_end": sent_end,
                    "sent_text": sent_text,
                    "entities": sent_entities
                })
            
            if section_instances:
                sections[location] = section_instances
        
        instances[doc_id] = {
            "metadata": metadata,
            "sections": sections
        }
    
    return instances