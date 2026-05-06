# GetGut@AAU — CLEF 2026 Challenge Repository

Repository for the **GetGut@AAU** submission to the **CLEF 2026 challenge**.

> Repo description: *GetGut@AAU - GitHub Repository for CLEF 2026 Challenge*

## Overview

This repository contains:

- **Annotations/**: input data in the expected JSON format (e.g., test articles).
- **InferenceScripts/**: scripts to run the full inference pipeline and produce submission JSON files.
- **Train/**: training code (organized by task).
- **Utils/**: helper artifacts (e.g., label mappings, contextual dictionaries/vectors).
- **Predictions/**: generated outputs in submission format.

The current codebase is Python-only.

## Tasks / Pipeline

The pipeline is split into the following stages:

1. **NER (Named Entity Recognition)**
   - Script: `InferenceScripts/NER/ner_predict_and_save_output_gliner.py`
   - Output:
     - `Predictions/NER/GetGut@AAU_T611_runID.json`
     - `Predictions/NER/predictions_for_NERD_MRE.json` (intermediate file used by later stages)

2. **NERD (Entity Linking / Normalization)**
   - Script: `InferenceScripts/NERD/Predict_NERD.py`
   - Output:
     - `Predictions/NERD/GetGut@AAU_T612_runID.json`

3. **MRE (Mention-level Relation Extraction)**
   - Script: `InferenceScripts/MRE/InferencePURESentenceLevelRE.py`
   - Output (intermediate + submission):
     - `Predictions/MRE/predictions_for_CRE.json` (intermediate for CRE)
     - `Predictions/MRE/GetGut@AAU_T621_runID.json` (submission)

4. **CRE (Concept-level Relation Extraction)**
   - Script: `InferenceScripts/CRE/Predict_CRE.py`
   - Output:
     - `Predictions/CRE/GetGut@AAU_T622_runID.json`

## Installation

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate  # (Linux/macOS)
# .venv\Scripts\activate   # (Windows)
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Running inference

The scripts use sensible repo-relative defaults (e.g., `Models/...`, `Annotations/...`, `Predictions/...`) and also support overriding paths via environment variables.

### NER

```bash
python InferenceScripts/NER/ner_predict_and_save_output_gliner.py
```

Notes:
- Expects a trained NER checkpoint at: `Models/NER/final_model/` (or override via `GETGUT_NER_MODEL_PATH`).
- Expects input at: `Annotations/Test/json_format/articles_test.json` (or override via `GETGUT_NER_INPUT_JSON`).

### NERD

```bash
python InferenceScripts/NERD/Predict_NERD.py
```

Notes:
- Expects a SapBERT-style SentenceTransformer model at: `Models/NERD_CRE/SapBERT-from-PubMedBERT-fulltext/`.
- Expects dictionary files:
  - `Utils/sapbert_contextual_vectors_best.pt`
  - `Utils/sapbert_contextual_mapping_best.json`

### MRE

```bash
python InferenceScripts/MRE/InferencePURESentenceLevelRE.py
```

Notes:
- Expects an HF Transformers sequence classification model at: `Models/RE/` (or override via `GETGUT_RE_MODEL_PATH`).
- Expects label mapping files:
  - `Utils/processed_re_data/label2id.json`
  - `Utils/processed_re_data/id2label.json`

### CRE

```bash
python InferenceScripts/CRE/Predict_CRE.py
```

Notes:
- Uses the same SapBERT model + dictionary files as NERD.
- Consumes `Predictions/MRE/predictions_for_CRE.json`.

## Model citations (required)

This repository uses / loads the following model families in code. If you use or redistribute these models, **please cite the original works** and comply with their licenses.

1. **GLiNER** (used for NER)
   - Code uses: `from gliner import GLiNER` and loads a checkpoint via `GLiNER.from_pretrained(...)`.
   - Project: https://github.com/urchade/GLiNER
   - Please cite GLiNER per the project’s recommended citation.

2. **SapBERT** (used for NERD and for URI prediction in CRE)
   - Code uses: `SentenceTransformer(...)` with a model directory named `SapBERT-from-PubMedBERT-fulltext`.
   - Paper: Liu et al., *Self-alignment Pretraining for Biomedical Entity Representations* (SapBERT).
   - Please cite SapBERT (and PubMedBERT if applicable to your checkpoint) according to the checkpoint you use.

3. **PURE-style relation extraction (sentence-level RE)**
   - MRE script loads a Transformers `AutoModelForSequenceClassification` checkpoint from `Models/RE/`.
   - If your checkpoint is based on the PURE approach, please cite the PURE paper:
     - Zhong and Chen, *A Frustratingly Easy Approach for Entity and Relation Extraction* (PURE).

4. **Hugging Face Transformers / Sentence-Transformers toolkits**
   - Tooling used to load and run models.
   - Please cite if required by your venue:
     - Hugging Face Transformers: https://github.com/huggingface/transformers
     - Sentence-Transformers: https://www.sbert.net/

> Important: the exact checkpoint names (and therefore the exact citations) depend on what is placed in `Models/`. Update this section with the **precise model IDs** (e.g., `urchade/gliner_multi_v2.1`, `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, etc.) once finalized.

## License

This repository is licensed under the **MIT License** (see `LICENSE`).
