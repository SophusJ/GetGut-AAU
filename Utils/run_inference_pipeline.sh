#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

run_step() {
  local name="$1"
  local script_path="$2"

  echo "============================================================"
  echo "Running ${name}"
  echo "Script: ${script_path}"
  echo "============================================================"
  "$PYTHON_BIN" "$script_path"
  echo "Finished ${name}"
  echo
}

run_step "NER inference" "$REPO_ROOT/InferenceScripts/NER/ner_predict_and_save_output_gliner.py"
run_step "NERD inference" "$REPO_ROOT/InferenceScripts/NERD/Predict_NERD.py"
run_step "MRE inference" "$REPO_ROOT/InferenceScripts/MRE/InferencePURESentenceLevelRE.py"
run_step "CRE inference" "$REPO_ROOT/InferenceScripts/CRE/Predict_CRE.py"

echo "All inference steps completed successfully."

