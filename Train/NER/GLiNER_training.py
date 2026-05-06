import json
import torch
import os
import logging
from datetime import datetime
from gliner import GLiNER
from types import SimpleNamespace

#Setup logging for debugging.
log_dir = "./gliner_biomed_logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(
    log_dir,
    f"gliner_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "Predictions", "NER")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

logger.info(f"Device: {DEVICE}")
logger.info(f"Output root: {OUTPUT_ROOT}")

train_root = os.path.join(PROJECT_ROOT, "Annotations", "Train", "json_format")
dev_path = os.path.join(PROJECT_ROOT, "Annotations", "Dev", "json_format", "dev.json")

# Lowercase labels for the Biomed model's natural language search
ENTITY_TYPES = [
    "anatomical location", "animal", "biomedical technique", "bacteria",
    "chemical", "dietary supplement", "ddf", "drug", "food", 
    "gene", "human", "microbiome", "statistical technique"
]
# Funnel setup with more granular stages and specific training steps per stage
STAGES = [
    ("bronze", os.path.join(train_root, "train_bronze.json"), 1000),
    ("silver2025", os.path.join(train_root, "train_silver_2025.json"), 1500),
    ("silver", os.path.join(train_root, "train_silver.json"), 2000),
    ("gold", os.path.join(train_root, "train_gold.json"), 3000),
]

#Helper function to process the JSON data into sentence-level instances with entity annotations.
def preprocess_data(path):
    import re
    
    logger.debug(f"Starting preprocessing of {path}")
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    logger.debug(f"Loaded {len(raw_data)} documents from {path}")

    dataset = []
    
    for doc_id, doc_content in raw_data.items():
        title = doc_content.get("metadata", {}).get("title", "")
        abstract = doc_content.get("metadata", {}).get("abstract", "")
        entities = doc_content.get("entities", [])
        
        # Process title and abstract separately
        for text, location in [(title, "title"), (abstract, "abstract")]:
            if not text:
                continue
            
            # Tokenize using word tokenization
            matches = list(re.finditer(r'\w+|[^\w\s]', text))
            tokenized_text = [m.group() for m in matches]
            token_spans = [m.span() for m in matches]
            
            ner_list = []
            
            # Filter entities for this location (title OR abstract)
            relevant_entities = [e for e in entities if e.get("location") == location]
            
            for ent in relevant_entities:
                char_start = ent.get("start_idx")
                char_end = ent.get("end_idx")
                label = str(ent.get("label", "")).upper()
                
                if char_start is None or char_end is None:
                    continue
                
                start_token_idx = -1
                end_token_idx = -1
                
                # Map character indices to token indices
                for idx, (tok_start, tok_end) in enumerate(token_spans):
                    if start_token_idx == -1 and tok_start <= char_start < tok_end:
                        start_token_idx = idx
                    
                    if tok_start <= (char_end - 1) < tok_end:
                        end_token_idx = idx
                        break
                
                if start_token_idx != -1 and end_token_idx != -1:
                    ner_list.append([start_token_idx, end_token_idx, label])
            
            # Only add non-empty samples
            if tokenized_text:
                dataset.append({
                    "id": f"{doc_id}_{location}",
                    "tokenized_text": tokenized_text,
                    "ner": ner_list
                })
    
    logger.info(f"Successfully loaded {len(dataset)} samples from {os.path.basename(path)}")
    return dataset


# Training function
def train_biomed(model, config, train_data, eval_data, stage_name):

    output_dir = os.path.join(OUTPUT_ROOT, stage_name, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"{'='*80}")
    logger.info(f"Starting stage: {stage_name}")
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Evaluation samples: {len(eval_data)}")
    logger.info(f"Learning rate (encoder): {config.lr_encoder}")
    logger.info(f"Learning rate (others): {config.lr_others}")
    logger.info(f"Batch size: {config.train_batch_size}")
    logger.info(f"Training steps: {config.num_steps}")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*80}")

    # Create training arguments using GLiNER's built-in method
    logger.info(f"[{stage_name}] Creating training arguments.")
    training_args = model.create_training_args(
        output_dir=output_dir,
        learning_rate=config.lr_encoder,
        others_lr=config.lr_others,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.train_batch_size,
        max_steps=config.num_steps,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.eval_every,
        logging_steps=config.eval_every,
        save_total_limit=3,
        use_cpu=(DEVICE == 'cpu'),
        report_to='none'
    )
    logger.info(f"[{stage_name}] Training arguments created")

    # Train using GLiNER's train_model API
    logger.info(f"[{stage_name}] Starting training.")
    try:
        trainer = model.train_model(
            train_dataset=train_data,
            eval_dataset=eval_data,
            training_args=training_args,
            freeze_components=None  # Don't freeze anything by default
        )
        logger.info(f"[{stage_name}] Training completed successfully")
    except Exception as e:
        logger.exception(f"[{stage_name}] Training failed with exception")
        raise

    # Save the final best model
    best_model_path = os.path.join(OUTPUT_ROOT, stage_name, "final_model")
    os.makedirs(best_model_path, exist_ok=True)
    logger.info(f"[{stage_name}] Saving final model to: {best_model_path}")
    model.save_pretrained(best_model_path)
    logger.info(f"[{stage_name}] Model saved successfully")

    logger.info(f"[{stage_name}] Stage completed\n")

    return best_model_path

# MAIN SCRIPT STARTS HERE
def main():
    logger.info("="*80)
    logger.info("GLINER BIOMED FUNNEL TRAINING PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    logger.info("Loading pretrained GLiNER Biomed model...")
    model = GLiNER.from_pretrained("Ihor/gliner-biomed-large-v1.0")
    logger.info("Model loaded successfully")

    # Load Eval Data Once
    logger.info("Preprocessing evaluation dataset...")
    eval_samples = preprocess_data(dev_path)
    logger.info(f"Evaluation dataset loaded: {len(eval_samples)} samples")

    config = SimpleNamespace(
        eval_every=200, 
        train_batch_size=8, 
        warmup_ratio=0.1, 
        lr_encoder=1e-5, 
        lr_others=5e-5
    )
    logger.info(f"Configuration: eval_every={config.eval_every}, batch_size={config.train_batch_size}, warmup_ratio={config.warmup_ratio}")

    logger.info(f"Starting pipeline with {len(STAGES)} stages.")
    stage_results = []

    for idx, (name, path, steps) in enumerate(STAGES, 1):
        logger.info(f"\nProcessing stage {idx}/{len(STAGES)}: {name}")
        logger.info(f"Training data path: {path}")

        logger.info(f"[{name}] Preprocessing training dataset")
        train_samples = preprocess_data(path)
        logger.info(f"[{name}] Training dataset loaded: {len(train_samples)} samples")

        # Override steps for specifc stages
        config.num_steps = steps
        
        best_path = train_biomed(model, config, train_samples, eval_samples, name)
        
        stage_results.append({
            "stage": name,
            "train_data_path": path,
            "num_train_samples": len(train_samples),
            "num_eval_samples": len(eval_samples),
            "training_steps": steps,
            "batch_size": config.train_batch_size,
            "lr_encoder": config.lr_encoder,
            "lr_others": config.lr_others,
            "output_path": best_path,
            "timestamp": datetime.now().isoformat()
        })

        # Reload the best performing version before moving to next stage
        logger.info(f"[{name}] Reloading best model for next funnel step.")
        model = GLiNER.from_pretrained(best_path)
        logger.info(f"[{name}] Model reloaded successfully")

    # Save training summary
    summary_path = os.path.join(OUTPUT_ROOT, "gliner_training_summary.json")
    logger.info(f"Saving training summary to: {summary_path}")
    with open(summary_path, "w") as f:
        json.dump(stage_results, f, indent=2)
    logger.info("Training summary saved")

    logger.info("="*80)
    logger.info("GLINER BIOMED FUNNEL TRAINING PIPELINE COMPLETED")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Logs saved to: {log_file}")
    logger.info("="*80)

if __name__ == "__main__":
    main()