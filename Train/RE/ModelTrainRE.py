import json
import os
import logging
from datetime import datetime

import evaluate
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from re_training_utils import (
    BASE_MODEL_PATH,
    DEFAULT_DATASET_SPECS,
    DEFAULT_EVAL_JSON_PATH,
    SPECIAL_MARKER_TOKENS,
    build_re_dataset_from_json,
    build_shared_re_label_mapping,
)

# Configure logging
log_dir = "./re_funnel_logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(
    log_dir,
    f"re_funnel_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


for json_path, _ in DEFAULT_DATASET_SPECS:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing dataset file: {json_path}")
if not os.path.exists(DEFAULT_EVAL_JSON_PATH):
    raise FileNotFoundError(f"Missing eval file: {DEFAULT_EVAL_JSON_PATH}")


def stage_name_from_path(path: str) -> str:
    filename = os.path.basename(path)
    name, _ = os.path.splitext(filename)
    return name.replace("train_", "")


STAGES = [
    (stage_name_from_path(path), path, weight)
    for path, weight in DEFAULT_DATASET_SPECS
]


class WeightedDataCollator:
    def __init__(self, tokenizer):
        self.base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        sample_weights = [example.pop("sample_weight", 1.0) for example in features]
        batch = self.base_collator(features)
        batch["sample_weight"] = torch.tensor(sample_weights, dtype=torch.float32)
        return batch


class WeightedSequenceClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weight")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_example_loss = loss_fct(logits, labels)
        weighted_loss = (per_example_loss * sample_weights.float()).sum() / sample_weights.sum().clamp(min=1e-8)

        return (weighted_loss, outputs) if return_outputs else weighted_loss


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        **accuracy_metric.compute(predictions=predictions, references=labels),
        **f1_metric.compute(predictions=predictions, references=labels, average="macro"),
        **precision_metric.compute(predictions=predictions, references=labels, average="macro"),
        **recall_metric.compute(predictions=predictions, references=labels, average="macro"),
    }


def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        remove_unused_columns=False,
    )


def train_stage(
    stage_name,
    stage_json_path,
    sample_weight,
    model_source,
    tokenizer,
    label2id,
    id2label,
    eval_dataset,
    output_root,
):
    logger.info("=" * 80)
    logger.info(f"Starting stage: {stage_name}")
    logger.info(f"Train data: {stage_json_path}")
    logger.info(f"Model source: {model_source}")
    logger.info("=" * 80)

    stage_dataset = build_re_dataset_from_json(
        stage_json_path,
        label2id=label2id,
        sample_weight=sample_weight,
        positive_pairs_only=False,
        negative_to_positive_ratio=3.0,
        max_negatives_when_no_positive=3,
        sampling_seed=42,
    )
    logger.info(f"[{stage_name}] Dataset loaded. Size: {len(stage_dataset)}")

    if "label" in stage_dataset.column_names:
        stage_dataset = stage_dataset.rename_column("label", "labels")
    stage_dataset = stage_dataset.map(lambda batch: tokenize_function(tokenizer, batch), batched=True, remove_columns=['text'])
    logger.info(f"[{stage_name}] Dataset tokenized. Columns: {stage_dataset.column_names}")

    is_first_stage = (model_source == BASE_MODEL_PATH)
    if is_first_stage:
        logger.info(f"[{stage_name}] First stage detected - loading with explicit label configuration")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )
    else:
        logger.info(f"[{stage_name}] Subsequent stage - loading from checkpoint")
        model = AutoModelForSequenceClassification.from_pretrained(model_source)
        if model.config.num_labels != len(label2id):
            logger.error(f"[{stage_name}] Label count mismatch: {model.config.num_labels} vs {len(label2id)}")
            raise ValueError(
                f"Label count mismatch for {stage_name}: {model.config.num_labels} vs {len(label2id)}"
            )

    model.resize_token_embeddings(len(tokenizer))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{stage_name}] Model loaded. Total parameters: {total_params}, Trainable: {trainable_params}")

    stage_output_dir = os.path.join(output_root, f"re_funnel_{stage_name}")
    trainer = WeightedSequenceClassificationTrainer(
        model=model,
        args=get_training_args(stage_output_dir),
        train_dataset=stage_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=WeightedDataCollator(tokenizer),
        compute_metrics=compute_metrics,
    )

    logger.info(f"[{stage_name}] Starting training...")
    try:
        trainer.train()
        logger.info(f"[{stage_name}] Training completed successfully")
    except Exception as e:
        logger.exception(f"[{stage_name}] Training crashed")
        raise

    eval_metrics = trainer.evaluate()
    logger.info(f"[{stage_name}] Eval metrics: {eval_metrics}")

    final_model_dir = os.path.join(stage_output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"[{stage_name}] Saved final model to {final_model_dir}")

    return {
        "stage": stage_name,
        "train_data": stage_json_path,
        "output_dir": stage_output_dir,
        "final_model_dir": final_model_dir,
        "eval_metrics": eval_metrics,
    }


def main():
    logger.info("=" * 80)
    logger.info("RE FUNNELED TRAINING PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    logger.info("Building shared label mapping...")
    label2id, id2label = build_shared_re_label_mapping(
        [path for path, _ in DEFAULT_DATASET_SPECS] + [DEFAULT_EVAL_JSON_PATH]
    )
    logger.info(f"Label mapping created with {len(label2id)} labels")

    logger.info("Loading tokenizer and adding special marker tokens...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.add_tokens(SPECIAL_MARKER_TOKENS)
    logger.info(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size}")

    logger.info("Loading evaluation dataset...")
    eval_dataset = build_re_dataset_from_json(
        DEFAULT_EVAL_JSON_PATH,
        label2id=label2id,
        sample_weight=1.0,
        positive_pairs_only=False,
        negative_to_positive_ratio=3.0,
        max_negatives_when_no_positive=3,
        sampling_seed=42,
    )
    if "label" in eval_dataset.column_names:
        eval_dataset = eval_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.map(lambda batch: tokenize_function(tokenizer, batch), batched=True, remove_columns=['text'])
    logger.info(f"Evaluation dataset loaded. Size: {len(eval_dataset)}")

    output_root = "./re_funnel"
    os.makedirs(output_root, exist_ok=True)

    stage_results = []
    current_model_source = BASE_MODEL_PATH

    logger.info(f"Starting funneled training with {len(STAGES)} stages...")
    for stage_name, stage_json_path, sample_weight in STAGES:
        logger.info(f"Processing stage: {stage_name}")
        result = train_stage(
            stage_name=stage_name,
            stage_json_path=stage_json_path,
            sample_weight=sample_weight,
            model_source=current_model_source,
            tokenizer=tokenizer,
            label2id=label2id,
            id2label=id2label,
            eval_dataset=eval_dataset,
            output_root=output_root,
        )
        stage_results.append(result)
        current_model_source = result["final_model_dir"]
        logger.info(f"Stage {stage_name} completed. Next model source: {current_model_source}")

    summary_path = os.path.join(output_root, "re_funnel_training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stage_results, f, indent=2)
    logger.info(f"Training summary saved to: {summary_path}")

    logger.info("=" * 80)
    logger.info("RE FUNNELED TRAINING PIPELINE COMPLETED")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Logs saved to: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    logger.info("Script invoked as main")
    main()
