import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version




data_files = {"train": "data/corrected_seq_only_X_train.csv", "validation": "data/corrected_seq_only_X_val.csv"}
raw_datasets = load_dataset("csv", data_files=data_files)

print(raw_datasets)


tokenizer = PreTrainedTokenizerFast(tokenizer_file="bert-rna-tokenizer.json")
tokenizer.pad_token = "[PAD]"
tokenizer.cls_token = "[CLS]"

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_seqs = raw_datasets.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("./out_mlm/checkpoint-4500", num_labels=30)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
    metric_for_best_model="accuracy",
    eval_steps= 1000,
    evaluation_strategy= 'steps',
    save_total_limit = 3,
    log_level = 'info',
    logging_steps = 500,
    dataloader_num_workers = 10,
    save_strategy='steps',
    save_steps=500,
)


def compute_metrics2(p: EvalPrediction):
    
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_seqs["train"],
    eval_dataset=tokenized_seqs["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics2,
    data_collator=data_collator,
)

result = trainer.train()
print(result)
result = trainer.evaluate()
print(result)