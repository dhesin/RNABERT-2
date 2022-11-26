import logging
import os
import random
import sys
import ast
from dataclasses import dataclass, field
from typing import Optional
from torchmetrics.classification import MulticlassAccuracy
import datasets
import numpy as np
from datasets import load_dataset
import sklearn
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
from custom_classification_head import BertForCustomClassification




data_files = {"train": "data/pd_k_mer_train.csv", "validation": "data/pd_k_mer_val.csv"}
raw_datasets = load_dataset("csv", data_files=data_files)

print(raw_datasets)


tokenizer = PreTrainedTokenizerFast(tokenizer_file="bert-rna-k-mer-tokenizer.json")
tokenizer.pad_token = "[PAD]"
tokenizer.cls_token = "[CLS]"

def preprocess_function(examples):
    #print(examples['text'][0])
    #print(str.replace(examples['text'][0]," ", ""))
    #print(ast.literal_eval(examples['text'][0]))
    examples['text'] = [str.replace(examples['text'][i],"'", "") for i in range(len(examples['text']))]
    examples['text'] = [str.replace(examples['text'][i],",", "") for i in range(len(examples['text']))]
    examples['text'] = [str.replace(examples['text'][i],"[", "") for i in range(len(examples['text']))]
    examples['text'] = [str.replace(examples['text'][i],"]", "") for i in range(len(examples['text']))]
    
    #print(examples['text'][0])
    return tokenizer(
                examples['text'],
                padding=True,
                truncation=True,
                max_length=512,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )


tokenized_seqs = raw_datasets.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = BertForCustomClassification.from_pretrained("./out_mlm/checkpoint-360000", num_labels=30)
#model = AutoModelForSequenceClassification.from_pretrained("./out_mlm/checkpoint-360000", num_labels=30)
#model = AutoModelForSequenceClassification.from_pretrained("./out_finetune/checkpoint-5000", num_labels=30)

training_args = TrainingArguments(
    output_dir="./out_finetune",
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
    metric_for_best_model="accuracy",
    eval_steps= 500,
    evaluation_strategy= 'steps',
    save_total_limit = 3,
    log_level = 'info',
    logging_steps = 100,
    dataloader_num_workers = 10,
    save_strategy='steps',
    save_steps=500,
)


def compute_metrics2(p: EvalPrediction):
    
    cls = sklearn.metrics.classification_report(np.argmax(p.predictions, axis=1), p.label_ids, output_dict=True)
    return cls
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    metric = MulticlassAccuracy(num_classes=30, average=None)
    result = metric(p.predictions, p.label_ids)
    return {"result":result}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_seqs["train"],
    eval_dataset=tokenized_seqs["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics2,
    data_collator=data_collator,
)

for batch in trainer.get_train_dataloader():
    print(batch)
    break

#trainer.train(resume_from_checkpoint = True)
result = trainer.train()
print(result)
#p = trainer.predict(tokenized_seqs["validation"])
#print(p)
#cls = sklearn.metrics.classification_report(np.argmax(p.predictions, axis=1), p.label_ids)
#print(cls)