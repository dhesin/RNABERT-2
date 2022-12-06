import logging
import os
import random
import sys
import ast
import json
from dataclasses import dataclass, field
from typing import Optional
from torchmetrics.classification import MulticlassAccuracy
import datasets
import numpy as np
from datasets import load_dataset, Dataset
import sklearn
import evaluate
import transformers
import pandas as pd
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
    BertForSequenceClassification,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from custom_classification_head import BertForCustomClassification


#df_test = pd.read_csv("data/pd_6_mer_test_1.csv")
#df_test = pd.DataFrame(df_test)
#raw_datasets_test = Dataset.from_pandas(df_test)

data_files = {"train": "data/harvar_var_len/6_mer_100_train_1.csv", "validation": "data/harvar_var_len/6_mer_100_val_1.csv", "test": "data/harvar_var_len/6_mer_100_test_1.csv"}
raw_datasets = load_dataset("csv", data_files=data_files)


tokenizer = PreTrainedTokenizerFast(tokenizer_file="bert-rna-6-mer-tokenizer.json")
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
    #text_len = len(examples['text'])
    #x_repeat = 512/text_len+1
    #tmp_text = [examples['text'] + ' ' for i in range(x_repeat)] 
    
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

#model = BertForSequenceClassification.from_pretrained("./out_mlm_6layer_3/checkpoint-360000", num_labels=31)
#model = BertForSequenceClassification.from_pretrained("./out_mlm/checkpoint-10", num_labels=31)
#model = BertForSequenceClassification.from_pretrained("./out_finetune_with_pretrain/checkpoint-12500", num_labels=31)
#model = AutoModelForSequenceClassification.from_pretrained("./out_mlm/checkpoint-360000", num_labels=30)
model = BertForSequenceClassification.from_pretrained("./out_finetune_equal_length_1/checkpoint-5500", num_labels=31)


training_args = TrainingArguments(
    output_dir="./out_finetune",
    learning_rate=5e-5,
    per_device_train_batch_size=50,
    per_device_eval_batch_size=50,
    num_train_epochs=10,
    #weight_decay=0.01,
    metric_for_best_model="accuracy",
    eval_steps= 200,
    evaluation_strategy= 'steps',
    save_total_limit = 3,
    log_level = 'info',
    logging_steps = 100,
    #dataloader_num_workers = 10,
    save_strategy='steps',
    save_steps=200,
    load_best_model_at_end = True,
)


def compute_metrics2(p: EvalPrediction):
    
    cls = sklearn.metrics.classification_report(p.label_ids, np.argmax(p.predictions, axis=1), output_dict=True)
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

#trainer.train(resume_from_checkpoint = False)
#result = trainer.train()
#print(json.dumps(result))

#df_test_2 = pd.DataFrame({'label':tokenized_seqs_test['label'][:]})
#print(df_test_2['label'].value_counts())
#print(df_test_2['label'].dtype)

p = trainer.predict(tokenized_seqs['test'])
print(p)


cls = sklearn.metrics.classification_report( p.label_ids, np.argmax(p.predictions, axis=1),
                                            labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
                                            target_names=["RF00001.fa.csv", "RF00003.fa.csv", "RF00004.fa.csv", "RF00005.fa.csv", 
                                                    "RF00007.fa.csv", "RF00017.fa.csv", "RF00019.fa.csv", "RF00026.fa.csv", 
                                                    "RF00029.fa.csv", "RF00032.fa.csv", "RF00059.fa.csv", "RF00097.fa.csv", 
                                                    "RF00100.fa.csv", "RF00163.fa.csv", "RF00174.fa.csv", "RF00177.fa.csv", 
                                                    "RF00230.fa.csv", "RF00436.fa.csv", "RF00906.fa.csv", "RF00994.fa.csv",
                                                    "RF01315.fa.csv", "RF01317.fa.csv", "RF01787.fa.csv", "RF01960.fa.csv",
                                                    "RF02271.fa.csv", "RF02541.fa.csv", "RF02543.fa.csv", "RF04021.fa.csv", 
                                                    "RF04088.fa.csv", "human_mrna.csv", "virus_mrna.fasta.csv"]
                                                    )
print(cls)


#cm = sklearn.metrics.confusion_matrix(p.label_ids, np.argmax(p.predictions, axis=1))
#print(cm) 
