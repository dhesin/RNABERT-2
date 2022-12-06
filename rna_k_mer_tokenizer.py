from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing

import glob
import os
import numpy as np
import matplotlib.pyplot as plt


bert_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
bert_tokenizer.pre_tokenizer = Whitespace()
bert_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)
trainer = WordLevelTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]","[UNK]", "[MASK]"])
bert_tokenizer.train(["./data/pd_6_mer_pretrain.txt"], trainer)
bert_tokenizer.save("./bert-rna-6-mer-tokenizer.json")


