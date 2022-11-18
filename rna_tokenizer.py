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

#bert_tokenizer.normalizer = normalizers.Sequence([Lowercase()])
#bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
bert_tokenizer.pre_tokenizer = Whitespace()
bert_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)
#trainer = WordLevelTrainer(
#    vocab_size=14, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
#)
trainer = WordLevelTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]","[UNK]", "[MASK]"])


rna_seqs_4_pretrain = []
rna_seqs_4_tokenizer = []

file_ind = 0
seq_lengths_all = []
#file_list = os.listdir("/home/desin/CS230/RNABERT/data/Rfam")
#print(file_list)
for file_name in glob.glob("/home/desin/CS230/RNABERT/data/Rfam/*"):
    print(file_name)
    seq_lengths = []

    file_rna = open(file_name, 'r')
    Lines = file_rna.readlines()
    file_rna.close()

    for line in Lines:
        if line[0] != '>':
            #line = line[:-1]
            rna_str_1 = f'{file_name}, {file_ind},'
            rna_str_2 = ''
            #print("******", line, "*****")
            for ch in line:
                if ch.lower() not in ['c', 'a', 'g', 't', 'u', 'n', 'w', 'r', 'k', 'm', 'y', 's', 'v', 'h', 'd', 'b', '\n']:
                    print(ch, "NOT IN VOCAB:", line)
                rna_str_1 = rna_str_1 + ch.lower() + ' '
                rna_str_2 = rna_str_2 + ch.lower() + ' '
            #rna_str = rna_str[:-1]

            if len(line)/2 < 400:
                rna_seqs_4_pretrain.append(rna_str_1)
                rna_seqs_4_tokenizer.append(rna_str_2)
                seq_lengths.append(len(line)/2)

    seq_lengths_all.append(np.array(seq_lengths))
    file_ind = file_ind + 1
    #if file_ind > 3:
    #    break



# make data:
print(seq_lengths_all[0].shape)

# plot
fig, ax = plt.subplots()
VP = ax.boxplot(seq_lengths_all)
plt.savefig("./boxplot.png")
plt.show()


file_rna = open('./Rfam_sequences.fa.txt', 'w')
file_rna.writelines(rna_seqs_4_tokenizer)
file_rna.close()
#bert_tokenizer.train_from_iterator(rna_seqs_4_tokenizer, trainer=trainer)
#bert_tokenizer.save("./bert-rna-tokenizer.json")


