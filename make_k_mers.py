import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import json

k_mer = 6
split = "_1"
seq_len = 50

train_file_name = f"data/harvar_var_len/len_{seq_len}_X_train{split}.csv"
k_mer_train_file_name = f"data/harvar_var_len/{k_mer}_mer_{seq_len}_train{split}.csv"

val_file_name = f"data/harvar_var_len/len_{seq_len}_X_val{split}.csv"
k_mer_val_file_name = f"data/harvar_var_len/{k_mer}_mer_{seq_len}_val{split}.csv"

test_file_name = f"data/harvar_var_len/len_{seq_len}_X_test{split}.csv"
k_mer_test_file_name = f"data/harvar_var_len/{k_mer}_mer_{seq_len}_test{split}.csv"

#f_pretrain = open("data/pd_k_mer_pretrain.txt", "w")
############################################################################################
#
#
###########################################################################################
df_train = pd.read_csv(train_file_name)
df_train = df_train[pd.notna(df_train['corrected_sequences'])]

labels = []
seqs = []
for seq, family in zip(df_train['corrected_sequences'].to_list(), df_train['label'].to_list()):

    line2 = ""
    line = []
    #print(seq)
    for i in range(len(seq)-k_mer):
        if seq[i].lower() not in ['c', 'a', 'g', 't', 'u', 'n', 'w', 'r', 'k', 'm', 'y', 's', 'v', 'h', 'd', 'b', '\n']:
            print(seq[i], "NOT IN VOCAB:", line)
        k_mer_word = ""
        for j in range(k_mer):
            k_mer_word = k_mer_word + seq[i+j].lower()
        line2 = line2 + str(k_mer_word) + " "
        line.append(k_mer_word)

    labels.append(family)
    seqs.append(line)
    #f_pretrain.write(line2+'\n')

#print(seqs)
df_train_out = pd.DataFrame({"label":labels, "text":seqs})
df_train_out.to_csv(k_mer_train_file_name, index=False)


############################################################################################
#
#
###########################################################################################
df_validation = pd.read_csv(val_file_name)
df_validation = df_validation[pd.notna(df_validation['corrected_sequences'])]

labels = []
seqs = []

for seq, family in zip(df_validation['corrected_sequences'].to_list(), df_validation['label'].to_list()):

    line2 = ""
    line = []
    #print(seq)
    for i in range(len(seq)-k_mer):
        if seq[i].lower() not in ['c', 'a', 'g', 't', 'u', 'n', 'w', 'r', 'k', 'm', 'y', 's', 'v', 'h', 'd', 'b', '\n']:
            print(seq[i], "NOT IN VOCAB:", line)
        k_mer_word = ""
        for j in range(k_mer):
            k_mer_word = k_mer_word + seq[i+j].lower()
        line2 = line2 + str(k_mer_word) + " "
        line.append(k_mer_word)

    labels.append(family)
    seqs.append(line)
    #f_pretrain.write(line2+'\n')

df_val_out = pd.DataFrame({"label":labels, "text":seqs})
df_val_out.to_csv(k_mer_val_file_name, index=False)
############################################################################################
#
#
###########################################################################################
df_test = pd.read_csv(test_file_name)
df_test = df_test[pd.notna(df_test['corrected_sequences'])]
#df_test = df_test.dropna()
labels = []
seqs = []

for seq, family in zip(df_test['corrected_sequences'].to_list(), df_test['label'].to_list()):

    line2 = ""
    line = []
    #print(seq)
    for i in range(len(seq)-k_mer):
        if seq[i].lower() not in ['c', 'a', 'g', 't', 'u', 'n', 'w', 'r', 'k', 'm', 'y', 's', 'v', 'h', 'd', 'b', '\n']:
            print(seq[i], "NOT IN VOCAB:", line)
        k_mer_word = ""
        for j in range(k_mer):
            k_mer_word = k_mer_word + seq[i+j].lower()
        line2 = line2 + str(k_mer_word) + " "
        line.append(k_mer_word)

    labels.append(family)
    seqs.append(line)
    #f_pretrain.write(line2+'\n')

df_test_out = pd.DataFrame({"label":labels, "text":seqs})
df_test_out.to_csv(k_mer_test_file_name, index=False)
############################################################################################
#
#
###########################################################################################

#f_pretrain.close()

df_train = pd.read_csv(k_mer_train_file_name)
df_validation = pd.read_csv(k_mer_val_file_name)
df_test = pd.read_csv(k_mer_test_file_name)

df_all = pd.concat([df_train, df_validation, df_test])
#df_all = pd.concat([df_train, df_test])

with open('data/family_to_index.txt') as f:
    data = f.read()
    family_to_index = json.loads(data)
  

df_train = df_train.rename(columns={"label": "label_text"})
df_validation = df_validation.rename(columns={"label": "label_text"})
df_test = df_test.rename(columns={"label": "label_text"})

df_train['label'] = df_train['label_text'].apply(lambda x: family_to_index[x])
df_validation['label'] = df_validation['label_text'].apply(lambda x: family_to_index[x])
df_test['label'] = df_test['label_text'].apply(lambda x: family_to_index[x])


df_train.to_csv(k_mer_train_file_name, index=False)
df_validation.to_csv(k_mer_val_file_name, index=False)
df_test.to_csv(k_mer_test_file_name, index=False)
                             