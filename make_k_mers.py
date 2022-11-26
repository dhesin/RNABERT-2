import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
k_mer = 6

f_pretrain = open("data/pd_k_mer_pretrain.txt", "w")
############################################################################################
#
#
###########################################################################################
df_train = pd.read_csv("data/corrected_sequences_X_train.csv")
df_train = df_train.dropna()

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
    f_pretrain.write(line2+'\n')

#print(seqs)
df_train_out = pd.DataFrame({"label":labels, "text":seqs})
df_train_out.to_csv("data/pd_k_mer_train.csv", index=False)


############################################################################################
#
#
###########################################################################################

df_validation = pd.read_csv("data/corrected_sequences_X_val.csv")
df_validation = df_validation.dropna()

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
    f_pretrain.write(line2+'\n')

df_val_out = pd.DataFrame({"label":labels, "text":seqs})
df_val_out.to_csv("data/pd_k_mer_val.csv", index=False)

############################################################################################
#
#
###########################################################################################

df_test = pd.read_csv("data/corrected_sequences_X_test.csv")
df_test = df_test.dropna()
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
    f_pretrain.write(line2+'\n')

df_test_out = pd.DataFrame({"label":labels, "text":seqs})
df_test_out.to_csv("data/pd_k_mer_test.csv", index=False)


############################################################################################
#
#
###########################################################################################

f_pretrain.close()

df_train = pd.read_csv("data/pd_k_mer_train.csv")
df_validation = pd.read_csv("data/pd_k_mer_val.csv")
df_test = pd.read_csv("data/pd_k_mer_test.csv")

df_all = pd.concat([df_train, df_validation, df_test])

family_to_index = {label:str(i) for i,label in enumerate(np.unique(df_all['label'].to_list()))}
num_labels = len(family_to_index.keys())
print("Number of labels:", num_labels)

family_to_one_hot = {}
for family in family_to_index.keys():
    one_hot = [0  if x != family_to_index[family] else 1 for x in range(num_labels)]
    family_to_one_hot[family] = one_hot


df_train['label_id'] = df_train['label'].apply(lambda x: family_to_index[x])
df_validation['label_id'] = df_validation['label'].apply(lambda x: family_to_index[x])
df_test['label_id'] = df_test['label'].apply(lambda x: family_to_index[x])


df_train.to_csv("data/pd_k_mer_train.csv", index=False)
df_validation.to_csv("data/pd_k_mer_val.csv", index=False)
df_test.to_csv("data/pd_k_mer_test.csv", index=False)

