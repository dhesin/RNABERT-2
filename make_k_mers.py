import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
k_mer = 6


f_pretrain = open("data/corrected_k_mer_X.csv", "w")

############################################################################################
#
#
###########################################################################################
df_train = pd.read_csv("data/corrected_sequences_X_train.csv")
df_train = df_train.dropna()
f = open("data/corrected_k_mer_X_train.csv", "w")
f.write("label,text\n")
for seq, family in zip(df_train['corrected_sequences'].to_list(), df_train['label'].to_list()):

    line = ''
    #print(seq)
    for i in range(len(seq)-k_mer):
        if seq[i].lower() not in ['c', 'a', 'g', 't', 'u', 'n', 'w', 'r', 'k', 'm', 'y', 's', 'v', 'h', 'd', 'b', '\n']:
            print(seq[i], "NOT IN VOCAB:", line)
        k_mer_word = ''
        for j in range(k_mer):
            k_mer_word = k_mer_word + seq[i+j].lower()
        line = line + k_mer_word + ' '
    line = line + '\n'
    f_pretrain.write(line)
    f.write(family + "," + line)

f.close()


############################################################################################
#
#
###########################################################################################

df_validation = pd.read_csv("data/corrected_sequences_X_val.csv")
df_validation = df_validation.dropna()
f = open("data/corrected_k_mer_X_val.csv", "w")
f.write("label,text\n")

for seq, family in zip(df_validation['corrected_sequences'].to_list(), df_validation['label'].to_list()):

    line = ''
    #print(seq)
    for i in range(len(seq)-k_mer):
        if seq[i].lower() not in ['c', 'a', 'g', 't', 'u', 'n', 'w', 'r', 'k', 'm', 'y', 's', 'v', 'h', 'd', 'b', '\n']:
            print(seq[i], "NOT IN VOCAB:", line)
        k_mer_word = ''
        for j in range(k_mer):
            k_mer_word = k_mer_word + seq[i+j].lower()
        line = line + k_mer_word + ' '
    line = line + '\n'
    f_pretrain.write(line)
    f.write(family + "," + line)

f.close()


############################################################################################
#
#
###########################################################################################

lengths = {}

df_test = pd.read_csv("data/corrected_sequences_X_test.csv")
df_test = df_test.dropna()
f = open("data/corrected_k_mer_X_test.csv", "w")
f.write("label,text\n")

for seq, family in zip(df_test['corrected_sequences'].to_list(), df_test['label'].to_list()):

    line = ''
    #print(seq)
    for i in range(len(seq)-k_mer):
        if seq[i].lower() not in ['c', 'a', 'g', 't', 'u', 'n', 'w', 'r', 'k', 'm', 'y', 's', 'v', 'h', 'd', 'b', '\n']:
            print(seq[i], "NOT IN VOCAB:", line)
        k_mer_word = ''
        for j in range(k_mer):
            k_mer_word = k_mer_word + seq[i+j].lower()
        line = line + k_mer_word + ' '
    line = line + '\n'
    f_pretrain.write(line)
    f.write(family + "," + line)

f.close()


############################################################################################
#
#
###########################################################################################

f_pretrain.close()

df_train = pd.read_csv("data/corrected_k_mer_X_train.csv")
df_validation = pd.read_csv("data/corrected_k_mer_X_val.csv")
df_test = pd.read_csv("data/corrected_k_mer_X_test.csv")

df_all = pd.concat([df_train, df_validation, df_test])

family_to_index = {label:str(i) for i,label in enumerate(np.unique(df_all['label'].to_list()))}
num_labels = len(family_to_index.keys())
print("Number of labels:", num_labels)

family_to_one_hot = {}
for family in family_to_index.keys():
    one_hot = [0  if x != family_to_index[family] else 1 for x in range(num_labels)]
    family_to_one_hot[family] = one_hot


df_train['label'] = df_train['label'].apply(lambda x: family_to_index[x])
df_validation['label'] = df_validation['label'].apply(lambda x: family_to_index[x])
df_test['label'] = df_test['label'].apply(lambda x: family_to_index[x])

df_train.to_csv("data/corrected_k_mer_X_train.csv")
df_validation.to_csv("data/corrected_k_mer_X_val.csv")
df_test.to_csv("data/corrected_k_mer_X_test.csv")

