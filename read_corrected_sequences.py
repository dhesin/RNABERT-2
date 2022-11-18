import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/corrected_dataset_512_v1.csv")
df = df.dropna()

family_to_index = {label:str(i) for i,label in enumerate(np.unique(df['label'].to_list()))}
print(family_to_index)
num_labels = len(family_to_index.keys())

family_to_one_hot = {}
for family in family_to_index.keys():
    one_hot = [0  if x != family_to_index[family] else 1 for x in range(num_labels)]
    family_to_one_hot[family] = one_hot

############################################################################################
#
#
###########################################################################################

lengths = {}
df_train = pd.read_csv("data/corrected_sequences_X_train.csv")
df_train = df_train.dropna()
f = open("data/corrected_seq_only_X_train.csv", "w")

print(df.head())

for seq, len, family in zip(df_train['corrected_sequences'].to_list(), df_train['len'].to_list(), df_train['label'].to_list()):

    line = ''
    #print(seq)
    for ch in seq:
        if ch.lower() not in ['c', 'a', 'g', 't', 'u', 'n', 'w', 'r', 'k', 'm', 'y', 's', 'v', 'h', 'd', 'b', '\n']:
            print(ch, "NOT IN VOCAB:", line)
        line = line + ch.lower() + ' '
    line = line + '\n'

    f.write(family_to_index[family])
    f.write("," + line)
    
    if family not in lengths:
        lengths[family] = [len]
    else:
        lengths[family].append(len)

f.close()

print(lengths.keys())
# plot
seq_lengths = []
for k in lengths.keys():
    seq_lengths.append(np.array(lengths[k]))

fig, ax = plt.subplots()
fig.set_size_inches(20,15)
plt.xticks(rotation = 90, fontsize=20, fontweight='bold')
VP = ax.boxplot(seq_lengths, labels=[x.replace(".csv", "") for x in lengths.keys()])
plt.savefig("./boxplot_viral_subset_train.png")
plt.show()


############################################################################################
#
#
###########################################################################################

lengths = {}

df_validation = pd.read_csv("data/corrected_sequences_X_val.csv")
df_validation = df_validation.dropna()
f = open("data/corrected_seq_only_X_val.csv", "w")

for seq, len, family in zip(df_validation['corrected_sequences'].to_list(), df_validation['len'].to_list(), df_validation['label'].to_list()):

    line = ''
    for ch in seq:
        if ch.lower() not in ['c', 'a', 'g', 't', 'u', 'n', 'w', 'r', 'k', 'm', 'y', 's', 'v', 'h', 'd', 'b', '\n']:
            print(ch, "NOT IN VOCAB:", line)
        line = line + ch.lower() + ' '
    line = line + '\n'

    f.write(family_to_index[family])
    f.write("," + line)
    
    if family not in lengths:
        lengths[family] = [len]
    else:
        lengths[family].append(len)

f.close()

print(lengths.keys())
# plot
seq_lengths = []
for k in lengths.keys():
    seq_lengths.append(np.array(lengths[k]))

fig, ax = plt.subplots()
fig.set_size_inches(20,15)
plt.xticks(rotation = 90, fontsize=20, fontweight='bold')
VP = ax.boxplot(seq_lengths, labels=[x.replace(".csv", "") for x in lengths.keys()])
plt.savefig("./boxplot_viral_subset_validation.png")
plt.show()




############################################################################################
#
#
###########################################################################################

lengths = {}

df_test = pd.read_csv("data/corrected_sequences_X_test.csv")
df_test = df_test.dropna()
f = open("data/corrected_seq_only_X_test.csv", "w")

for seq, len, family in zip(df_test['corrected_sequences'].to_list(), df_test['len'].to_list(), df_test['label'].to_list()):

    line = ''
    for ch in seq:
        if ch.lower() not in ['c', 'a', 'g', 't', 'u', 'n', 'w', 'r', 'k', 'm', 'y', 's', 'v', 'h', 'd', 'b', '\n']:
            print(ch, "NOT IN VOCAB:", line)
        line = line + ch.lower() + ' '
    line = line + '\n'

    f.write(family_to_index[family])
    f.write("," + line)
    
    if family not in lengths:
        lengths[family] = [len]
    else:
        lengths[family].append(len)

f.close()

print(lengths.keys())
# plot
seq_lengths = []
for k in lengths.keys():
    seq_lengths.append(np.array(lengths[k]))

fig, ax = plt.subplots()
fig.set_size_inches(20,15)
plt.xticks(rotation = 90, fontsize=20, fontweight='bold')
VP = ax.boxplot(seq_lengths, labels=[x.replace(".csv", "") for x in lengths.keys()])
plt.savefig("./boxplot_viral_subset_test.png")
plt.show()









