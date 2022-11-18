import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/corrected_dataset_512_v1.csv")
df = df.dropna()
df = df.sample(frac=1).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)
df_len = len(df)
print(df.head())
print("DF LEN:", df_len)
train_size = int(df_len*0.9)
print("TRAIN_SIZE:", train_size)
df_train = df.loc[:train_size]
print(df_train.head())
df_validation = df.loc[train_size:]
print(df_validation.head())


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
f = open("data/corrected_dataset_512_v1_finetune_train.csv", "w")

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

f = open("data/corrected_dataset_512_v1_finetune_validation.csv", "w")

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
lines = []
for seq, len, family in zip(df['corrected_sequences'].to_list(), df['len'].to_list(), df['label'].to_list()):

    line = ''
    for ch in seq:
        if ch.lower() not in ['c', 'a', 'g', 't', 'u', 'n', 'w', 'r', 'k', 'm', 'y', 's', 'v', 'h', 'd', 'b', '\n']:
            print(ch, "NOT IN VOCAB:", line)
        line = line + ch.lower() + ' '
    line = line + '\n'
    
    lines.append(line)
    if family not in lengths:
        lengths[family] = [len]
    else:
        lengths[family].append(len)


print(lengths.keys())
# plot
seq_lengths = []
for k in lengths.keys():
    seq_lengths.append(np.array(lengths[k]))

fig, ax = plt.subplots()
fig.set_size_inches(20,15)
plt.xticks(rotation = 90, fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
VP = ax.boxplot(seq_lengths, labels=[x.replace(".csv", "") for x in lengths.keys()])
plt.savefig("./boxplot_viral_subset_pretrain.png")
plt.show()


f = open("data/corrected_dataset_512_v1_pretrain.txt", "w")
f.writelines(lines)
f.close()







