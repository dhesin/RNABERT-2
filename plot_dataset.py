import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

df_train = pd.read_csv("data/corrected_sequences_X_train.csv")
df_validation = pd.read_csv("data/corrected_sequences_X_val.csv")
df_test = pd.read_csv("data/corrected_sequences_X_test.csv")

print(df_train.head())

labelss = df_train['label'].unique()
datas = []
for lbl in labelss:
    data = df_train[df_train['label'] == lbl]['len'].values
    datas.append(data)

print(labelss)
labelss = [str.replace(x, "fa.csv", "") for x in labelss]
labelss = [str.replace(x, "fasta.csv", "") for x in labelss]
labelss = [str.replace(x, "RF", "") for x in labelss]
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.boxplot(datas)

pos = np.arange(len(labelss)) + 1
ax.set_xticks(pos, labels=labelss, rotation=90, fontsize=8)




plt.show()

plt.savefig("dataset.png")


