import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

df_train = pd.read_csv("data/harvar_var_len/len_250_X_train_1.csv")
#df_validation = pd.read_csv("data/corrected_sequences_X_val.csv")
#df_test = pd.read_csv("data/corrected_sequences_X_test.csv")

#print(df_train.head())

#labelss = df_train['label'].unique()
labelss = np.unique(df_train['label'].to_list())
datas = []
data_lens = []
for lbl in labelss:
    data = df_train[df_train['label'] == lbl]['len'].values
    datas.append(data)

    data_lens.append(data.size)


print(labelss)


labelss = [str.replace(x, "fa.csv", "") for x in labelss]
labelss = [str.replace(x, "fasta.csv", "") for x in labelss]
labelss = [str.replace(x, "RF", "") for x in labelss]
fig, ax = plt.subplots()
fig.set_figwidth(15)
fig.set_figheight(10)

fig.subplots_adjust(bottom=0.2)
mpl.rcParams['boxplot.boxprops.color'] = 'blue'
mpl.rcParams['boxplot.flierprops.color'] = 'blue'
mpl.rcParams['boxplot.whiskerprops.color'] = 'blue'
box = ax.boxplot(datas)



pos = np.arange(len(labelss)) + 1
ax.set_xticks(pos, labels=labelss, rotation=90, fontsize=18)
plt.yticks(fontsize=20)

ax2 = ax.twinx()

#ax2.plot(pos, data_lens, linestyle = 'dashed', marker = '*',
#         markerfacecolor = 'red', markersize = 8, label='family size')
ax2.plot(pos, data_lens, 'ro', markersize = 8, label='family size')
ax.set_ylabel('sequence length distribution', color='blue', fontsize=20)
ax2.set_ylabel('family size', color='red', fontsize=20)
ax.set_xlabel('families', fontsize=20)



plt.show()
plt.savefig("dataset_len_distribution.png")
