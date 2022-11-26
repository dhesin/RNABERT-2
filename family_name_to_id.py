import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast


df_train = pd.read_csv("data/corrected_sequences_X_train.csv")
df_train = df_train.dropna()
df_validation = pd.read_csv("data/corrected_sequences_X_val.csv")
df_validation = df_validation.dropna()
df_test = pd.read_csv("data/corrected_sequences_X_test.csv")
df_test = df_test.dropna()

df_all = pd.concat([df_train, df_validation, df_test])

family_to_index = {label:str(i) for i,label in enumerate(np.unique(df_all['label'].to_list()))}
index_to_family = {str(i):label for i,label in enumerate(np.unique(df_all['label'].to_list()))}
num_labels = len(family_to_index.keys())
print("Number of labels:", num_labels)

print(family_to_index)
print(index_to_family)

