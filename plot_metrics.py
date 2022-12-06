import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


df = pd.read_json("out_finetune_harvar_var_len_250/checkpoint-800/trainer_state.json")

dict_1 = df.log_history.to_dict()

df = pd.DataFrame.from_dict(dict_1)
print(df.loc['loss'])
print(df.loc['epoch'])

df_1 = pd.DataFrame({'loss':df.loc['loss'], 'epoch':df.loc['epoch']})
df_1 = df_1.dropna()

df_2 = pd.DataFrame({'eval_loss':df.loc['eval_loss'], 'epoch':df.loc['epoch']})
df_2 = df_2.dropna()


#plt.ylim([0,1])
#plt.xlim([0,20])
plt.xlabel("epochs")
plt.plot(df_1["epoch"], df_1['loss'], color='red', label='train loss')
plt.plot(df_2["epoch"], df_2['eval_loss'], color = 'green',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'green', markersize = 4, label='eval loss')
plt.plot(df.loc["epoch"].values, df.loc["eval_accuracy"].values, color = 'blue',
         linestyle = 'dashed', marker = '*',
         markerfacecolor = 'green', markersize = 4, label='Eval accuracy')
plt.legend(loc='best')

plt.show()
plt.savefig("loss.png", dpi=300)


