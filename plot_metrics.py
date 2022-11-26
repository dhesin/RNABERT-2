import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


df = pd.read_json("out_mlm/checkpoint-45000/trainer_state.json")

dict_1 = df.log_history.to_dict()

df = pd.DataFrame.from_dict(dict_1)
print(df)


plt.ylim([0,2])
plt.xlabel("epochs")
plt.plot(df.loc["epoch"].values, df.loc['loss'].values, color='red', label='train loss')
plt.plot(df.loc["epoch"].values, df.loc["eval_loss"].values, color = 'green',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'green', markersize = 4, label='eval loss')
plt.plot(df.loc["epoch"].values, df.loc["eval_accuracy"].values, color = 'blue',
         linestyle = 'dashed', marker = '*',
         markerfacecolor = 'green', markersize = 4, label='MLM eval accuracy')
plt.legend(loc='best')

plt.show()
plt.savefig("loss.png", dpi=300)


