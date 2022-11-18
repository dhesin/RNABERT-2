import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


df = pd.read_json("out_mlm/checkpoint-134000/trainer_state.json")

dict_1 = df.log_history.to_dict()

df = pd.DataFrame.from_dict(dict_1)
print(df)


fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(20,20))


ax0.set_title('Loss', fontsize=25)
ax0.set_xlabel('Step', fontsize=25)
ax0.plot(df.loc["step"], df.loc["loss"])
ax1.set_title('Learning Rate', fontsize=25)
ax1.set_xlabel('Step', fontsize=25)
ax1.plot(df.loc["step"], df.loc["learning_rate"])

eval_loss = df.loc["eval_loss"]
eval_loss = eval_loss[~eval_loss.isnull()]
print(eval_loss)
eval_accuracy = df.loc["eval_accuracy"]
eval_accuracy = eval_accuracy[~eval_accuracy.isnull()]
print(eval_accuracy)

ax2.set_title('Evaluation Loss', fontsize=25)
ax2.set_xlabel('Eval Step', fontsize=25)
ax2.plot(eval_loss.index, eval_loss.values)
ax3.set_title('Evaluation Accuracy', fontsize=25)
ax3.set_xlabel('Eval Step', fontsize=25)
ax3.plot(eval_accuracy.index, eval_accuracy.values)
fig.tight_layout(pad=5.0)

plt.show()
plt.savefig("loss.png", dpi=300)


