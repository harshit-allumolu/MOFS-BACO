import pandas as pd
from evaluation import evaluation

dataset = pd.read_csv("datasets/ionosphere.csv")
y = dataset.iloc[:,0].to_numpy()
x = dataset.iloc[:,1:].to_numpy()


fitness = evaluation(x,y)

acc = fitness * (1 + 0.01 * len(x[0]))

print(acc*100)
print(len(x[0]))