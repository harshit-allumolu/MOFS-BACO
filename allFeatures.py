import pandas as pd
import numpy as np
import math
from evaluation import evaluation

filename = "datasets/" + input("Enter the dataset name : ")
dataset = pd.read_csv(filename)
y = dataset.iloc[:,0].to_numpy()
x = dataset.iloc[:,1:].to_numpy()
k = int(math.sqrt(len(x[0])))

lambda_ = max(1,(10**len(str(len(x[0])))//len(x[0])-1)//2) * 10 ** (-1*(len(str(len(x[0])))))
f,acc = evaluation(x,y,lambda_,k=k)
print("\n\nTotal number of features = {}".format(len(x[0])))
print("k value used for knn = {}".format(k))
print("\nAccuracy = {}%\n".format(round(acc*100,2)))