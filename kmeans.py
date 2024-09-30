import csv
import random
import numpy as np
import pandas
import matplotlib.pyplot as plt

fields = []
rows = []

file="data-2.csv"
with open(file,"r") as f:
    read=csv.reader(f)
    fields=next(read)
    # prfloat(type(read))
    # prfloat(type(read[0])) #TypeError: '_csv.reader' object is not subscriptable


    for row in read:
        row.pop()
        rows.append(row)


x=np.array([i[0] for i in rows])
y=np.array([i[1] for i in rows])
# print(y[0:5])

# print(x[0:5])
# plt.scatter(x,y,color='green')
# plt.xlim(min(x), max(x))
# plt.ylim(min(y), max(y))
# plt.show()

df  = pandas.read_csv("data-3.csv")
  # plots all columns against index
df.plot(kind='scatter',x='x',y='y',color="red") 
plt.show()


k=3
clustercentroid=[]
PtsOfCent=[]
for i in range(k):
    clustercentroid.append(random.choice(rows))

ccx=np.array([i[0] for i in clustercentroid])
ccy=np.array([i[1] for i in clustercentroid])   

# plt.scatter(ccx,ccy,color='green')
# plt.xlim(min(x), max(x))
# plt.ylim(min(y), max(y))
# plt.show()

plt.show()

print(clustercentroid)

# def elud_dist
