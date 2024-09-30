import csv
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import cauchy

fields = []
rows = []

file="realestate.csv"
with open(file,"r") as f:
    read=csv.reader(f)
    fields=next(read)
    # prfloat(type(read))
    # prfloat(type(read[0])) #TypeError: '_csv.reader' object is not subscriptable


    for row in read:
        rows.append(row)

rows = [list([e[3],e[7],e[2] ]) for e in rows]
x=[float(i[0]) for i in rows[:200]]
y=[float(i[1]) for i in rows[:200]]

x=np.array([float(i)/float(max(x)) for i in x],dtype=np.float64)
y=np.array([float(i)/float(max(y))  for i in y],dtype=np.float64)
# z=np.array([i[2] for i in rows])

x_s = np.linspace(-5, 5, 50)
y_s = cauchy.pdf(x=x_s, loc=0, scale=1)
plt.scatter(x_s, y_s, s=100)
# plt.show()

# s = np.random.standard_cauchy(300)
# print(type(s))
# print(s[:10])
 
# Creating plot
# plt.scatter(x, y, color = "green")
# plt.show()

def costfunc(y,y_pred):
    z=y-y_pred
    return np.sum(z**2)/(2*len(y))

w=np.random.rand(6,1)
w=np.zeros((6,1))
b=0

feat=np.array([x_s,x_s**2,x_s**3,x_s**4,x_s**5,x_s**6]).transpose()
print(feat.shape)
print(x_s.shape)


steps=1000
alpha=0.000005
for j in range(steps):
    y_pred=np.matmul(feat,w)+b
    djdw0= np.sum(np.dot((y_pred-y_s),x_s))/len(y_s)
    djdw1= np.sum(np.dot((y_pred-y_s),x_s**2))/len(y_s)
    djdw2= np.sum(np.dot((y_pred-y_s),x_s**3))/len(y_s)
    djdw3= np.sum(np.dot((y_pred-y_s),x_s**4))/len(y_s)
    djdw4= np.sum(np.dot((y_pred-y_s),x_s**5))/len(y_s)
    djdw5= np.sum(np.dot((y_pred-y_s),x_s**6))/len(y_s)

    djdb= np.sum((y_pred-y_s))/len(y_s)


    w[0]= w[0] - alpha*(djdw0)
    w[1]= w[1] - alpha*(djdw1)
    w[2]= w[2] - alpha*(djdw2)
    w[3]= w[3] - alpha*(djdw3)
    w[4]= w[4] - alpha*(djdw4)
    w[5]= w[5] - alpha*(djdw5)

    b= b - alpha*djdb
    if j%500==0:
        print(costfunc(y_s,y_pred))

print(w)
print(b)
w = w[::-1]
prediction = np.append(w, [b])
x = np.linspace(0,1,100)
y = [np.polyval(prediction, i) for i in x]
plt.plot(x,y)


plt.show()





