# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ypNu-eIdIoK9av33NtuXxNFGIMeypYo-
"""



x,y=0,1

while y<50:
    print(y)
    x,y = y,x+y

lis=[1,2,3]
max=0
for i in lis:
  if i>max:
    print (lis)

a=0
b=1
n=7
for i in range (1,n):
  c=a+b
  a=b
  b=c
  print(c)

def fib(n):
  a=0
  b=1
  n=7
  for i in range (1,n):
    c=a+b
    a=b
    b=c
    print(c)

fib(50)

import numpy as np

zero_vector=np.zeros(20)
print (zero_vector)

array=np.array([1, 2, 3])
array

a=np.mat('[1 2;3 4]')

a=np.array([[1,3,5],[2,5,1],[2,3,8]])

b=np.mat('[3 4;5 6]')

np.dot(a,b)

x = [1, 2, 3]
y = [3, 7, 5]
np.cross(x, y)

a=np.random.randn(3,3)
b=np.random.randn(3,3)
resultant_dot=np.dot(a,b)
resultant_cross=np.cross(a,b)
resultant_add=np.add(a,b)

resultant_dot

resultant_cross

resultant_add



import pandas as pd
open=pd.read_csv("winequality_red.csv")
#open.head()
open.isnull().any()
f=open.fillna(open.mean())
f.isnull().any()
f.loc[:,['pH']]

import pandas as pd
n=pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
#n.head()
#n.isnull().values.any()
n = n.drop(columns="petal.width",axis=0)
#n.head()
n1 = n.iloc[1:5]
n1



new=n["sepal.length"]
new

final=pd.concat([open,n])

final.head()

final1=pd.concat([open,n],axis=1)
final1.head(2)

con=pd.concat([open,n],ignore_index=True)

con.head()

con.dropna(axis=0)
con

a

