#!/usr/bin/env python
# coding: utf-8

# # FIRST DATASET (BOSTON HOUSING)

# In[161]:


import pandas as pd
f=pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")


# In[30]:


#f.head()
#df=f['crim']
#df1=f['nox']


# In[22]:


#k=f.replace({2.18:nan}).head()


# In[23]:


#k.isnull().any()


# In[3]:


import numpy as np
from numpy import nan


# # HISTOGRAM

# In[160]:


import matplotlib.pyplot as plt
plt.hist(df)


# # SCATTER PLOT

# In[96]:


plt.scatter(df,df1,alpha=0.5)#, c=df, s=df, label=df1, alpha=0.3, edgecolors='none')


# In[33]:


f.head()


# # BAR PLOT

# In[46]:


plt.bar(df,df1)


# # boxplot

# In[57]:


plt.boxplot(df[2:85])


# # Violinplot

# In[62]:


plt.violinplot(df[2:15])


# # SECOND DATASET (IRIS)

# In[63]:


import pandas as pd
d=pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
d.head()


# In[64]:


vec1=d['sepal.length']


# In[65]:


vec2=d['petal.length']


# # Violinplot

# In[74]:


plt.violinplot(vec1)


# # Boxplot

# In[67]:


plt.boxplot(vec1)


# # Bar Plot

# In[71]:


plt.bar(vec1,vec2)


# # scatter plot

# In[149]:


plt.scatter(vec1,vec2,alpha=0.5,marker='o')#, c=df, s=df, label=df1, alpha=0.3, edgecolors='none')


# # Histogram

# In[81]:


plt.hist(vec1)


# # THIRD DATASET (winequality)

# In[82]:


data1=pd.read_csv("winequality_red.csv")


# In[83]:


data1.head()


# In[84]:


arm1=data1['fixed acidity']
arm2=data1['volatile acidity']


# # violinplot

# In[85]:


plt.violinplot(arm1)


# # boxplot

# In[86]:


plt.boxplot(arm1)


# # stackbar plot

# In[158]:


plt.bar(arm1,arm3,color='b')
plt.bar(arm1,arm2,color='r')
plt.show()


# # scatter plot

# In[147]:


plt.scatter(arm1,arm2,alpha=0.5,marker='^')


# # histogram

# In[90]:


plt.hist(arm1)


# In[92]:


plt.barh(arm1,arm2)


# In[93]:


N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2
colors = theta

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)


# In[121]:


arm3=data1['density']


# In[141]:


plt.scatter(arm1, arm2, s=arm2, marker='^', c=arm1)


# In[146]:


import numpy as np
np.random.seed(19680801)
N = 100
r0 = 0.6
x = 0.9 * np.random.rand(N)
y = 0.9 * np.random.rand(N)
area = (20 * np.random.rand(N))**2  # 0 to 10 point radii
c = np.sqrt(area)
r = np.sqrt(x ** 2 + y ** 2)
area1 = np.ma.masked_where(r < r0, area)
area2 = np.ma.masked_where(r >= r0, area)
plt.scatter(x, y, s=area1, marker='^', c=c)
plt.scatter(x, y, s=area2, marker='o', c=c)
# Show the boundary between the regions:
#theta = np.arange(0, np.pi / 2, 0.01)
#plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))

#plt.show()


# In[ ]:




