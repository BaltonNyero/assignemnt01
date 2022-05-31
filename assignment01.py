#!/usr/bin/env python
# coding: utf-8

# <h2> Model to predict CO2Emissions in Cars</h2>

# <b> By Stephen Balton  NYERO </b>

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df =pd.read_csv("FuelConsumptionCo2.csv")
df.head()


# In[4]:


df.corr()


# In[5]:


df.tail()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


cdf =df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]


# In[9]:


cdf.head()


# <b> Plotting Each of these features </b>

# In[10]:


viz =cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
viz.hist()
viz.show()


# <b> Plotting each of these features vs the Emission, to see how linear their relation is </b>

# In[11]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color ="green")
plt.ylabel('CO2 Emission')
plt.xlabel('Engine Size')
plt.show()


# In[13]:


plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color ="green")
plt.ylabel('CO2 Emission')
plt.xlabel('CYLINDERS')
plt.show()


# In[14]:


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color ="red")
plt.ylabel('CO2 Emission')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.show()


# <b>Best STraight Line Graph "Co2 Emission Vs Engine Size" </b>

# <h2> Y = Mx +C where Y is Co2Emission and X is Engine size, m is the coefficeint and c is the y-intercept </h2>

# <b> Using sklearn package to model data.</b>

# In[ ]:





# In[ ]:




