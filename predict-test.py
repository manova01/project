#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('autosave', '0')


# In[4]:


import requests


# In[5]:


url = 'http://localhost:9696/predict'


# In[6]:


customer={'id': 45701,
 'gender': 'female',
 'age': 72.0,
 'hypertension': 0,
 'heart_disease': 1,
 'ever_married': 'no',
 'work_type': 'self-employed',
 'Residence_type': 'rural',
 'avg_glucose_level': 124.38,
 'bmi': 23.4,
 'smoking_status': 'formerly_smoked'}


# In[7]:


customer


# In[ ]:


requests.post(url, json=customer).json()

