#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df =pd.read_csv('healthcare-dataset-stroke-data.csv')


# In[3]:


df.head()


# # Data Cleaning

# In[4]:


df.dtypes


# In[5]:


strings = list(df.dtypes[df.dtypes=='object'].index)
strings


# In[6]:


for col in strings:
    df[col] = df[col].str.lower().str.replace(' ' ,'_')


# In[7]:


df.head()


# In[8]:


for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()


# In[9]:


# Check missing values
df.isnull().sum()


# In[10]:


# replace missing values with zero
df.fillna(0,inplace=True)


# In[11]:


df.isnull().sum()


# In[12]:


df.gender.value_counts()


# In[13]:


# Deleting rows where where gender column equals 'other'
df = df[df['gender'] != 'other']


# In[14]:


df.gender.value_counts()


# In[15]:


numerical_features = df[['bmi', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'stroke']]


# In[16]:


corrolation_matrix=numerical_features.corr()


# In[17]:


max_corr = corrolation_matrix.abs().unstack().sort_values(ascending=False)
top_corr = max_corr[max_corr <1].head(2)

print(top_corr)


# In[18]:


# plot between mode_of_shipment and reached_on_time_y_n
sns.countplot(y=df['age'], hue=df['hypertension'])



# # Setting Up Validation Frame Work

# In[18]:


# Perform train/validation/test using sklearn
from sklearn.model_selection import train_test_split


# In[19]:


# Divide the data into train,validation,test
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=42)


# In[20]:


len(df_full_train), len(df_test)


# In[21]:


df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state=42)


# In[22]:


len(df_train), len(df_val), len(df_test)


# In[23]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[24]:


y_train = df_train.stroke.values
y_val = df_val.stroke.values
y_test = df_test.stroke.values


# In[25]:


del df_train['stroke']
del df_val['stroke']
del df_test['stroke']


# # Data Analysis
# 

# In[26]:


df_full_train = df_full_train.reset_index(drop=True)


# In[27]:


# calculate the stroke rate
df_full_train.stroke.value_counts(normalize = True)


# In[28]:


global_stroke_rate= df_full_train.stroke.mean()
round(global_stroke_rate, 2)


# In[38]:


categorical_features=['smoking_status', 'Residence_type', 'ever_married', 'gender']


# # Mutual Information

# In[29]:


# mutual information : concept from information theory that tells how much we can learn about value
from sklearn.metrics import mutual_info_score


# In[30]:


mutual_info_score(df_full_train.stroke, df_full_train.age)


# In[31]:


mutual_info_score(df_full_train.stroke, df_full_train.gender)


# In[32]:


# apply mutual information to the wholle data
def mutual_info_stroke_score(series):
    return mutual_info_score(series, df_full_train.stroke)


# In[39]:


mi = df_full_train[categorical_features].apply(mutual_info_stroke_score)


# In[40]:


mi.sort_values(ascending=False)


# # One Hot Encoding

# In[45]:


from sklearn.feature_extraction import DictVectorizer


# In[46]:


train_dicts = df_train.to_dict(orient='records')


# In[47]:


train_dicts[0]


# In[48]:


dv = DictVectorizer(sparse=False)


# In[50]:


X_train = dv.fit_transform(train_dicts)


# In[51]:


val_dicts = df_val.to_dict(orient='records')


# In[52]:


X_val = dv.transform(val_dicts)


# # Logistic Regression

# In[53]:


from sklearn.linear_model import LogisticRegression


# In[54]:


model = LogisticRegression(solver = 'liblinear', C = 10, max_iter=1000,random_state=42)


# In[57]:


model.fit(X_train, y_train)


# In[58]:


model.intercept_[0]


# In[59]:


# this is the weight
model.coef_[0].round(3)


# In[60]:


# Hard prediction predict 0 and 1
model.predict(X_train)


# In[61]:


# soft prediction
model.predict_proba(X_train)


# In[62]:


# using it on validation dataset
y_pred = model.predict_proba(X_val)[:,1]


# In[63]:


stroke_decision = (y_pred>= 0.5)


# In[64]:


accuracy = (y_val==stroke_decision).mean()


# In[65]:


print(round(accuracy, 2))


# In[68]:


df_pred = pd.DataFrame()
df_pred['probability']= y_pred
df_pred['prediction'] = stroke_decision.astype(int)
df_pred['actual'] = y_val
df_pred['correct'] = df_pred.prediction==df_pred.actual


# In[69]:


df_pred


# In[70]:


len(X_val)


# In[71]:


(y_val == stroke_decision).sum()


# In[72]:


970/1022


# In[74]:


thresholds=np.linspace(0, 1,21)
scores= []
for t in thresholds:
    stroke_decision = (y_pred >=t)
    score = (y_val == stroke_decision).mean()
    print('%.2f%.3f'%(t, score))
    scores.append(score)


# In[75]:


plt.plot(thresholds,scores)


# In[76]:


from sklearn .metrics import accuracy_score


# In[77]:


accuracy_score(y_val, y_pred >=0.5)


# In[83]:


test_dicts = df_test.to_dict(orient='records')


# In[84]:


X_test = dv.transform(test_dicts)


# In[78]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[80]:


alpha_values = [0, 0.01, 0.1, 1, 10]


# In[81]:


rmse_score = {}


# In[82]:


for alpha in alpha_values:
    model=Ridge(alpha= alpha, solver = 'sag', random_state=42)
    model.fit(X_train, y_train)


# In[86]:


y_pred=model.predict(X_val)


# In[87]:


rmse = round(sqrt(mean_squared_error(y_val, y_pred)), 3)
rmse_score[alpha] = rmse


# In[89]:


for alpha, rmse in rmse_score.items():
    print(f'Alpha={alpha}: RMSE ={rmse}')


# In[93]:


from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer

# Assuming df_full_train is your training dataset with features and target

# Define the KFold object with 5 splits, shuffling the data, and setting a random seed
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Initialize a list to store AUC scores for each fold
auc_scores = []

# Initialize DictVectorizer
vectorizer = DictVectorizer()

# Iterate over different folds of df_full_train
for train_index, val_index in kf.split(df_full_train):
    # Split the data into train and validation sets based on the fold indices
    train_data, val_data = df_full_train.iloc[train_index], df_full_train.iloc[val_index]
    
    # Separate the features (X) and the target (y) variables for train and validation
    X_train, y_train = train_data.drop(columns=['stroke']), train_data['stroke']
    X_val, y_val = val_data.drop(columns=['stroke']), val_data['stroke']
    
    # Convert feature dataframes to dictionaries and then use DictVectorizer
    X_train_dict = X_train.to_dict(orient='records')
    X_val_dict = X_val.to_dict(orient='records')
    
    X_train_encoded = vectorizer.fit_transform(X_train_dict)
    X_val_encoded = vectorizer.transform(X_val_dict)
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_train_encoded, y_train)
    
    # Predict probabilities on the validation set
    y_pred_proba = model.predict_proba(X_val_encoded)[:, 1]
    
    # Calculate the AUC score and append it to the list
    auc_score = roc_auc_score(y_val, y_pred_proba)
    auc_scores.append(auc_score)

# Calculate and print the mean AUC score across all folds
mean_auc = sum(auc_scores) / len(auc_scores)
print("Mean AUC:", mean_auc)


# In[94]:


# Define the C values to iterate over
C_values = [0.01, 0.1, 0.5, 10]

# Initialize KFold with the same parameters as previously
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Initialize lists to store mean and std scores
mean_scores = []
std_scores = []

# Iterate over different C values
for C in C_values:
    # Initialize a list to store AUC scores for each fold
    auc_scores = []
    
    # Iterate over different folds of df_full_train
    for train_index, val_index in kf.split(df_full_train):
        # Split the data into train and validation sets based on the fold indices
        train_data, val_data = df_full_train.iloc[train_index], df_full_train.iloc[val_index]

        # Separate the features (X) and the target (y) variables for train and validation
        X_train, y_train = train_data.drop(columns=['stroke']), train_data['stroke']
        X_val, y_val = val_data.drop(columns=['stroke']), val_data['stroke']

        # Convert feature dataframes to dictionaries and then use DictVectorizer
        X_train_dict = X_train.to_dict(orient='records')
        X_val_dict = X_val.to_dict(orient='records')

        X_train_encoded = vectorizer.fit_transform(X_train_dict)
        X_val_encoded = vectorizer.transform(X_val_dict)

        # Initialize and train the Logistic Regression model with the current C value
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_train_encoded, y_train)

        # Predict probabilities on the validation set
        y_pred_proba = model.predict_proba(X_val_encoded)[:, 1]

        # Calculate the AUC score and append it to the list
        auc_score = roc_auc_score(y_val, y_pred_proba)
        auc_scores.append(auc_score)
    
    # Calculate the mean and standard deviation of AUC scores for the current C value
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    # Append the mean and std scores to the respective lists (rounded to 3 decimal digits)
    mean_scores.append(round(mean_auc, 3))
    std_scores.append(round(std_auc, 3))

# Print the results for each C value
for i, C in enumerate(C_values):
    print(f"C = {C}: Mean AUC = {mean_scores[i]}, Std = {std_scores[i]}")


# In[100]:


import pickle


# In[95]:


output_file = f'model_C={C}.bin'
output_file


# In[96]:


f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()


# In[101]:


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


# In[102]:


model_file = 'model_C=10.bin'


# In[103]:


with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[104]:


dv, model



# In[105]:


X =dv.transform([customer])


# In[106]:


model.predict_proba(X)[0,1]


# In[ ]:




