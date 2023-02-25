#!/usr/bin/env python
# coding: utf-8

# In[43]:


get_ipython().system('pip install imblearn')


# # Import libraries

# In[44]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from imblearn.over_sampling import SMOTE


# # Import DataSet

# In[2]:


test=pd.read_csv("E://Pune dataseince assignment//data science files//Naive_Baiyes//SalaryData_Test.csv")


# In[3]:


test


# # Data Description
# age --> age of a person
# 
# workclass --> A work class is a grouping of work
# 
# education -- >Education of an individuals
# 
# maritalstatus --> Marital status of an individulas
# 
# occupation --> occupation of an individuals
# 
# relationship -->
# 
# race --> Race of an Individual
# 
# sex --> Gender of an Individual
# 
# capitalgain --> profit received from the sale of an investment
# 
# capitalloss --> A decrease in the value of a capital asset
# 
# hoursperweek --> number of hours work per week
# 
# native --> Native of an individual
# 
# Salary --> salary of an individual

# # EDA(Exploratory data analysis)

# In[4]:


test.head()


# In[5]:


test.head()


# In[6]:


test.describe()


# In[7]:


test.columns


# In[8]:


test.isnull().any(axis=1)


# In[9]:


test.isnull().sum()


# In[10]:


test.shape


# In[11]:


test.dtypes


# In[12]:


test.drop(['native'],axis=1,inplace=True)


# In[13]:


test.head()


# ## define the categorical variables and Numeric variables

# In[14]:



categorical = [var for var in test.columns if test[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n', categorical)
print('\n')
numeric = [var for var in test.columns if test[var].dtype!='O']
print('There are {} Numeric variables\n'.format(len(numeric)))
print('The Numeric variables are :\n', numeric)


# In[15]:



categorical=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'Salary']
Continuous=['age', 'educationno', 'capitalgain', 'capitalloss', 'hoursperweek']


# In[16]:


for feature in Continuous:
  sns.displot(data = test, x=feature,height = 4, aspect = 2, palette='deep')


# In[17]:


for feature in Continuous:
  plt.figure()
  sns.violinplot(test[feature])
  plt.show()


# In[18]:


test.agg(["skew","kurt"])


# In[19]:


test['Salary'].value_counts()


# In[20]:


pd.crosstab(test['occupation'],test['Salary'])


# In[21]:


for i in numeric:
  plt.figure(figsize=(16,5))
  print("Skew: {}".format(test[i].skew()))
  print("Kurtosis: {}".format(test[i].kurtosis()))
  ax = sns.kdeplot(test[i],shade=True,color='g')
  plt.xticks([i for i in range(0,20,1)])
  plt.show()


# In[22]:


plt.figure(figsize=(20,7))
sns.heatmap(test.corr(),annot=True)


# In[23]:


sns.pairplot(test,hue='Salary')
plt.figure(figsize=(10,5))
plt.show()


# # Defining Dependent and Independent Variabl

# In[24]:


x = test.drop(["Salary"],axis=1)
y = test["Salary"]


# In[25]:


y


# # Encoding the Independent Variable and Dependent variable

# In[26]:


for columns in test.columns:#test.columns
  if test[columns].dtype=='object':
    print(columns)


# In[27]:


from sklearn.preprocessing import LabelEncoder
for col in categorical:
  le=LabelEncoder()
  le.fit(test[col])
  test[col]=le.transform(test[col])


# In[28]:


test.head()


# # Scaling the Data

# In[29]:


pd.Series(y).value_counts()


# In[30]:


scaler = MinMaxScaler()
model = scaler.fit(test)
scaled_data = model.transform(test)


# In[31]:


test1 = pd.DataFrame(scaled_data, columns = test.columns)
test1.head()


# In[32]:


x = test1.drop("Salary",axis=1)
y = test1.iloc[:,-1]


# In[33]:


test1['Salary'].value_counts()


# # As the data is highly imbalanced, we shall resample it and make it balanced

# In[45]:


from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE

#from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy='minority')

oversample=SMOTE()
x,y=oversample.fit_resample(x,y)


# In[46]:


y.value_counts()


# The testing dataset is ready to be feed in the model

# In[47]:


x.shape,y.shape


# In[48]:


pd.Series(y).value_counts()


# In[49]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3,random_state=42)


# In[50]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# # # Multinomial Naive Bayes
# 

# In[53]:


from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes

#Model Train
classifier_mb = MB()

#Model Test 

# Model Accuracy on train set
train_pred_m = classifier_mb.fit(x_train,y_train).predict(x_train)
accuracy_train_m = np.mean(train_pred_m == y_train)

# Model Accuracy on test set

test_pred_m = classifier_mb.fit(x_train,y_train).predict(x_test)
accuracy_test_m = np.mean(test_pred_m == y_test)


# In[54]:


accuracy_train_m


# In[55]:


accuracy_test_m


# In[56]:


test_pred_m


# # Gaussian Naive Bayes
# 

# In[58]:


from sklearn.naive_bayes import GaussianNB as GB

# Gaussian Naive Bayes

#Model Train
classifier_gb = GB()

#Model Test 

# Model Accuracy on train set

train_pred_g = classifier_gb.fit(x_train,y_train).predict(x_train)
accuracy_train_g = np.mean(train_pred_g == y_train)

# Model Accuracy on test set
test_pred_g = classifier_gb.fit(x_train,y_train).predict(x_test)
accuracy_test_g = np.mean(test_pred_g == y_test)


# In[59]:


accuracy_train_g 


# In[60]:


accuracy_test_g 


# In[61]:


test_pred_g 


# # confusion_matrix

# In[63]:


confusion_matrix(y_test,test_pred_m)


# In[64]:


print(classification_report(y_test, test_pred_m))


# In[65]:


# Confusion matrix
cm = confusion_matrix(y_test, test_pred_m)
plt.figure(figsize = (5,5))
sns.heatmap(cm, square=True, annot=True)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


# # AUC and ROC

# In[67]:


# AUC and ROC
from sklearn.metrics import roc_auc_score,roc_curve
print(f'Model AUC score: {roc_auc_score(y_test, test_pred_m)} \n\n')

fpr, tpr, thresholds = roc_curve(y_test, test_pred_m)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




