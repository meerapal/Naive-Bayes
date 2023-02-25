#!/usr/bin/env python
# coding: utf-8

# # train Data set

# # Import libaray

# In[3]:


get_ipython().system('pip install imblearn')


# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# In[5]:


train=pd.read_csv("E://Pune dataseince assignment//data science files//Naive_Baiyes//SalaryData_Train.csv")


# # Prepare a classification model using SVM for salary dat

# In[7]:


train


# # Data Description

# age --> age of a person
# 
# workclass	--> A work class is a grouping of work 
# 
# education	-- >Education of an individuals	
# 
# maritalstatus --> Marital status of an individulas
# 
# occupation	 --> occupation of an individuals
# 
# relationship --> 	
# 
# race -->  Race of an Individual
# 
# sex -->  Gender of an Individual
# 
# capitalgain -->  profit received from the sale of an investment
# 
# capitalloss	--> A decrease in the value of a capital asset
# 
# hoursperweek --> number of hours work per week	
# 
# native --> Native of an individual
# 
# Salary --> salary of an individual

# # EDA(Exploratory data analysis)
# 

# In[10]:


train.head()


# In[11]:


train.isnull().sum()


# In[12]:


train.isnull().any(axis=1)


# In[13]:


train.shape


# In[14]:


train.nunique()


# In[15]:


train.describe()


# In[16]:


train.columns


# In[17]:


train.dtypes


# In[18]:


train.drop(['native'],axis=1,inplace=True)


# In[19]:


train.head()


# # define the categorical variables and Numeric variables

# In[20]:


categorical=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'Salary']
Continuous=['age', 'educationno', 'capitalgain', 'capitalloss', 'hoursperweek']


# In[26]:


for feature in Continuous:
  plt.figure()
  sns.violinplot(train[feature])
  plt.show()


# In[27]:


train.agg(["skew","kurt"])


# In[28]:


train['Salary'].value_counts()


# In[29]:


pd.crosstab(train['occupation'],train['Salary'])


# In[30]:


for i in numeric:
  plt.figure(figsize=(16,5))
  print("Skew: {}".format(test[i].skew()))
  print("Kurtosis: {}".format(test[i].kurtosis()))
  ax = sns.kdeplot(train[i],shade=True,color='g')
  plt.xticks([i for i in range(0,20,1)])
  plt.show()


# In[31]:


plt.figure(figsize=(20,7))
sns.heatmap(train.corr(),annot=True)


# In[32]:



# scatter matrix to observe relation between every column attribute
pd.plotting.scatter_matrix(train,
                           figsize=[15,10],
                           diagonal='hist',
                           alpha=1,
                           s = 300, 
                           marker = '+',
                           edgecolor= ' red')
plt.show()


# In[33]:


sns.pairplot(train,hue='Salary')
plt.figure(figsize=(10,5))
plt.show()


# In[47]:


plt.figure(figsize=(10,7))
sns.countplot(y_train)
plt.show()


# # Encoding the Dependent Variable

# In[34]:


for columns in train.columns:#test.columns
  if train[columns].dtype=='object':
    print(columns)


# In[35]:


train.isnull()


# # Encoding the Dependent Variable and Independent Variable

# In[36]:


from sklearn.preprocessing import LabelEncoder
for col in categorical:
  le=LabelEncoder()
  le.fit(train[col])
  train[col]=le.transform(train[col])


# In[37]:


train.head()


# In[38]:


scaler = MinMaxScaler()
model = scaler.fit(train)
scaled_data = model.transform(train)


# In[39]:


train1 = pd.DataFrame(scaled_data, columns = train.columns)
train1.head()


# In[40]:


x = train1.drop("Salary",axis=1)
y = train1.iloc[:,-1]


# In[41]:


train1['Salary'].value_counts()


# # As the data is highly imbalanced, we shall resample it and make it balanced

# In[42]:


from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy='minority')

oversample=SMOTE()
x,y=oversample.fit_resample(x,y)
y.value_counts()


# # The testing dataset is ready to be feed in the model

# In[43]:


x.shape,y.shape


# In[44]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3,random_state=42)


# In[45]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# # Multinomial naive Bayes

# In[48]:


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


# In[49]:


accuracy_train_m


# In[50]:


accuracy_test_m


# In[51]:


test_pred_m


# # Gaussian Naive Bayes

# In[52]:


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


# In[53]:


accuracy_train_g 


# In[54]:


accuracy_test_g 


# In[55]:


test_pred_g 


# # confusion_matrix

# In[56]:


confusion_matrix(y_test,test_pred_m)


# In[ ]:





# In[58]:


# Confusion matrix
cm = confusion_matrix(y_test, test_pred_m)
plt.figure(figsize = (5,5))
sns.heatmap(cm, square=True, annot=True)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


# # Confusion matrix

# In[59]:


print(classification_report(y_test, test_pred_m))


# # AUC and ROC

# In[60]:


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




