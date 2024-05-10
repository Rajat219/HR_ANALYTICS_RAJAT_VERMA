#!/usr/bin/env python
# coding: utf-8

# # Objective: Identify the best source of recruitment for a tech startup, based on previous data of candidate sources and recruitment strategies
# 
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#supress warnings
import warnings
warnings.filterwarnings("ignore")


# In[3]:


df_hr = pd.read_csv("C:/Users/hp/Desktop/desktop data/EDA/HR_project/Recruitment_Data.csv")


# In[4]:


df_hr.head()


# In[5]:


df_hr.shape


# In[6]:


print('data has {} rows and {} columns'.format(df_hr.shape[0], df_hr.shape[1]))


# # Data Understanding

# In[7]:


df_hr.describe()


# In[8]:


df_hr.nunique()


# In[9]:


df_hr.info()


# In[10]:


df_hr.duplicated().sum()


# In[11]:


df_hr.isna().sum()


# Column recruitment_source contains 205 null values.

# ### Handling NULL values in Categorical column

# In[12]:


df_hr['recruiting_source']=df_hr['recruiting_source'].fillna(df_hr['recruiting_source'].mode()[0])


# In[13]:


df_hr['recruiting_source'].mode()


# In[14]:


df_hr.isna().sum()


# In[15]:


df_hr['recruiting_source']


# # Exploratory Data Analysis

# ### Univariate Data Analysis

# In[16]:


plt.figure(figsize=(5,4))
sns.countplot(df_hr['recruiting_source'],data=df_hr)
plt.show()


# In[17]:


plt.figure(figsize=(5,4))
sns.countplot(df_hr['attrition'],data=df_hr)
plt.show()


# Data is highly imbalanced 

# In[18]:


#plt.figure(figsize=(5,4))
df_hr['sales_quota_pct'].plot(kind='box', title='Sales Quota Pct') 
plt.show()


# Check for outliers

# In[19]:


def Check_Outliers(data,columnList):

    plt.figure(figsize=[6,5])
    plt.subplots_adjust(wspace=0.4,hspace=0.3)

    for i,j in enumerate(columnList):
        plt.subplot(2,2,i+1)

        sns.boxplot(y=data[j])    

        plt.suptitle("\nChecking Outliers using Boxplot",fontsize=10,color="blue")
        plt.ylabel(None)
        plt.title(j,fontsize=10,color='brown')


# In[20]:


num_cols = ["performance_rating",	"sales_quota_pct"]

Check_Outliers(df_hr,num_cols)


# "Performance rating" and "Sales_quota_pct":Both these variables contain outliers as can be seen in the boxplot So, These outliers needs to be treated for these variables.

# In[21]:


# Defining user define function to treat outliers via capping and flooring

def Outlier_treatment(df,columnList):
    for i in columnList:
        q1 = df[i].describe()["25%"]
        q3 = df[i].describe()["75%"]
        IQR = q3 - q1

        upper_bound = q3 + 1.5*IQR
        lower_bound = q1 - 1.5*IQR

        # capping upper_bound
        df[i] = np.where(df[i] > upper_bound, upper_bound,df[i])

        # flooring lower_bound
        df[i] = np.where(df[i] < lower_bound, lower_bound,df[i])


# In[22]:


# Checking outliers for numerical variables other than target variable 
capping_cols = ["performance_rating","sales_quota_pct"]

# UDF 
Outlier_treatment(df_hr,capping_cols)


# # Bivariate Analysis 

# In[23]:


plt.figure(figsize=(16, 4))
sns.pairplot(data=df_hr,vars=num_cols,hue="attrition")                                  
plt.show()


# In[24]:


# Heatmap to show correlation between numerical variables
sns.heatmap(data=df_hr[num_cols].corr(),cmap="Blues",annot=True)
plt.show()


# In[25]:


# Boxplot with attrition as hue
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
sns.boxplot(y = 'performance_rating', x = 'attrition', data = df_hr)
plt.subplot(1,2,2)
sns.boxplot(y = 'sales_quota_pct', x = 'attrition', data = df_hr)
plt.show()


# # Identify groups in the dataset

# In[27]:


# Which recruiting source resulted highest sales_quota_pct?
Sales_Quota_Pct = df_hr.groupby('recruiting_source')[['sales_quota_pct']].mean().sort_values('sales_quota_pct')
Sales_Quota_Pct.plot(kind='bar')
plt.ylabel('Sales quota retainment (%)')
plt.show()


# Recruitment source applied online performed higher average sales_quota_pct than others.
# 

# In[28]:


# Which recruiting source resulted in highest perfromance rating?
Perf_rate= df_hr.groupby('recruiting_source')[['performance_rating']].mean().sort_values('performance_rating')
Perf_rate.plot(kind='bar')
plt.ylabel('performance_rating')
plt.show()


# This shows that the perfromance rating of the employees who were hired via "Campus" channel which is higher than others. However the differences are small.

# In[32]:


# Which hires had lower attrition rate?¶
# Another quality of hire metric you can consider is the attrition rate, or how often hires leave the company.

Att_rate= df_hr.groupby('recruiting_source')[['attrition']].mean().sort_values('attrition')
Att_rate.plot(kind='bar')
plt.ylabel('Attrition_rate')
plt.show()


# This shows how attrition rate is highest for hires coming through Search Firms, while lowest for hires coming from "Applied Online".

# ## Get Average Sales Numbers and Attrition Numbers

# ### 1. Print out the average Sales Number grouped by Recruiting Source

# In[30]:


Avg_Sales_number = df_hr.groupby('recruiting_source')[['sales_quota_pct']].mean()


# In[31]:


Avg_Sales_number


# ### 2. Print out the average Attrition Number grouped by Recruiting Source

# In[33]:


Avg_Att_Number= df_hr.groupby('recruiting_source')[['attrition']].mean()


# In[34]:


Avg_Att_Number


# ### 3. Sources that have high Sales numbers and low Attrition numbers.

# In[43]:


High_Sales_number = df_hr.groupby('recruiting_source')[['sales_quota_pct']].mean().sort_values('sales_quota_pct', ascending=False)


# In[52]:


High_Sales_number


# Recruitment source which have High Sales number is "Applied Online".

# In[54]:


Low_Att_Number= df_hr.groupby('recruiting_source')[['attrition']].mean().sort_values('attrition', ascending=True)


# In[55]:


Low_Att_Number


# Recruitment source which have low attrition number is "Applied Online".
