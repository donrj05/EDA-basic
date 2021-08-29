#!/usr/bin/env python
# coding: utf-8

# # EDA

# In[1]:




import numpy as np # linear algebra
import math
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_style("dark")
from matplotlib.pyplot import pie, axis, show
from sklearn import preprocessing
from scipy.stats import skew, boxcox
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


# In[2]:


main_data = pd.read_csv("./PS_20174392719_1491204439457_log.csv")


# In[3]:


df = main_data.copy()
df['percentage'] = np.where(df['amount']>df['oldbalanceOrg'], 100.0, (df['amount']/df['oldbalanceOrg'])*100)
df['dest']=df['nameDest'].astype(str).str[0]


# In[4]:


main_data.info()


# ### Plotting different types of transactions

# In[13]:


fig, ax = plt.subplots()
plt.rcParams['figure.figsize'] = (20, 16)
sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")
sns.countplot(x='type', data=main_data,palette="tab20_r",edgecolor='black')
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
plt.title('Total transactions',fontsize=25,color='#E43A36')
df_pie=main_data.groupby(['type']).size()
#axis('equal');
#pie(df_pie, labels=df_pie.index);
# Pie chart
plt.rcParams['figure.figsize'] = (8, 6)
labels = df_pie.index
sizes = df_pie
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0.02, 0.02, 0.02, 0.02,0.02)
#add colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#FAE959']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90,wedgeprops={"edgecolor":"0",'linewidth': 0.5,'linestyle': 'solid', 'antialiased': True})
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.title('% OF ALL TYPES OF TRANSACTIONS',fontsize=25,color='#E43A36')
plt.show()


# ### Checking the performance of simulator 

# In[6]:


y_true = list(main_data['isFraud'])
y_pred = list(main_data['isFlaggedFraud'])
cf_matrix = metrics.confusion_matrix(y_true, y_pred)
sns.heatmap(cf_matrix, annot=True, cmap='YlGnBu')
plt.xlabel("isFlaggedFraud")
plt.ylabel("isFraud")
plt.title("Confusion matrix for simulator results")


# *The Fraud Check System in place has performed poorly, catching almost negligible percentage of fraud transcation, 16 in total, whereas there are more than 8000 fraud transactions in total, but it has not wrongly flagged legit transactions*

# ### Boxplotting the various parameters where the system has flagged transactions

# In[7]:


# This code is taken from https://www.kaggle.com/netzone/eda-and-fraud-detection

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
tmp = main_data.loc[(main_data.type == 'TRANSFER'), :]

a = sns.boxplot(x = 'isFlaggedFraud', y = 'amount', data = tmp, ax=axs[0][0])
axs[0][0].set_yscale('log')
b = sns.boxplot(x = 'isFlaggedFraud', y = 'oldbalanceDest', data = tmp, ax=axs[0][1])
axs[0][1].set(ylim=(0,0.5e8 ))
c = sns.boxplot(x = 'isFlaggedFraud', y = 'oldbalanceOrg', data=tmp, ax=axs[1][0])
axs[1][0].set(ylim=(0, 3e7))
d = sns.regplot(x = 'oldbalanceOrg', y = 'amount', data=tmp.loc[(tmp.isFlaggedFraud ==1), :], ax=axs[1][1])
plt.show()


# ### Plotting the Frequency of the Fraud Transaction on a graph

# In[8]:


plt.figure(figsize=(18,8))
ax = plt.gca()
sns.lineplot(x=list(range(1,744)),y=df.groupby('step')['isFraud'].sum(),color='blue',alpha=0.5,linewidth=4)
plt.xlim(0,200)
plt.xlabel('Time lapse')
plt.ylabel('Number of fraudulent transactions per hour')
plt.title('A time series graph of frequency of fraudulent transactions',fontsize=25,color='#E43A36')


# ### Plotting types of transactions which fall under the Fraud category

# In[9]:


fig, ax = plt.subplots()
df_fraud=main_data[main_data['isFraud']==1]
sns.countplot(x='type',data=df_fraud, ax=ax,palette="tab20_r",edgecolor='black',alpha=0.75)
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .5)
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
plt.title('FRAUDULENT TRANSCATIONS',fontsize=25,color='#E43A36')
plt.show()


# *There are only two types of transcations which fall under fraud category as shown in the above bar plot*

# 
# 
# ### There are two types of transactions based on Recipients, One is normal customer and the other one is Merchant, with simple checking it was found that all the transactions pertaining with the Merchants were legit and none of them was found to be fraud, same is described in the below pie chart.
# 
# 

# In[10]:


list1=list(df[df['isFraud']==1].groupby(['dest']).size())
list2=list(df[df['isFraud']==0].groupby(['dest']).size())
newlist=list1+list2
newlist
plt.rcParams['figure.figsize'] = (8, 6)
labels = ['Customers(Fraud)','Customers(not Fraud)','Merchants']
sizes = newlist
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0.05,0.05, 0.05)
#add colors
colors = ['#ff9999','#ff9999','#66b3ff']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90,wedgeprops={"edgecolor":"0",'linewidth': 0.5,'linestyle': 'solid', 'antialiased': True})
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.title('Total transactions distribution on types of recipients',fontsize=25,color='#E43A36',)
plt.show()


# *There are in total about 0.1% transactions and all are in the Customer's Category*

# ### Plotting heatmap to check correlation between different parameters

# In[11]:


df_heatmap = main_data[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud','isFlaggedFraud']]
sns.heatmap(df_heatmap.corr(), cmap="YlGnBu", annot=True)
plt.title('CORRELATION HEATMAP OF ALL PARAMS',fontsize=25,color='#E43A36')


# *Fraud category has more dependence on Amount category, we will try to plot it*

# ### Scatter Plot between Amount and Balance, highlighting the fraud transaction

# In[12]:


newscatplot=df[df['isFraud']==1]
plt.figure(figsize=(8,8))
ax = plt.gca()
ax.set_ylim(0,2*1e7)
ax.set_xlim(0,2*1e7)
df.plot.scatter(x='oldbalanceOrg',y='amount', ax=ax,edgecolors='black',s=100,alpha=0.1,label="Legit transaction")
newscatplot.plot.scatter(x='oldbalanceOrg',y='amount', color='#FCD735', ax=ax,edgecolors='red',s=100,alpha=0.1,label="Fraud transcation")
plt.title('Amount vs Balance',fontsize=25,color='#E43A36')


# There are various inferences from the above graph
# * *Firstly,The fraud line is inclined equally from both axis which implies that whenever the amount dealt was equal to the balance, it was most likely a Fraud*
# * *After a while the fraud line hits a constant limit, which implies that there is limit in some form to the maximum amount which can be CASH OUT or TRANSFER, hence there are no fraud amounts which surpass this limit*

# In[ ]:




