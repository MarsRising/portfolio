#!/usr/bin/env python
# coding: utf-8

# In[242]:


#Our Libraries
import pandas as pd # used for handling the dataset
import numpy as np #used for handling numbers

#Our tools
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder #encoding categorical data
from mlxtend.frequent_patterns import apriori, association_rules

print("Complete!")


# In[243]:


#create dataframe
df = pd.read_csv(r'C:\Users\tyler\OneDrive - SNHU\WGU\Data Preparation and Exploration\Megastore_Dataset_Task_3.csv')
print("df complete")
df.info()


# In[244]:


print(df.columns)


# In[245]:


#Knowing my question my Nominal Values can be OrderID and ProductName
#What could add to my analysis is OrderPriority and CustomerOrderSatisfaction
df = df.drop(columns=['Quantity', 'InvoiceDate', 'UnitPrice ', 'TotalCost ', 'Country', 'DiscountApplied', 'Region', 'ExpeditedShipping', 'PaymentMethod'])
df.info()


# In[246]:


print(df.head())


# In[247]:


#data clean
df['ProductName']=df['ProductName'].str.strip()
df.dropna(axis=0, subset=['OrderID'], inplace=True)
df['OrderID']=df['OrderID'].astype('str')


# In[248]:


#ordinal unique variabels
print(df['OrderPriority'].unique())
print(df['CustomerOrderSatisfaction'].unique())


# In[249]:


#ordinal encoding
#set order priority to category
df['OrderPriority'] = df['OrderPriority'].astype('str')
#Set order priority encoder
ordinal_order_pri = OrdinalEncoder(categories=[['Medium', 'High']])
#set customer satisfaction encoder
satis_encoder=OrdinalEncoder(categories=[['Very Dissatisfied', 'Dissatisfied', 'Prefer not to answer', 'Satisfied', 'Very Satisfied']])
#set ordinal columns
df['OrderPriority']=ordinal_order_pri.fit_transform(df[['OrderPriority']])
df['CustomerOrderSatisfaction']=satis_encoder.fit_transform(df[['CustomerOrderSatisfaction']])
#check
print(df[['OrderPriority']].head())
print(df[['CustomerOrderSatisfaction']].head())


# In[250]:


#Realize that for the apriori algorithm to work, I needed binary code to be encoded.
#with that being said I did decide to encode anything not satisfied to 0 and satisfied to 1.
#This seemed like the best choice, so we can seperate based on who was satisfied and who was not.
df['CustomerOrderSatisfaction'] = df['CustomerOrderSatisfaction'].map({
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1
})


# In[251]:


#check the encoding
print("OrderPriority encoding mapping:")
print(ordinal_order_pri.categories_)
print("CustomerOrderSatisfaction encoding mapping:")
print(satis_encoder.categories_)


# In[252]:


#cerate dictionary for these values, to check anytime
reverse_order_mapping = {i: category for i, category in enumerate(ordinal_order_pri.categories_[0])}
reverse_satisfaction_mapping = {i: category for i, category in enumerate(satis_encoder.categories_[0])}

#check
print("Reverse OrderPriority mapping:")
print(reverse_order_mapping)
print("Reverse CustomerOrderSatisfaction mapping:")
print(reverse_satisfaction_mapping)


# In[253]:


#ordinal unique variabels
print(df['OrderID'].unique())
print(df['ProductName'].unique())
print(df['Segment'].unique())


# In[254]:


#one hot encoding
df = pd.get_dummies(df, columns=[col for col in df.columns if 'ProductName' in col], drop_first=True)
#Label Encoding to maintain just one column
label_encoder = LabelEncoder()
df['Segment'] = label_encoder.fit_transform(df['Segment'])

#check
print(df.head())


# In[255]:


print(df_encoded.dtypes)


# In[256]:


print(df.columns)


# In[257]:


#transactionalize the data by orders (orderid)
df = df.groupby('OrderID').agg({
    'OrderPriority': 'first',
    'CustomerOrderSatisfaction': 'first',
    'Segment': 'first',
    **{col: 'sum' for col in df.columns if 'ProductName' in col}
})
print(df.head())


# In[258]:


#after transactionalizing the data OrderID is the index and no longer is a column
print(df.columns)


# In[259]:


#print(df['OrderID'].unique())
#print(df['ProductName'].unique())
print(df['OrderPriority'].unique())
print(df['CustomerOrderSatisfaction'].unique())


# In[260]:


#check for unique variables
product_columns = [col for col in df.columns if 'ProductName' in col]
for col in product_columns:
    print(f"{col}: {df[col].unique()}")


# In[261]:


# Loop through the product columns and fix the values
#product_columns = [col for col in df.columns if 'ProductName' in col]
#df[product_columns] = df[product_columns].map(lambda x: True if x > 0 else False)

# Step 4: Verify that the values are now only 0 or 1
#print(df.head())


# In[262]:


df.to_csv(r'C:\Users\tyler\OneDrive - SNHU\WGU\Data Preparation and Exploration\clean_dataset.csv')
print("csv saved")
df.to_excel(r'C:\Users\tyler\OneDrive - SNHU\WGU\Data Preparation and Exploration\clean_dataset.xlsx')
print("excel saved")


# In[263]:


df=df.drop(['OrderPriority', 'CustomerOrderSatisfaction', 'Segment'], axis=1)
print(df.info)


# In[264]:


#some values came as 2 by mistake... fix it to 1 instead for binary
df = df.map(lambda x: 1 if x > 0 else 0)


# In[265]:


#deprecation waring suggested using boolean values instead of binary
df = df.astype(bool)
#apriori algoritm
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
print(frequent_itemsets.head())


# In[266]:


#generate rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1, num_itemsets=2)
print(rules.head())
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


# In[267]:


print(rules[['support', 'confidence', 'lift']])


# In[268]:


top_3_rules=rules.sort_values('lift', ascending=False).head(3)
print(top_3_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


# In[ ]:




