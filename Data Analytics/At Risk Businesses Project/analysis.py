#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print("libs downloaded!")

#define path
file_path = 'C:/Users/tyler/OneDrive - SNHU/WGU/Analytics Programming/D598 Data Set.xlsx'
#import excel to dataframe
df=pd.read_excel(file_path)
print(df.head)

#check for duplicates
duplicates = df.duplicated()
print(df[duplicates])

#no duplicates it appears
#Group by State
grouped = df.groupby('Business State')
print(grouped)

#Descriptive stats (mean, median, min, and max)
grouped_stats = grouped.agg({
  'Total Long-term Debt':['mean','median','min','max'],
  'Total Equity': ['mean','median','min','max'],
  'Debt to Equity': ['mean','median','min','max'],
  'Total Liabilities': ['mean','median','min','max'],
  'Total Revenue': ['mean','median','min','max'],
  'Profit Margin': ['mean','median','min','max']
})
print(grouped_stats)

#Debt-to-income ratio = debt-to-income/revenue
df['Debt to Income Ratio']=df['Total Long-term Debt']/df['Total Revenue']
#filter to find businesses with negative debt-to-income ratio
negative_debt_to_income = df[df['Debt to Income Ratio']<0]
print(negative_debt_to_income)

#df['Debt to Income Ratio']=df['Total Long-term Debt']/df['Total Revenue']
#the code above completed the DF.
print(df)

#Scatterplot for debt-to-income ratio to demonstrate our spread
plt.figure(figsize=(10, 6)) 
plt.scatter(df['Business ID'], df['Debt to Income Ratio'], color='blue', alpha=0.6)
#title of visual and labels
plt.title('Debt to Income Ratio Scatter Plot')
plt.xlabel('Bussiness ID')
plt.ylabel('Debt to Income Ratio')
plt.show()

#Filter the data to include only Debt-to-Income ratios greater than 1. These could be of risk.
filtered_df = df[df['Debt to Income Ratio'] > 1]

# Create the bar chart
plt.figure(figsize=(12, 6))
plt.bar(filtered_df['Business ID'].astype(str), filtered_df['Debt to Income Ratio'], color='skyblue')

# Customize the plot with titles and labels
plt.title('Debt to Income Ratios Above 1 by Business ID')
plt.xlabel('Business ID')
plt.ylabel('Debt to Income Ratio')

# Rotate x-axis labels to make them readable if necessary
plt.xticks(rotation=90)

# Show the plot
plt.show()

#Boxplot to what is considered an outlier in this data. Which company's our at risk?
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Debt to Income Ratio'], color='lightblue')
plt.title('Boxplot of Debt to Income Ratios')
plt.xlabel('Debt to Income Ratio')
plt.show()


