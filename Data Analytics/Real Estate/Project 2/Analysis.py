#Import necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, pointbiserialr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import RFE
print("libs downloaded!")


# In[198]:


#define path
file_path = 'C:/Users/tyler/OneDrive - SNHU/WGU/Statistical Data Mining/D600 Task 2 Dataset 1 Housing Information.csv'
#import csv to dataframe
df=pd.read_csv(file_path)
print(df.head)


# In[199]:


duplicates = df[df['ID'].duplicated()]
print(duplicates)


# In[200]:


#drop irrelevatn columns
df.drop(columns=["ID"], inplace=True)
df.info()


# In[201]:


#was not. Update it to categorical
df['IsLuxury'] = df['IsLuxury'].astype('category')
print('luxury')


# In[202]:


#numeric columns
numeric_df = df.select_dtypes(include=['number'])
#Point-Biseral Correlation for our dependent binary variable correlation to continuous variables
for col in numeric_df.columns:
    corr, _ = pointbiserialr(df['IsLuxury'], df[col])
    print(f"Correlation between IsLuxury and {col}: {corr}")


# In[203]:


#get rid of insignificant correlations
df = df.drop(columns=['BackyardSpace', 'CrimeRate', 'SchoolRating', 'AgeOfHome', 'DistanceToCityCenter', 'EmploymentRate', 'PropertyTaxRate', 'RenovationQuality', 'LocalAmenities', 'TransportAccess', 'Floors', 'Windows'])
df.info()


# In[204]:


#numeric columns
numeric_df = df.select_dtypes(include=['number'])
#Point-Biseral Correlation for our dependent binary variable correlation to continuous variables
for col in numeric_df.columns:
    corr, _ = pointbiserialr(df['IsLuxury'], df[col])
    print(f"Correlation between IsLuxury and {col}: {corr}")


# In[205]:


# Select the continuous independent variables
continuous_vars = ['Price', 'SquareFootage', 'NumBathrooms', 'NumBedrooms', 'PreviousSalePrice']

#constant
x = df[continuous_vars]
x = sm.add_constant(x)

#Variance Inflation Factor
vif_data = pd.DataFrame()
vif_data["Variable"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print(vif_data)


# In[206]:


#categorical variables next
categorical_df = df.select_dtypes(include=['category', 'object'])

#Chi-square test
for column in categorical_df.columns:
    if column != 'IsLuxury':
        contingency_table = pd.crosstab(df[column], df['IsLuxury'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-Square test for {column}:")
        print(f"  p-value: {p}")
        if p < 0.05:
            print(f"  The variable {column} is significantly related to IsLuxury.")
        else:
            print(f"  The variable {column} is NOT significantly related to IsLuxury.")
        print("\n")


# In[207]:


#get rid of insignificant correlations
df = df.drop(columns=['Fireplace', 'HouseColor', 'Garage'])
df.info()


# In[208]:


#NO SCIENTIFIC NOTATION
pd.set_option('display.float_format', '{:,.2f}'.format)
#DESCRIPTIVE STATISTICS
df.describe(include='all')


# In[209]:


#Univariate analysis Price
sns.histplot(df['Price'], kde=False, bins=20, color='skyblue')
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
#boxplot
sns.boxplot(x=df['Price'], color='lightgreen')
plt.title('Boxplot of Price')
plt.xlabel('Price')
plt.show()


# In[141]:


#SquareFootage
#Histogram
sns.histplot(df['SquareFootage'], kde=False, bins=20, color='skyblue')
plt.title('Distribution of Square Footage')
plt.xlabel('Square Footage')
plt.ylabel('Frequency')
plt.show()
#boxplot
sns.boxplot(x=df['SquareFootage'], color='lightgreen')
plt.title('Boxplot of Square Footage')
plt.xlabel('Square Footage')
plt.show()


# In[210]:


#NumBathrooms
#Histogram
sns.histplot(df['NumBathrooms'], kde=False, bins=20, color='skyblue')
plt.title('Distribution of Number of Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Frequency')
plt.show()
#boxplot
sns.boxplot(x=df['NumBathrooms'], color='lightgreen')
plt.title('Boxplot of NumBathrooms')
plt.xlabel('Number of Bathrooms')
plt.show()


# In[143]:


#NumBedrooms
#Histogram
sns.histplot(df['NumBedrooms'], kde=False, bins=20, color='skyblue')
plt.title('Distribution of Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Frequency')
plt.show()
#boxplot
sns.boxplot(x=df['NumBedrooms'], color='lightgreen')
plt.title('Boxplot of NumBedrooms')
plt.xlabel('Number of Bedrooms')
plt.show()


# In[211]:


#PreviousSalePrice
#Histogram
sns.histplot(df['PreviousSalePrice'], kde=False, bins=20, color='skyblue')
plt.title('Distribution of PreviousSalePrice')
plt.xlabel('PreviousSalePrice')
plt.ylabel('Frequency')
plt.show()
#boxplot
sns.boxplot(x=df['PreviousSalePrice'], color='lightgreen')
plt.title('Boxplot of PreviousSalePrice')
plt.xlabel('PreviousSalePrice')
plt.show()


# In[212]:


#IsLuxury CATEGORICAL
sns.countplot(x='IsLuxury', data=df)
plt.title('IsLuxury Distribution')
plt.xlabel('IsLuxury')
plt.ylabel('Count')
plt.show()


# In[213]:


#bivariate analysis
for var in continuous_vars:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='IsLuxury', y=var, data=df, hue=df['IsLuxury'], legend=False)
    plt.title(f'Boxplot of {var} vs IsLuxury')
    plt.xlabel('IsLuxury')
    plt.ylabel(var)
    plt.show()


# In[214]:


#Standardization
scaler = StandardScaler()
numeric_columns = ['Price', 'SquareFootage', 'NumBathrooms', 'NumBedrooms', 'PreviousSalePrice']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print('Complete!')


# In[215]:


print(df.isnull().sum())


# In[216]:


x = df.drop('IsLuxury', axis=1)
y = df['IsLuxury']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print('Complete!')


# In[217]:


#logistic regression
model = LogisticRegression()
model.fit(x_train, y_train)
#predict
y_pred = model.predict(x_test)
#accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# In[218]:


#optimie with backward stepwise elimination
#set constant
x = sm.add_constant(x)
model = sm.Logit(y_train, x_train).fit()
#BACKWARD STEPWISE ELIMINATION
def backward_elimination(x, y, significance_level=0.05):
    initial_features = x.columns.tolist()
    while len(initial_features) > 0:
        model = sm.Logit(y, x[initial_features]).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()
        # If p-value is greater than the significance level, remove the corresponding feature
        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            initial_features.remove(excluded_feature)
            print(f"Excluded {excluded_feature} with p-value {max_p_value}")
        else:
            break
    
    return initial_features

#Run Backward Stepwise Elimination
selected_features = backward_elimination(x_train, y_train)
#optimized
optimized_model = sm.Logit(y_train, x_train[selected_features]).fit()
print(optimized_model.summary())


# In[219]:


#AIC, BIC, Pseudo R2
print(f"AIC: {optimized_model.aic}")
print(f"BIC: {optimized_model.bic}")
print(f"Pseudo R-squared: {optimized_model.prsquared}")


# In[220]:


#make predictions
y_pred_prob = optimized_model.predict(x_test[selected_features])
y_pred = (y_pred_prob >= 0.5).astype(int)
#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
#accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


# In[221]:


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# In[222]:


print(df['IsLuxury'].unique())


# In[223]:


#intercept = optimized_model.params['const']  # Retrieve the intercept (constant term)
print(optimized_model.params)


# In[224]:


#training set
y_train_pred_prob = optimized_model.predict(x_train[selected_features])
#binary class labels
y_train_pred = (y_train_pred_prob >= 0.5).astype(int)
#Confusion matrix
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
print("Training Set Confusion Matrix:")
print(conf_matrix_train)
#accuracy for the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Set Accuracy: {train_accuracy:.4f}")


# In[225]:


train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
print(f"Training Set Precision: {train_precision:.4f}")
print(f"Training Set Recall: {train_recall:.4f}")
print(f"Training Set F1-Score: {train_f1:.4f}")
