#Import necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
print("libs downloaded!")


# In[449]:


#define path
file_path = 'C:/Users/tyler/OneDrive - SNHU/WGU/Statistical Data Mining/D600 Task 3 Dataset 1 Housing Information.csv'
#import excel to dataframe
df=pd.read_csv(file_path)
print(df.head)


# In[450]:


#drop irrelevatn columns
df.drop(columns=["ID"], inplace=True)
df.info()


# In[451]:


df.describe(include='all')


# In[452]:


#use jsut the numeric variables to find significinat continuous variables
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix=numeric_df.corr()
print(correlation_matrix)


# In[453]:


#get rid of insignificant correlations
df = df.drop(columns=['BackyardSpace', 'CrimeRate', 'SchoolRating', 'AgeOfHome', 'DistanceToCityCenter', 'EmploymentRate', 'PropertyTaxRate', 'LocalAmenities', 'TransportAccess', 'Floors', 'Windows'])
df.info()


# In[454]:


#xheck pearson correlation jsut for the significant ones
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix=numeric_df.corr()
print(correlation_matrix)


# In[455]:


#t-test for binary variables 
fireplace_yes = df[df['Fireplace'] == 'Yes']['Price']
fireplace_no = df[df['Fireplace'] == 'No']['Price']
Garage_yes = df[df['Garage'] == 'Yes']['Price']
Garage_no = df[df['Garage'] == 'No']['Price']
IsLuxury_yes = df[df['IsLuxury'] == 1]['Price']
IsLuxury_no = df[df['IsLuxury'] == 0]['Price']

t_stat, p_value = ttest_ind(fireplace_yes, fireplace_no)
print(f"T-test p-value for 'Fireplace' and 'Price': {p_value}")
t_stat, p_value = ttest_ind(Garage_yes, Garage_no)
print(f"T-test p-value for 'Garage' and 'Price': {p_value}")
t_stat, p_value = ttest_ind(IsLuxury_yes, IsLuxury_no)
print(f"T-test p-value for 'IsLuxury' and 'Price': {p_value}")


# In[456]:


#ANOVA due to multiple categorical values in house colors
house_colors = [df[df['HouseColor'] == color]['Price'] for color in df['HouseColor'].unique()]
f_stat, p_value = f_oneway(*house_colors)
print(f"ANOVA p-value for 'HouseColor' and 'Price': {p_value}")


# In[457]:


#Out of the 4 categorical values the only 2 significant variables are Fireplace and IsLuxury DROP the other two
df.drop(columns=['HouseColor', 'Garage'], inplace=True)
df.info()


# In[458]:


#Now I am down to simply the variables that have the most significance.
#Before moving forward Variance Inflation Factor is important to handle milticollinearity
#VIF=1 NO CORRELATION     VIF>1 CORRELATED BUT NOT TOO STRONG   VIF>5 HIGH MULTICOLLINEARITY
cont_vars = df[['SquareFootage', 'NumBathrooms', 'NumBedrooms', 'RenovationQuality', 'PreviousSalePrice']]
#constraint
X = add_constant(cont_vars)
#Calculate VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)


# In[459]:


#ALL VARIABLES ARE GOOD! No Multicollinearity
#check isluxury is set up accurately
print(df.dtypes)
#was not. Update it to categorical
df['IsLuxury'] = df['IsLuxury'].astype('category')
#DESCRIPTIVE STATISTICS
df.describe(include='all')


# In[460]:


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


# In[461]:


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


# In[462]:


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


# In[463]:


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


# In[464]:


#RenovationQuality
#Histogram
sns.histplot(df['RenovationQuality'], kde=False, bins=20, color='skyblue')
plt.title('Distribution of RenovationQuality')
plt.xlabel('RenovationQuality')
plt.ylabel('Frequency')
plt.show()
#boxplot
sns.boxplot(x=df['RenovationQuality'], color='lightgreen')
plt.title('Boxplot of RenovationQuality')
plt.xlabel('RenovationQuality')
plt.show()


# In[465]:


#Fireplace CATEGORICAL
sns.countplot(x='Fireplace', data=df)
plt.title('Fireplace Distribution')
plt.xlabel('Fireplace')
plt.ylabel('Count')
plt.show()


# In[466]:


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


# In[467]:


#IsLuxury CATEGORICAL
sns.countplot(x='IsLuxury', data=df)
plt.title('IsLuxury Distribution')
plt.xlabel('IsLuxury')
plt.ylabel('Count')
plt.show()


# In[468]:


continuous_vars = ['SquareFootage', 'NumBathrooms', 'NumBedrooms', 'RenovationQuality', 'PreviousSalePrice']

# Create scatter plots for each continuous variable against Price
for var in continuous_vars:
    plt.figure(figsize=(6, 4))
    sns.lmplot(x=var, y='Price', data=df)
    sns.scatterplot(x=df[var], y=df['Price'])
    plt.title(f'Scatterplot of {var} vs Price')
    plt.xlabel(var)
    plt.ylabel('Price')
    plt.show()


# In[469]:


#Bivariate for my categoricals 
sns.boxplot(x='Fireplace', y='Price', data=df)
plt.title('Boxplot of Price by Fireplace')
plt.xlabel('Fireplace')
plt.ylabel('Price')
plt.show()
sns.boxplot(x='IsLuxury', y='Price', data=df)
plt.title('Boxplot of Price by IsLuxury')
plt.xlabel('IsLuxury')
plt.ylabel('Price')
plt.show()


# In[470]:


#The boxes in the Fireplac, as well as Medians, are very close.
#I will remove Fireplace as this may not be necessarily as strong as first expected, and I would like to not complicate my model
df.drop(columns=["Fireplace"], inplace=True)
df.info()


# In[471]:


#Standardization
scaler = StandardScaler()
numeric_columns = ['SquareFootage', 'NumBathrooms', 'NumBedrooms', 'RenovationQuality', 'PreviousSalePrice']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print('Complete!')


# In[472]:


#Principal Component Analysis
pca = PCA()
x_pca = pca.fit_transform(df[numeric_columns])
print('pca')


# In[473]:


#PCA MATRIX
components = pca.components_
print("Principal Components (Matrix):")
print(components)


# In[474]:


components_df = pd.DataFrame(pca.components_, columns=numeric_columns, index=[f'PC{i+1}' for i in range(len(pca.components_))])
print("Principal Components Loadings:")
print(components_df)


# In[475]:


#Variance Ratio
explained_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.show()


# In[476]:


#EXPLAINED VARIANCE
eigenvalues = pca.explained_variance_
#eigenvalues greater than 1
components_to_keep = np.sum(eigenvalues > 1)
print(f"Number of components to retain (Kaiser rule): {components_to_keep}")


# In[477]:


print(df.isnull().sum())


# In[478]:


#PCA setting
pca = PCA(n_components=3)
x_pca = pca.fit_transform(df[numeric_columns])


# In[479]:


x = x_pca
y = df['Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_with_const = sm.add_constant(x_train)
print('Complete!')


x_train_with_const_df = sm.add_constant(x_train)
x_train_with_const_df = pd.DataFrame(x_train_with_const, columns=['const'] + [f'PC{i+1}' for i in range(x_train.shape[1])])
x_train_with_const_df.index = y_train.index

# Print the indices of both y_train and x_train_with_const
print("y_train indices:")
print(y_train.index)

print("\nx_train_with_const indices:")
print(x_train_with_const_df.index)


# In[480]:


#I am choosing to use BACKWARD STEPWISE ELIMINATION

threshold = 0.05
current_features = x_train_with_const_df.columns.tolist()

#BACKWARD STEPWISE ELIMINATION
while True:
    #model
    model = sm.OLS(y_train, x_train_with_const_df[current_features]).fit()
    #P_VALUES
    p_values = model.pvalues
    #highest p-value
    max_p_value = p_values.max()
    
    #Maximum p-value is greater than the threshold, remove the feature with the highest p-value
    if max_p_value > threshold:
        feature_to_remove = p_values.idxmax()
        print(f"Removing {feature_to_remove} with p-value {max_p_value:.4f}")
        #drop feature from the list of current features
        current_features.remove(feature_to_remove)       
    else:
        #no feature has a p-value greater than the threshold, break
        break

print("\nSelected features after backward elimination:")
print(current_features)

#linear regression model with the selected features
x_train_selected = x_train_with_const_df[current_features]
x_test_selected_df = pd.DataFrame(x_test, columns=[f'PC{i+1}' for i in range(x_test.shape[1])])

#only the relevant features from x
x_test_selected = x_test_selected_df[current_features[1:]]  # Exclude 'const' from test data

#add constant to test set
x_test_selected_with_const = sm.add_constant(x_test_selected)

#Linear regression model/selected features
lr_model = sm.OLS(y_train, x_train_selected).fit()

#Predictions test set
y_pred = lr_model.predict(x_test_selected_with_const)

print(lr_model.summary())
print(f'\nR^2: {r2_score(y_test, y_pred):.4f}')
print(f'Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.4f}')
print(f'Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}')


# In[481]:


#train model with the selected features
lr_model = sm.OLS(y_train, x_train_selected).fit()
print(lr_model.summary())

#extract deets
adjusted_r2 = lr_model.rsquared_adj
r2 = lr_model.rsquared
f_statistic = lr_model.fvalue
p_value_f_statistic = lr_model.f_pvalue
coefficients = lr_model.params
p_values = lr_model.pvalues

print(f"Adjusted R²: {adjusted_r2:.4f}")
print(f"R²: {r2:.4f}")
print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value of F-statistic: {p_value_f_statistic:.4f}")

print("\nCoefficient Estimates:")
print(coefficients)

print("\nP-values of each independent variable:")
print(p_values)


# In[482]:


#linear regression model/selected features
lr_model = sm.OLS(y_train, x_train_selected).fit()

#Predictions training set
y_train_pred = lr_model.predict(x_train_selected)

#MSE training set
mse_train = mean_squared_error(y_train, y_train_pred)
print(f'Mean Squared Error (MSE) on the training set: {mse_train:.4f}')

#predictions test set
y_test_pred = lr_model.predict(x_test_selected_with_const)

#MSE on the test set
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'Mean Squared Error (MSE) on the test set: {mse_test:.4f}')


# In[483]:


#extract the coefficients from the model
coefficients = lr_model.params
print("Regression Equation:")
equation = f"Price = {coefficients['const']:.4f}"

#Loop features and coefficients for equation
for feature, coef in zip(current_features[1:], coefficients[1:]):  # Skip 'const'
    equation += f" + ({coef:.4f}) * {feature}"

print(equation)
