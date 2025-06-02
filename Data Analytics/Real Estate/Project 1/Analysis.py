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
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import RFE
print("libs downloaded!")

#define path
file_path = 'C:/Users/tyler/OneDrive - SNHU/WGU/Statistical Data Mining/D600 Task 1 Dataset 1 Housing Information.csv'
#import excel to dataframe
df=pd.read_csv(file_path)
print(df.head)


#drop irrelevatn columns
df.drop(columns=["ID"], inplace=True)
df.info()


df.describe(include='all')

#use jsut the numeric variables to find significinat continuous variables
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix=numeric_df.corr()
print(correlation_matrix)


#get rid of insignificant correlations
df = df.drop(columns=['BackyardSpace', 'CrimeRate', 'SchoolRating', 'AgeOfHome', 'DistanceToCityCenter', 'EmploymentRate', 'PropertyTaxRate', 'LocalAmenities', 'TransportAccess', 'Floors', 'Windows'])
df.info()


#xheck pearson correlation jsut for the significant ones
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix=numeric_df.corr()
print(correlation_matrix)

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

#ANOVA due to multiple categorical values in house colors
house_colors = [df[df['HouseColor'] == color]['Price'] for color in df['HouseColor'].unique()]
f_stat, p_value = f_oneway(*house_colors)
print(f"ANOVA p-value for 'HouseColor' and 'Price': {p_value}")


#Out of the 4 categorical values the only 2 significant variables are Fireplace and IsLuxury DROP the other two
df.drop(columns=['HouseColor', 'Garage'], inplace=True)
df.info()

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


#ALL VARIABLES ARE GOOD! No Multicollinearity
#check isluxury is set up accurately
print(df.dtypes)
#was not. Update it to categorical
df['IsLuxury'] = df['IsLuxury'].astype('category')
#DESCRIPTIVE STATISTICS
df.describe(include='all')


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


#Fireplace CATEGORICAL
sns.countplot(x='Fireplace', data=df)
plt.title('Fireplace Distribution')
plt.xlabel('Fireplace')
plt.ylabel('Count')
plt.show()


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


#IsLuxury CATEGORICAL
sns.countplot(x='IsLuxury', data=df)
plt.title('IsLuxury Distribution')
plt.xlabel('IsLuxury')
plt.ylabel('Count')
plt.show()


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


#The boxes in the Fireplac, as well as Medians, are very close.
#I will remove Fireplace as this may not be necessarily as strong as first expected, and I would like to not complicate my model
df.drop(columns=["Fireplace"], inplace=True)
df.info()


#Standardization
scaler = StandardScaler()
numeric_columns = ['SquareFootage', 'NumBathrooms', 'NumBedrooms', 'RenovationQuality', 'PreviousSalePrice']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print('Complete!')


print(df.isnull().sum())


x = df.drop('Price', axis=1)
y = df['Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print('Complete!')

#Linear Regression
model = LinearRegression()
#I am choosing to use Recursive Feature Elimination as I have already narrowed down manually, and would like to use the top 5
#choosing top 5 will help with overfitting
rfe = RFE(model, n_features_to_select=5)
x_rfe = rfe.fit_transform(x_train, y_train)
#show me the top 5 variables that impact house Prices
selected_features = x_train.columns[rfe.support_]
print("Selected Features:", selected_features)
#run linear regression
model.fit(x_rfe, y_train)
#update test data to reflect Recursive Feature Elimination
x_test_rfe = rfe.transform(x_test)
#predict
y_pred = model.predict(x_test_rfe)


#Renovation Quality was eliminated as it wasn't in our top 5!
#OLS to see statistics
#add constraint
x_train_with_const = sm.add_constant(x_train[selected_features])
#fit model
ols_model = sm.OLS(y_train, x_train_with_const)
results = ols_model.fit()
#statistics
print(results.summary())

#add constant to for intercept
x_test_rfe = x_test[selected_features]
x_test_with_const = sm.add_constant(x_test_rfe)

#Predict
y_pred = results.predict(x_test_with_const)



#print top 10 predictions
print(y_pred[:10])
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())


coef = results.params
p_values = results.pvalues
print("Coefficients:\n", coef)
print("\nP-values:\n", p_values.apply(lambda x: f'{x:.100f}'))


#easily see values
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")


coefficients = model.coef_
intercept = model.intercept_

print("Model Coefficients:", coefficients)
print("Intercept:", intercept)



mse_train = mean_squared_error(y_train, y_train)
print("MSE (Training Set):", mse_train)
mse_test = mean_squared_error(y_test, y_pred)
print("MSE (Test Set):", mse_test)
