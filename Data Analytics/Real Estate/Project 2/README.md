In this analysis I aimed to answer the questions, "What are the factors that most influence if a house is considered luxury?” 

I optimized my logistic regression model using backward stepwise elimination. The reason I chose backward stepwise elimination is due to its process of including all variables and iteratively removing each variable until the model doesn’t improve anymore. This model selects the most important variables and takes all the variables into account when starting the process.  This process eliminated all variables until I was left with my dependent variable ‘IsLuxury’ and my independent variables ‘Price’ and ‘PreviousSalePrice’.

![image](https://github.com/user-attachments/assets/6d687848-1eb3-4ff2-b55a-981e12459f69)


![image](https://github.com/user-attachments/assets/d006d5f3-1800-4e5b-97bf-3350100e6d14)


![image](https://github.com/user-attachments/assets/cef317b8-5939-4efc-96bf-28fe3a99b89c)


There are assumptions made by the logistic regression model and I will speak on 4 of them. This is first that our dependent variable is binary. Observations are independent. It also assumes there is no multicollinearity between our independent variables. The last assumption is that the sample is sufficiently large and we have 7,000 datapoints. 

