My research question for this scenario is, “Which principal components explain the most variation in house prices?”. 
This analysis aims to understand which factors influence house prices the most. Knowing the principal components that determine house prices, we can create a predictive model to help estimate what house prices should be. Setting these house prices will be important in making profits and investment decisions


![image](https://github.com/user-attachments/assets/441371ff-ef16-44ba-a62f-5ac762b289b6)

From the screenshot above, I can determine the principal components' matrix. We can see the percentage that each variable impacts each principal component. The first component we can interpret is that ‘PreviousSalePrice’ and ‘RenovationQuality’ are the strongest contributors, with a weight of .584 and .475, respectively. The second component shows that ‘NumBedrooms’, is the strongest contributor, with a weight of .847. Our third component shows that ‘NumBathrooms’ is the strongest contributor at a weight of .755. The fourth component we can interpret is that the ‘RenovationQuality’ is the strongest contributor with a weight of .878. Lastly, in the final component, we can see that ‘PreviousSalePrice’ is the strongest contributor, at a weight of .751. 

![image](https://github.com/user-attachments/assets/d066a858-c7b8-4787-ad83-54e831273a52)


After creating my scree plot as pictured above, I decided to go with the elbow rule. I looked at the Kaiser rule, but it seemed very restrictive as it told me only one component, and I would later implement backward stepwise elimination for optimization. To be accurate with the elbow rule, we must look to a sharp change in directionality. We can easily see from the above visual that after 2 components there is a sharp change in directionality. Our elbow is at the 2 Principal Components. The elbow rule demonstrates that 2 components are what I will retain and use in my linear regression model, as after 2 components, the others contribute less.


We found that the first two principal components explained the variation the best. This included the ‘PreviousSalePrice’, ‘RenovationQuality’, and ‘NumBedrooms’ variables. The p-values for these components were significant as they were less than 0.05. However, as stated above there is room for growth with an R^2 being 0.663. We have obtained this analysis's question and goal, but I recommend working on this linear model for more accurate and reliable price prediction. Having a more precise and reliable price prediction model, we, as an organization, could have more accuracy when setting out prices. This can lead to better sales and growth for our organization. 	
