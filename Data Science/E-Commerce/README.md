# E-Commerce Fraud Detection

## Can I create an effective gradient-boosting model to detect fraud in e-commerce transactions?

Null Hypothesis- The gradient boosting model has a recall of less than 70%.

Alternative Hypothesis- The gradient boosting model has a recall greater than 70%.

![image](https://github.com/user-attachments/assets/8af7cd28-6b75-4802-9c05-b514f61bb095)
![image](https://github.com/user-attachments/assets/c2253ab7-1aa3-4cd6-aa6e-23bc10b76705)



Dataset Citation: Jagtap, Shriyash. (2024, April 7). Fraudulent E-Commerce Transactions. Kaggle. https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions
![image](https://github.com/user-attachments/assets/f96c296d-eeea-4605-83df-5dc0d822f8da)


My data analysis implies that the model does pass the alternative hypothesis test with a recall of 72%. I have answered the business question, but there is room for improvement. There is room for growth in our weak precision and low F1 score. The precision of only 26% demonstrates that there will be a lot of false red flags. This model is limited in its accuracy, and I suggest further tuning of this model. I could investigate the variables and see if some multicollinearity is affecting the results and perform some feature selection. Another method I can try is using more computationally heavy hyperparameter tuning for better metrics or even testing a neural network that may increase fraud detection. Either way, I suggest further improving fraud detection.
