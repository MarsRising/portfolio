# Sentiment Analysis with LSTM Neural Network
For this project, I will use the Yelp dataset containing restaurant reviews. My research question is, “Can we create a neural network that will predict sentiment accurately?”.  This analysis aims to have a neural network that can accurately predict sentiment from restaurant reviews so we can understand how well we have been doing with our customers. The neural network I will be using is the long short-term memory (LSTM) neural network. The LSTM network is more efficient due to its ability to remember long-term dependencies, that a recurrent neural network (RNN) struggles with due to vanishing and exploding gradients. It is crucial to understand the context of the customer’s reviews, and with the advanced LSTM capabilities, this would be the best neural network to use.

![image](https://github.com/user-attachments/assets/4dcf7db8-8ead-49f0-9de4-ec556cf76a69)

While our model did stop after a few epochs due to validation loss no longer improving, we can see accuracy increasing from 47.60% to 67.39%, loss decreased from 69.47% to 62.13%, validation accuracy increased from 51.33% to 78.67%, and of course, our validation loss started at 69.89% and ended at 52.68%. 

![image](https://github.com/user-attachments/assets/c640546d-5d9a-482f-b455-14aee51c32c0)


My Bidirectional LSTM neural network is functional. The choices I made from early stopping and saving at each check point where the validation loss improved, improved my model and ensured the most accuracy without overfitting. Choosing to use a Bidirectional LSTM was right in line with performing an appropriate and thorough neural network. The model has room for growth, and I recommend gathering more Yelp reviews to train the neural network or training it on other reviews, so the model becomes more complex and accurate as it grows. 
