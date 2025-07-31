# Spam-Email-detection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

emails = [
   " Get rich by filling the lottery ticket form! ",
   " At 7 we have a scrum meeting",
   "Congratulations you have won 1000rs as a free gift",
   "Please help me with this excel data.",
   "Click here to claim Iphone!",
   "Congratulations you have won 1000rs as a free gift",
   " Get rich by filling the lottery ticket form! ",
   " At 7 we have a scrum meeting",
   "Please help me with this excel data.",
   "Click here to claim Iphone!",
   "Please help me with this excel data.",
   "Click here to claim Iphone!"
]
lables=[1,0,1,0,1,1,1,0,0,1,0,1]
vectorizer=CountVectorizer()
x=vectorizer.fit_transform(emails)
X_train,X_test,y_train,y_test=train_test_split(x,lables,test_size=0.2)

model=MultinomialNB()
model.fit(X_train,y_train)

y_predict=model.predict(X_test)

accuracy=accuracy_score(y_test,y_predict)
print(accuracy)

def predict_spam(message):
    message = vectorizer.transform([message])
    prediction = model.predict(message)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'

new_email = input('Enter email to predict spam:')
prediction = predict_spam(new_email)
print(prediction)
