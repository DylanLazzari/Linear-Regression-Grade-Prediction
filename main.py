import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model


#This is Loading in our data
data = pd.read_csv("student_mat_2173a47420.csv", sep = ';')


#This trims out the data and only takes what we need!
data = data[['G1', 'G2', 'G3', 'goout', 'freetime', 'absences']]
#print(data.head())


#This is the label(what we are trying to predict )
#Everything else is now a feature that will help us arrive at our label value
predict = ['G3']
feature = ['G1', 'G2', 'goout', 'freetime', 'absences']


X = np.array(data[feature])
y = np.array(data[predict])


#Now i need to train the data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)


#------------------------Algorithm--------------------------
linear = linear_model.LinearRegression() 
linear.fit(x_train, y_train) #Fitting the data to the algorithm 
accuracy = linear.score(x_train, y_train) 

print(accuracy)

print('Coefficient: \n', linear.coef_) # These are each of the slope values
print('Intercept: \n', linear.intercept_) # This is the intercept\


#----------------Prediction------------------------
predictions = linear.predict(x_test)  #This is basically getting a list of all the predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
