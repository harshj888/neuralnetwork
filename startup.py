# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:13:14 2020

@author: Harsh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda

startup = pd.read_csv("D:\STUDY\Excelr Assignment\Assignment 16 - Neural Networks\Startups.csv")

##Creating dummy variables for the state column

startup_dummy = pd.get_dummies(startup["State"])

Startup = pd.concat([startup,startup_dummy],axis=1)

Startup.drop(["State"],axis=1, inplace =True)

Data = Startup.describe()

##The scales of the data are differnt so we normalise
def norm_func(i):
    x = (i - i.min())/(i.max()-i.min())
    return (x)

Start_up = norm_func(Startup)

#Using this Data set and we build the model. 
def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

predictors = Start_up.iloc[:,[0,1,2,4,5,6]]
target = Start_up.iloc[:,3]

#Partitioning the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25)

first_model = prep_model([6,50,1])
first_model.fit(np.array(x_train),np.array(y_train),epochs=900)
pred_train = first_model.predict(np.array(x_train))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-y_train)**2))
np.corrcoef(pred_train,y_train) 


#Visualising 
plt.plot(pred_train,y_train,"bo")

##Predicting on test data
pred_test = first_model.predict(np.array(x_test))
pred_test = pd.Series([i[0] for i in pred_test])
rmse_test = np.sqrt(np.mean((pred_test-y_test)**2))
np.corrcoef(pred_test,y_test)

##Visualizing
plt.plot(pred_test,y_test,"bo")
