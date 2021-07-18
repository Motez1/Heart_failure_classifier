#Importing the libraries:
import numpy as np
from numpy import random
from NeuralNet import *
import matplotlib.pyplot as plt
import pandas as pd

# Importing the data :
path = "heart_data.csv"
dataframe = pd.read_csv(path)
print(dataframe.head())
print(dataframe.info())

# Normalizing the data :
df_z_scaled = dataframe.copy()
for column in df_z_scaled.columns:
    df_z_scaled[column] = (df_z_scaled[column] -
                           df_z_scaled[column].mean()) / df_z_scaled[column].std()

heart_data = pd.concat([df_z_scaled[["age","ejection_fraction","serum_creatinine","serum_sodium","time"]],dataframe[["DEATH_EVENT"]]],axis=1) 

# Shuffling and splitting the data :
np.random.seed(42) # to make results replication possible
heart_data = heart_data.sample(frac=1).reset_index(drop=True)

heart_data_train = heart_data[:239]
heart_data_cross = heart_data[239:269]
heart_data_test  = heart_data[269:]

X_train = heart_data_train[["age","ejection_fraction","serum_creatinine","serum_sodium","time"]].values.reshape(239,5,1)
Y_train = heart_data_train[["DEATH_EVENT"]].values.reshape(239,1,1)

X_cross = heart_data_cross[["age","ejection_fraction","serum_creatinine","serum_sodium","time"]].values.reshape(30,5,1)
Y_cross = heart_data_cross[["DEATH_EVENT"]].values

X_test = heart_data_test[["age","ejection_fraction","serum_creatinine","serum_sodium","time"]].values.reshape(30,5,1)
Y_test = heart_data_test[["DEATH_EVENT"]].values

#Building the model:
np.random.seed(579) #choosed using the training and cross validation to get the best accuracy for the validation set
layers = [
            Dense(5,5),
            Tanh(),
            Dense(5,5),
            Tanh(),
            Dense(5,5),
            Tanh(),
            Dense(5,1),
            Sigmoid()
        ]
classifier = NeuralNetwork(layers)
classifier.train(X_train,Y_train,epochs=7,learning_rate=0.01 )
#Training accuracy calculation:
predictions = []
for k in range(X_train.shape[0]):
    predictions.append(classifier.predict(X_train[k]))
arr_prediction = np.array(predictions).reshape(239,1)
arr_prediction[arr_prediction <0.5] = 0
arr_prediction[arr_prediction >= 0.5] = 1
train_accuracy = 1 - np.sum(np.absolute(arr_prediction - Y_train.reshape(239,1)))/239
print("================================================")
print(f"train_accuracy : {train_accuracy}")

#Cross validation accuracy calculation:
predictions = []
for k in range(X_cross.shape[0]):
    predictions.append(classifier.predict(X_cross[k]))
arr_prediction = np.array(predictions).reshape(30,1)
arr_prediction[arr_prediction <0.5] = 0
arr_prediction[arr_prediction >= 0.5] = 1
cross_accuracy = 1 - np.sum(np.absolute(arr_prediction - Y_cross))/30
print(f"cross_accuracy : {cross_accuracy}")

#Test set accuracy calculation:
predictions = []
for d in range(X_test.shape[0]):
    predictions.append(classifier.predict(X_test[d]))
arr_prediction = np.array(predictions).reshape(30,1)
arr_prediction[arr_prediction <0.5] = 0
arr_prediction[arr_prediction >= 0.5] = 1
accuracy = 1 - np.sum(np.absolute(arr_prediction - Y_test))/30
print(f"accuracy: {accuracy}")
print("================================================")