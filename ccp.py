
from temp import predict_churn
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv("C:\\Users\\Rishi\\Downloads\\Telco-Customer-Churn.csv")
df.sample(6)
df.drop("customerID",axis="columns",inplace=True)
df[pd.to_numeric(df.TotalCharges,errors="coerce").isnull()]
df.shape
df1=df[df.TotalCharges!=" "]
df1.shape
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
df1.replace("No internet service","No",inplace=True)
df1.replace("No phone service","No",inplace=True)
Y_N_columns=["Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling","Churn"]
for col in Y_N_columns: df1[col].replace({"Yes":1,"No":0},inplace=True)
df1.sample(6)
df1["gender"].replace({"Female":1,"Male":0},inplace=True)
df2=pd.get_dummies(data=df1, columns=["InternetService","Contract","PaymentMethod"])
df2.sample(6)
df2.dtypes
scale_columns=["tenure","MonthlyCharges","TotalCharges"]
import sklearn
from sklearn.preprocessing import MinMaxScaler
df2[scale_columns] = MinMaxScaler().fit_transform(df2[scale_columns])
pd.set_option("display.max_columns", None)
df2.sample(6)
X=df2.drop("Churn",axis="columns")
Y=df2["Churn"]

from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)
#X_train.shape
#X_test.shape

import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([keras.layers.Dense(20, input_shape=(26,), activation="relu"),keras.layers.Dense(15, activation="relu"),keras.layers.Dense(1, activation="sigmoid")])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#model.fit(X_train,Y_train,epochs=100)
#model.evaluate(X_test,Y_test)

from imblearn.combine import SMOTEENN
sm = SMOTEENN()
X_resampled, Y_resampled = sm.fit_resample(X,Y)
Xr_train,Xr_test,Yr_train,Yr_test=train_test_split(X_resampled, Y_resampled,test_size=0.2)
model.fit(Xr_train,Yr_train,epochs=100)
model.evaluate(Xr_test,Yr_test)
Xr_test = Xr_test.reset_index(drop = True)
y_pred = model.predict(Xr_test)
Data = list(Xr_test.index)
predict_churn(y_pred,Data)


