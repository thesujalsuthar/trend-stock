import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model

st.title('Stock Trend Predictor')
user_input=st.text_input('Enter the Ticker','RELIANCE.NS')
data=yf.Ticker(user_input)
dt=st.text_input('Enter Date From (Before 2022 only)','2012-01-30')
df=data.history(start=dt)
columns_to_drop = ['Date', 'Dividends', 'Stock Splits']
columns_exist = all(col in df.columns for col in columns_to_drop)
if columns_exist:
    df.drop(columns_to_drop, axis=1, inplace=True)
else:
    print("One or more columns not found in the DataFrame.")

#Describing Data
st.subheader(f'Data From {dt} to latest')
st.write(df.describe())

#Visualizations
st.subheader("Closing Price vs Time")
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time with 100 days MA")
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time with 200 days MA")
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma200,'g')
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time with 100 & 200 days MA")
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


#Splitting into training and testing 
train_data=pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
test_data=pd.DataFrame(df['Close'][int(len(df)*0.75):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

train_data_array=scaler.fit_transform(train_data)

# #splitting data into X_train and y_train
# x_train=[]
# y_train=[]

# for i in range(100, train_data_array.shape[0]):
#     x_train.append(train_data_array[i-100:i])
#     y_train.append(train_data_array[i, 0])

# x_train, y_train= np.array(x_train), np.array(y_train)


#Load the model
model=load_model('keras_model.h5')



#Testing data
past_100_days=train_data.tail(100)
final_df=past_100_days.append(test_data, ignore_index=True)

input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test=np.array(x_test), np.array(y_test)

y_predicted=model.predict(x_test)

scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor



#Final visualization
st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(10,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
