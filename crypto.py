# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas_datareader as pdr
import pandas as pd

# +
import requests
import pandas as pd

# Historical prices for BTC/USD
url = "https://api.tiingo.com/tiingo/crypto/prices"
params = {
    "tickers": "btcusd",
    "startDate": "2019-12-02",
    "endDate": "2024-11-30",
    "resampleFreq": "1Day",  # Correct format

}
headers = {"Authorization": "Token c4eebe93b747af6ac85b1d668ddb31af68e907d9"}

response = requests.get(url, headers=headers, params=params)

# Check response and convert to DataFrame
if response.status_code == 200:
    data = response.json()
    if data:
        df = pd.DataFrame(data[0]['priceData'])
        print(df.head())
    else:
        print("No data found.")
else:
    print(f"Error: {response.status_code}, {response.text}")
# -

df.to_csv("crypto_data.csv", index=False)
print("Data saved as crypto_data.csv")

df = pd.read_csv("crypto_data.csv")
df.head()

X = df[['open', 'high', 'low', 'volume', 'volumeNotional', 'tradesDone']].values
y = df['close'].values.reshape(-1, 1) 


from sklearn.preprocessing import MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)


import numpy as np
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


from keras.models import Sequential
from keras.layers import LSTM, Dense
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')


# Train the model
model.fit(X_train_reshaped, y_train, epochs=100, batch_size=1, verbose=1)


# +

train_predict = model.predict(X_train_reshaped)
test_predict = model.predict(X_test_reshaped)


# +
train_predict = scaler_y.inverse_transform(train_predict)
y_train_actual = scaler_y.inverse_transform(y_train)

test_predict = scaler_y.inverse_transform(test_predict)
y_test_actual = scaler_y.inverse_transform(y_test)

# -

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


# +

mse = mean_squared_error(y_test_actual, test_predict)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, test_predict)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')


# +
import numpy as np

# Example input data (replace with your actual input)
input_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])  # Adjust shape if needed

# Convert input data to float32
input_data = input_data.astype('float32')

# -

# After training your model, save it correctly as a .h5 file
model.save('crypto_model.h5')  # or model.save('crypto_model.keras')


# +
import gradio as gr
import tensorflow as tf
import numpy as np

# Load your trained model (replace with the path to your model)
model = tf.keras.models.load_model('crypto_model.h5')

# Define the prediction function
def predict(input_data):
    input_data = np.array(input_data).reshape(1, -1)  # Adjust the shape as per your model input
    prediction = model.predict(input_data)
    return prediction[0][0]

# Create a simple Gradio interface
interface = gr.Interface(fn=predict, inputs="text", outputs="text")

# Launch the interface
interface.launch()

# -


