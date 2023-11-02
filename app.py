from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import yfinance as yf
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app)

# Load the pre-trained LSTM model and scaler
model = load_model('bitcoin_lstm_model.h5')
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = np.load('scaler_params.npy')

# Load historical data for 'BTC-USD' up to the current date and time
start_date = '2014-01-01'
df = yf.download('BTC-USD', start=start_date)

def update_data():
    while True:
        last_price = df['Close'].iloc[-1]  # Get the last Bitcoin price

        sequence_length = 60  # Make sure it matches the training sequence length
        input_data = df['Close'][-sequence_length:].values
        scaled_input_data = scaler.transform(input_data.reshape(-1, 1))
        X_test = np.reshape(scaled_input_data, (1, sequence_length, 1))

        # Predict Bitcoin prices
        predicted_prices = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
        predicted_prices = predicted_prices.flatten()

        # Emit data to the connected clients
        socketio.emit('update_data', {
            'last_price': last_price,
            'predicted_prices': predicted_prices.tolist()
        })

        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.start_background_task(update_data)
    socketio.run(app, debug=True)
