from flask import Flask, request, json, jsonify
from helper import *
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
import tensorflow as tf
app = Flask(__name__)

@app.route('/', methods = ['POST'])





def index():
    payload = json.loads(request.get_data().decode('utf-8'))

    
    p = prediction(payload, 'saved_model_last/my_model')
    



    price_today = payload["Adj Close"]


    out = {'The close price for the entered day was': float(price_today), 'The predicted close price for the next day is': float(p)}
    
    
    return json.dumps(out)

if __name__ == "__main__":
    app.run(debug = True)


