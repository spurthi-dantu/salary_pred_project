import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import numpy as np
import pickle
import pandas as pd



def load_models():
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

app = Flask(__name__)
@app.route('/predict', methods=['GET'])

def predict():
    # # stub input features
    # request_json = request.get_json()
    # x = request_json['input']
    # #print(x)
    # Convert input list to numpy array
    #x = np.array([data_in], dtype=float)
    feature_names=['Rating', 'age', 'Size', 'job_role', 'job_seniority', 'job_state']
    # # Reshape input array to match the expected input shape of the model
    # input_array = input_array.reshape(1, -1)

    x=pd.DataFrame([data_in], columns=feature_names)
    print(x)
    
    #x=data_in
    # # load model
    model = load_models()
    prediction = model.predict(x)
    prediction_list = prediction.tolist()
    response = json.dumps({'response': prediction_list})
    
    return response, 200

if __name__ == '__main__':
    application.run(debug=True)