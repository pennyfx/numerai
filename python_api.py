import os
import subprocess
import json
import re
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from flask import Flask, request, jsonify


def add(params):
    return params['a'] + params['b']

def predict(params):
    model_filename = os.path.join(os.getcwd(), 'model.dat')
    # load model from file
    loaded_model = pickle.load(open(model_filename, "rb"))
    X = params 
    x_test = pd.DataFrame(X)
    # only used for final model with trained with fewer dimensions
    feature_list = ['feature19', 'feature6', 'feature7']
    x_test = x_test[feature_list]
    y_pred = loaded_model.predict(x_test)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    return int(y_pred[0])

functions_list = [add, predict]

app = Flask(__name__)

@app.route('/<func_name>', methods=['POST'])
def api_root(func_name):
    for function in functions_list:
        if function.__name__ == func_name:
            try:
                json_req_data = request.get_json()
                if json_req_data:
                    res = function(json_req_data)
                else:
                    return jsonify({"error": "error in receiving the json input"})
            except Exception as e:
                return jsonify({"error": "error while running the function"})
            return jsonify({"result": res})
    output_string = 'function: %s not found' % func_name
    return jsonify({"error": output_string})

if __name__ == '__main__':
    app.run(host='0.0.0.0')