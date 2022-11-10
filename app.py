from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import numpy as np
import boto3
import s3fs
import os
import io
import datetime
import time
import json


app = Flask(__name__)



@app.route('/')
def index():
    return 'Index Page'


@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['file']
    f.save(secure_filename(f.filename))
    return 'Load'



@app.route('/load_data', methods=['GET', 'POST'])
def load_data():
    if request.method == 'POST':
        #conecta con S3
        content = request.get_json()

        source = content['source']
        s3 = boto3.client('s3', aws_access_key_id=source['aws_access_key_id'], aws_secret_access_key= source['aws_secret_access_key'])

        bucket_name = source['bucket_name']
        orderData_Features = pd.DataFrame()

        for s3_object in source['object']:

            obj = s3.get_object(Bucket=bucket_name, Key=s3_object)
            orderData_Features_aux = pd.read_json(io.BytesIO(obj['Body'].read()))
            orderData_Features_aux['fecha'] =  pd.to_datetime(orderData_Features_aux['fecha'],unit='ms')
            orderData_Features = orderData_Features.append(orderData_Features_aux)
            
        file_name = s3_object[0:s3_object.find('.')] + '_' + str(round(time.time())) + '.pkl'
        orderData_Features.to_pickle(file_name)
        
        return {'file_name': file_name, 'registros': len(orderData_Features)}

    else:
        files_str = []
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        for f in files:
            if f[-4:] == '.pkl':
                files_str.append(f)
                
        return  {'filenames': files_str}



@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        timestamp_start = datetime.datetime.now()
        content = request.get_json()

        # define hyperparameters
        hyperparameters = content['hyperparameters']

        n_estimators = hyperparameters['n_estimators'] if ('n_estimators' in hyperparameters) else 80
        max_depth = hyperparameters['max_depth'] if ('max_depth' in hyperparameters) else 4
        learning_rate = hyperparameters['learning_rate'] if ('learning_rate' in hyperparameters) else 0.1
        min_child_weight = hyperparameters['min_child_weight'] if ('min_child_weight' in hyperparameters) else 10
        booster = hyperparameters['booster'] if ('booster' in hyperparameters) else 'gbtree'
        print('ok carga yperparametros')

        #conecta con S3
        source = content['source']
        #s3 = boto3.client('s3', aws_access_key_id=source['aws_access_key_id'], aws_secret_access_key= source['aws_secret_access_key'])

        #bucket_name = source['bucket_name']
        #s3_object = source['object']

        #obj = s3.get_object(Bucket=bucket_name, Key=s3_object)
        #orderData_Features = pd.read_json(io.BytesIO(obj['Body'].read()))
        orderData_Features = pd.read_pickle(source['file_name']) 
        orderData_Features['fecha'] =  pd.to_datetime(orderData_Features['fecha'],unit='ms')
        print('ok carga data desde s3')

        timestamp_end_extraction = datetime.datetime.now()

        # create model instance

        TARGET = 'qty'
        FEATURES = ['outlier', 'cluster', 'date_day_of_week', 'date_day_of_month', 'date_day_of_year', 'date_week', 'date_month', 'qty_1DA', 'qty_2DA', 'qty_3DA', 'qty_4DA', 'qty_5DA', 'qty_6DA', 'qty_7DA', 'qty_8DA', 'qty_9DA', 'qty_10DA', 'qty_11DA', 'qty_12DA', 'qty_13DA', 'qty_14DA', 'qty_21DA', 'qty_28DA']

        X_train = orderData_Features[orderData_Features.fecha < '2022-04-01'][FEATURES]
        y_train = orderData_Features[orderData_Features.fecha < '2022-04-01'][TARGET]
        
        X_test = orderData_Features[orderData_Features.fecha > '2022-04-03'][FEATURES]
        y_test = orderData_Features[orderData_Features.fecha > '2022-04-03'][TARGET]
        print('ok transfformacionn de la data')

        
        # fit model
        bst = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, min_child_weight=min_child_weight, booster=booster)
        bst.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], verbose = False)
        print('ok listailor')
        
        timestamp_end = datetime.datetime.now()

        model_name = ''
        if content['options']['save']:
            model_name= source['file_name'][:-4] + "_model.json"
            bst.save_model(model_name)


        return {'hyperparameters': {'n_estimators': n_estimators,'max_depth':max_depth,'learning_rate':learning_rate, 'min_child_weight': min_child_weight,'booster':booster },
                'results': {'train_data_result': bst.evals_result()['validation_0']['rmse'][-1], 'test_data_result':  bst.evals_result()['validation_1']['rmse'][-1]}, 
                'execution_time': {'time_model_execution': (timestamp_end-timestamp_end_extraction).total_seconds(),'time_data_extraction': (timestamp_end_extraction-timestamp_start).total_seconds()},
                'model': {'saved_model_file': model_name}
                }
    else:
        return 'nooo'


@app.route('/predict', methods=['GET', 'POST'])
def make_prediction():
    timestamp_start = datetime.datetime.now()
    content = request.get_json()

    bst = xgb.XGBRegressor()
    bst.load_model(content['saved_model_file'])

    X_predict = np.array(content['X'])
    y_predict = bst.predict(X_predict)

    return json.dumps(y_predict.tolist())  

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
