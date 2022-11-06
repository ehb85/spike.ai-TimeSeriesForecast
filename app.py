from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page'


@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['file']
    f.save(secure_filename(f.filename))
    return 'Load'

@app.route('/load_data', methods=['POST'])
def load_data():
    # preprocesa tdos
    orderData_Features = pd.read_pickle("orderData_Features_wLags.pkl") 
    TARGET = 'qty'
    FEATURES = ['outlier', 'cluster', 'date_day_of_week', 'date_day_of_month', 'date_day_of_year', 'date_week', 'date_month', 'qty_1DA', 'qty_2DA', 'qty_3DA', 'qty_4DA', 'qty_5DA', 'qty_6DA', 'qty_7DA', 'qty_8DA', 'qty_9DA', 'qty_10DA', 'qty_11DA', 'qty_12DA', 'qty_13DA', 'qty_14DA', 'qty_21DA', 'qty_28DA']

    X_train = orderData_Features[orderData_Features.fecha < '2022-04-01'][FEATURES]
    y_train = orderData_Features[orderData_Features.fecha < '2022-04-01'][TARGET]
    
    X_test = orderData_Features[orderData_Features.fecha > '2022-04-03'][FEATURES]
    y_test = orderData_Features[orderData_Features.fecha > '2022-04-03'][TARGET]

    return '200, cargado'


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        content = request.get_json()

        # define hyperparameters
        hyperparameters = content['hyperparameters']

        n_estimators = hyperparameters['n_estimators'] if ('n_estimators' in hyperparameters) else 80
        max_depth = hyperparameters['max_depth'] if ('max_depth' in hyperparameters) else 4
        learning_rate = hyperparameters['learning_rate'] if ('learning_rate' in hyperparameters) else 0.1
        min_child_weight = hyperparameters['min_child_weight'] if ('min_child_weight' in hyperparameters) else 10
        booster = hyperparameters['booster'] if ('booster' in hyperparameters) else 'gbtree'

        # create model instance
        bst = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, min_child_weight=min_child_weight, booster=booster)

        orderData_Features = pd.read_pickle("orderData_Features_wLags.pkl") 
        TARGET = 'qty'
        FEATURES = ['outlier', 'cluster', 'date_day_of_week', 'date_day_of_month', 'date_day_of_year', 'date_week', 'date_month', 'qty_1DA', 'qty_2DA', 'qty_3DA', 'qty_4DA', 'qty_5DA', 'qty_6DA', 'qty_7DA', 'qty_8DA', 'qty_9DA', 'qty_10DA', 'qty_11DA', 'qty_12DA', 'qty_13DA', 'qty_14DA', 'qty_21DA', 'qty_28DA']

        X_train = orderData_Features[orderData_Features.fecha < '2022-04-01'][FEATURES]
        y_train = orderData_Features[orderData_Features.fecha < '2022-04-01'][TARGET]
        
        X_test = orderData_Features[orderData_Features.fecha > '2022-04-03'][FEATURES]
        y_test = orderData_Features[orderData_Features.fecha > '2022-04-03'][TARGET]


        


        # fit model
        bst.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], verbose = 10)

        
        return bst
    else:
        return 'nooo'


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
