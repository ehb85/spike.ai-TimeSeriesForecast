from flask import Flask, request, jsonify
import threading
import xgboost as xgb


app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page'


@app.route('/upload', methods=['POST'])
def upload_file():
    from werkzeug.datastructures import FileStorage
    #FileStorage(request.stream).save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    FileStorage(request.stream).save('file.pkl')
    return 'OK', 200


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

        # data to train 
        X_train = content['train_data']['target']
        y_train = content['train_data']['features']

        # data to test 
        X_test = content['test_data']['target']
        y_test = content['test_data']['features']

        # fit model
        bst.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], verbose = 10)

        
        return bst
    else:
        return 'nooo'


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
