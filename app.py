from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page'

@app.route('/train')
def train():
    if request.method == 'POST':
        content = request.get_json()
        return content['data']
    else:
        return 'nooo'


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
