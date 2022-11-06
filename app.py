from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/train')
def add_message():
    if request.method == 'POST':
        content = request.json
        aux = content
    else:
        aux = "test..."
    return aux




if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
