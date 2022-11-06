from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/train/', methods=['GET', 'POST'])
def add_message():
    content = request.json
    return content

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
