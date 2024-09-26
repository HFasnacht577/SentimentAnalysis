from flask import Flask, render_template, request, jsonify
from bot import train_log

app = Flask(__name__)

model,vectorizer = train_log()

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/analyze',methods=['POST'])
def analyze():
    user_input = request.form['text']

    sentiment = model.predict(vectorizer.transform([user_input]))

    if sentiment == 1:
        output = "Positive"
    else:
        output = "Negative"

    return jsonify(output)
