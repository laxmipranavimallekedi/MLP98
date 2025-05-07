from flask import Flask, render_template, request, jsonify
from utils import predict_sleep_health, generate_tip

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        sleep_duration = float(request.form['sleep_duration'])
        activity = float(request.form['activity'])
        stress = float(request.form['stress'])
        steps = float(request.form['steps'])

        result = predict_sleep_health(age, sleep_duration, activity, stress, steps)
        result['tip'] = generate_tip(result['sleep_quality'], result['heart_rate'])

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
