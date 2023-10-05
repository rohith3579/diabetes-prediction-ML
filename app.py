import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('Diabetes.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    print(final_features)
    prediction=model.predict_proba(final_features) ## Predicting the output
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output < str(0.5):
        return render_template('index.html', prediction_text='THE PATIENT IS NOT LIKELY TO HAVE A DIABETES')
    else:
         return render_template('index.html', prediction_text='THE PATIENT IS LIKELY TO HAVE A DIABETES')
        
@app.route('/predict_api',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)