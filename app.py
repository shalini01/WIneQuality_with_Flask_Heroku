    
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model_wine.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Wine_quality.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction=model.predict_proba(final_features)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if  (0 <= request.form.values()[0] <= 1) and (0 <= request.form.values()[1] <= 1) and (0 <= request.form.values()[2] <= 14) and (0 <= request.form.values()[3] <= 100):
        
        if output>str(0.9):
            return render_template('Wine_quality.html',prediction_text='Wine quality is Excellent.')
        elif ( output>str(0.6) and output<=str(0.9)):
            return render_template('Wine_quality.html',prediction_text='Wine quality is Mediocre.')
        else:
            return render_template('Wine_quality.html',prediction_text='Wine quality is Inferior.')

    else:
        return render_template('Wine_quality.html',prediction_text='Incorrect Data Value')


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)  #debug = True will help you refresh browser directly after any change
    
