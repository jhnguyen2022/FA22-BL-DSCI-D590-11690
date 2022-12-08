from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    float_input =[float(x) for x in request.form.values()]
    vals = [np.array(float_input)]
    
    prediction = model.predict(vals)
    
    return render_template('prediction_test.html', prediction = prediction)

@app.route('/predict_test', methods=['GET','POST'])
def predict_test():
    return render_template('prediction_test.html')

@app.route('/analysis_overall')
def analysis_overall():
    return render_template('analysis_overall.html')

@app.route('/analysis_DataScience')
def analysis_ds():
    return render_template('analysis_ds.html')
     

if __name__ == '__main__':
    app.run(debug=True)