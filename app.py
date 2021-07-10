import pickle
import numpy as np
from flask import Flask,redirect,url_for,render_template,request

app=Flask(__name__)
lr=pickle.load(open("static/DATA PROJECT1.pkl",'rb'))

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method=="POST":
        
        html_input=[int(i) for i in request.form.values() ]
        final_features=[np.array(html_input)]
        prediction=lr.predict(final_features)
        msg=round(prediction[0],2)
    return render_template('result.html',msg=msg)       
    
if __name__=='__main__':
    app.run()
