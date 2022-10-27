
from flask import Flask
import predict
app = Flask(__name__)

@app.route('/')
def hello_world():
    
    test_features =[33,2,2,4,5,6,2,3,1,7,8,9,5,7,8,2,2,5,3,2,4,9,8]
    # test_features2=[3.531289,4.312772,9.004999,21.952221,0.315674,0.325269,0.282931,0.195474,0.438748,0.629365,1.58019,5.503635,0.164347,0.101833,3.869108,4.963869,9.80051,26.036513,0.377624,0.487647,0.516817,0.318591,0.548999,0.295804]
    
    ln1 = predict.lung_cancer_prediction_Logistic(test_features)
    ln2 = predict.lung_cancer_prediction_Decision(test_features)
    ln3 = predict.lung_cancer_prediction_Random(test_features) 
    a = "<br>"
    return str(ln1+a+ln2+a+ln3)


if __name__=="__main__":
    app.run(debug=True,port=8000)