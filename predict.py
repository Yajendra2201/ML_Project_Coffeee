import pickle
import warnings
warnings.filterwarnings('ignore')

def lung_cancer_prediction_Logistic(features):
    
    pickled_model = pickle.load(open('model/lung_cancer_detection_Logistic.pkl', 'rb'))
    can_predict = str(round(list(pickled_model.predict([features]))[0]))
    

    return str("Lung Cancer Level using Logistic Regression is: Level " + can_predict)

def lung_cancer_prediction_Decision(features):
    
    pickled_model1 = pickle.load(open('model/lung_cancer_detection_Decision.pkl', 'rb'))
    can_predict1 = str(round(list(pickled_model1.predict([features]))[0]))
    


    return str("Lung Cancer Level using Decision Tree is: Level " + can_predict1)
               

def lung_cancer_prediction_Random(features):
    
    pickled_model2 = pickle.load(open('model/lung_cancer_detection_Random.pkl', 'rb'))
    can_predict2 = str(round(list(pickled_model2.predict([features]))[0]))


    return str("Lung Cancer Level using Random Forest is: Level " + can_predict2)