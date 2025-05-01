import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def test_model():
    test_data = pd.read_csv('dataset/iris_test.csv')
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    model = joblib.load('model.pkl')
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    
    metrics = {'accuracy': accuracy, 'report': report}
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f)