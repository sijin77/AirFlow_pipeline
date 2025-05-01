from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib


def train_model():
    train_data = pd.read_csv("dataset/iris.csv")
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")
