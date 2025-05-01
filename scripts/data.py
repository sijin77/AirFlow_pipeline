from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    os.makedirs("dataset", exist_ok=True)
    df.to_csv("dataset/iris.csv", index=False)


def prepare_data():
    df = pd.read_csv("dataset/iris.csv")
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv("dataset/iris_train.csv", index=False)
    test.to_csv("dataset/iris_test.csv", index=False)
