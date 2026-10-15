"""Classify and regress on a table with ChakraTab, directly and through scikit-learn."""

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import cross_val_score, train_test_split

from yhatlabs import Client
from yhatlabs.integrations import ChakraTabClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

with Client() as yh:
    res = yh.predict(X_train.assign(target=y_train), "target", X_test)
    print("accuracy", (res.predictions == y_test).mean())
    print(res.probabilities.head())

    Xr, yr = load_diabetes(return_X_y=True, as_frame=True)
    reg = yh.predict(Xr.iloc[:350].assign(progression=yr.iloc[:350]), "progression", Xr.iloc[350:], task="regress")
    print("MAE", (reg.predictions - yr.iloc[350:]).abs().mean())

print("5-fold accuracy", cross_val_score(ChakraTabClassifier(), X, y, cv=5).mean())
