from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = make_pipeline(
            Imputer(strategy='most_frequent'),
            RandomForestClassifier(max_depth=10, n_estimators=300,
                                   random_state=0),
            StandardScaler(),
            LogisticRegression(C=1.0, fit_intercept=True)
        )

    def __getattr__(self, attrname):
        return getattr(self.clf, attrname)

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    clf = Classifier()
    clf.fit(X, y)
    print(clf.predict(X))
