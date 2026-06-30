from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class RandomForestModel:
    def __init__(self,n_estimators=100,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features="sqrt",class_weight="balanced",random_state=42,n_jobs=-1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class LogisticRegressionModel:
    def __init__(self, penalty='l2', C=1.0, class_weight='balanced', random_state=42, n_jobs=-1):
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

