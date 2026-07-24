from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


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
    def __init__(self,C=1.0,class_weight="balanced",random_state=42,max_iter=2000):
        self.model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            max_iter=max_iter)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class SVMModel:
    def __init__(self,C=1.0,kernel="rbf",gamma="scale",class_weight="balanced",random_state=42):
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class MLPModel:
    def __init__(self,hidden_layer_sizes=(100,),activation="relu",solver="adam",alpha=0.0001,learning_rate_init=0.001,max_iter=1000,random_state=42, early_stopping=False):

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=early_stopping)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)