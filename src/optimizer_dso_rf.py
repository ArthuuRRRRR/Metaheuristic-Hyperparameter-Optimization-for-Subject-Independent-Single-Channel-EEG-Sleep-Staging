import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from dso import DSO


class DSO_Optimizer:
    def __init__(self,population_size=10,max_eval=100,random_state=42):
        self.population_size = population_size
        self.max_eval = max_eval
        self.random_state = random_state

        self.dim = 4

        self.lower_bound = np.array([50, 2, 2, 1])
        self.upper_bound = np.array([300, 30, 20, 10])

        self.best_position = None
        self.best_fitness = None
        self.best_params = None
        self.best_score = None

        self.fitness_history = None
        self.evaluation_history = None
        self.evaluations_used = None

    def decode_position_to_params(self, position):
        n_estimators = int(round(position[0]))
        max_depth = int(round(position[1]))
        min_samples_split = int(round(position[2]))
        min_samples_leaf = int(round(position[3]))

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": self.random_state,
            "n_jobs": -1
        }

        return params

    def objective_function(self, position, X_train, y_train):
        params = self.decode_position_to_params(position)

        model = RandomForestClassifier(**params)

        cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=self.random_state)

        scores = cross_val_score(model,X_train,y_train,cv=cv,scoring="f1_macro",n_jobs=1)

        mean_f1_macro = scores.mean()

        return 1 - mean_f1_macro

    def optimize(self, X_train, y_train):
        dso = DSO(
            dim=self.dim,
            population_size=self.population_size,
            max_eval=self.max_eval,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            objective_function=lambda position: self.objective_function(position,X_train,y_train),
            seed=self.random_state)

        (   self.best_position,
            self.best_fitness,
            self.fitness_history,
            self.evaluation_history,
            self.evaluations_used) = dso.run()

        self.best_params = self.decode_position_to_params(self.best_position)
        self.best_score = 1 - self.best_fitness

        return self.best_params, self.best_score