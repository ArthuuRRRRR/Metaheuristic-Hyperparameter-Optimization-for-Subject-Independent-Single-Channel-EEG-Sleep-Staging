import numpy as np
import pyswarms as ps

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score


class PSO_Optimizer:
    def __init__(self,n_particles=10,n_iterations=10,random_state=42):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.lower_bounds = np.array([50, 2, 2, 1])
        self.upper_bounds = np.array([300, 30, 20, 10])

        self.options = {
            "c1": 1.5,
            "c2": 1.5,
            "w": 0.7
        }

        self.best_position = None
        self.best_cost = None
        self.best_params = None
        self.best_score = None

    def decode_particle_to_params(self, particle):
        n_estimators = int(round(particle[0]))
        max_depth = int(round(particle[1]))
        min_samples_split = int(round(particle[2]))
        min_samples_leaf = int(round(particle[3]))

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

    def objective_function(self, particles, X_train, y_train, groups_train):
        losses = []

        cv = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=self.random_state)

        for particle in particles:
            params = self.decode_particle_to_params(particle)

            model = RandomForestClassifier(**params)

            scores = cross_val_score(model,X_train,y_train,groups=groups_train,cv=cv,scoring="f1_macro",n_jobs=1)

            mean_f1_macro = scores.mean()

            loss = 1 - mean_f1_macro
            losses.append(loss)

        return np.array(losses)

    def optimize(self, X_train, y_train, groups_train):
        np.random.seed(self.random_state)
        bounds = (self.lower_bounds, self.upper_bounds)

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=4,
            options=self.options,
            bounds=bounds
        )

        self.best_cost, self.best_position = optimizer.optimize(
            lambda particles: self.objective_function(
                particles,
                X_train,
                y_train,
                groups_train
            ),
            iters=self.n_iterations
        )

        self.best_params = self.decode_particle_to_params(self.best_position)
        self.best_score = 1 - self.best_cost

        return self.best_params, self.best_score