import numpy as np
import pyswarms as ps
import warnings

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.exceptions import ConvergenceWarning


class PSO_MLP_Optimizer:
    def __init__(self, n_particles=10, n_iterations=10, random_state=42):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.random_state = random_state

        # Position:
        # [hidden_1, hidden_2, hidden_3, n_layers, alpha_log, learning_rate_log]
        self.lower_bounds = np.array([20, 20, 20, 1, -5, -4])
        self.upper_bounds = np.array([200, 200, 200, 3, -1, -1])

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
        hidden_1 = int(round(particle[0]))
        hidden_2 = int(round(particle[1]))
        hidden_3 = int(round(particle[2]))

        n_layers = int(round(particle[3]))
        n_layers = max(1, min(3, n_layers))

        alpha = 10 ** particle[4]
        learning_rate_init = 10 ** particle[5]

        if n_layers == 1:
            hidden_layer_sizes = (hidden_1,)
        elif n_layers == 2:
            hidden_layer_sizes = (hidden_1, hidden_2)
        else:
            hidden_layer_sizes = (hidden_1, hidden_2, hidden_3)

        params = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": "relu",
            "solver": "adam",
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
            "max_iter": 1000,
            "random_state": self.random_state,
            "early_stopping": True
        }

        return params

    def objective_function(self, particles, X_train, y_train):
        losses = []

        cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=self.random_state
        )

        for particle in particles:
            params = self.decode_particle_to_params(particle)

            model = MLPClassifier(**params)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="f1_macro",
                    n_jobs=1
                )

            mean_f1_macro = scores.mean()
            loss = 1 - mean_f1_macro

            losses.append(loss)

        return np.array(losses)

    def optimize(self, X_train, y_train):
        bounds = (self.lower_bounds, self.upper_bounds)

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=6,
            options=self.options,
            bounds=bounds
        )

        self.best_cost, self.best_position = optimizer.optimize(
            lambda particles: self.objective_function(
                particles,
                X_train,
                y_train
            ),
            iters=self.n_iterations
        )

        self.best_params = self.decode_particle_to_params(self.best_position)
        self.best_score = 1 - self.best_cost

        return self.best_params, self.best_score