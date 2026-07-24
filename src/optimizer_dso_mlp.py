import numpy as np
import warnings

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.exceptions import ConvergenceWarning

from dso import DSO


class DSO_MLP_Optimizer:
    def __init__(self, population_size=5, max_eval=30, random_state=42):
        self.population_size = population_size
        self.max_eval = max_eval
        self.random_state = random_state

        # Position:
        # [hidden_1, hidden_2, hidden_3, n_layers, alpha_log, learning_rate_log]
        self.dim = 6

        self.lower_bound = np.array([20, 20, 20, 1, -5, -4])
        self.upper_bound = np.array([200, 200, 200, 3, -1, -1])

        self.best_position = None
        self.best_fitness = None
        self.best_params = None
        self.best_score = None

        self.fitness_history = None
        self.evaluation_history = None
        self.evaluations_used = None

    def decode_position_to_params(self, position):
        hidden_1 = int(round(position[0]))
        hidden_2 = int(round(position[1]))
        hidden_3 = int(round(position[2]))

        n_layers = int(round(position[3]))
        n_layers = max(1, min(3, n_layers))

        alpha = 10 ** position[4]
        learning_rate_init = 10 ** position[5]

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

    def objective_function(self, position, X_train, y_train, groups_train):
        params = self.decode_position_to_params(position)

        model = MLPClassifier(**params)

        cv = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=self.random_state)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            scores = cross_val_score(
                model,
                X_train,
                y_train,
                groups=groups_train,
                cv=cv,
                scoring="f1_macro",
                n_jobs=1)

        mean_f1_macro = scores.mean()

        return 1 - mean_f1_macro

    def optimize(self, X_train, y_train, groups_train):
        dso = DSO(
            dim=self.dim,
            population_size=self.population_size,
            max_eval=self.max_eval,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            objective_function=lambda position: self.objective_function(
                position,
                X_train,
                y_train,
                groups_train),
            seed=self.random_state
        )

        (
            self.best_position,
            self.best_fitness,
            self.fitness_history,
            self.evaluation_history,
            self.evaluations_used
        ) = dso.run()

        self.best_params = self.decode_position_to_params(self.best_position)
        self.best_score = 1 - self.best_fitness

        return self.best_params, self.best_score