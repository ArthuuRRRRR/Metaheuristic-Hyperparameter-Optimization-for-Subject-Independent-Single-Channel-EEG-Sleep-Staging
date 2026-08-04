# Subject-Independent EEG Sleep Staging with PSO and DSO

## Overview

This repository contains an end-to-end experiment for automatic sleep-stage classification from a single electroencephalography (EEG) channel. The study compares conventional machine-learning classifiers with models tuned using particle swarm optimization (PSO) and an independently reconstructed deep sleep optimizer (DSO).

The experiments use the complete Sleep Cassette subset of the Sleep-EDF Expanded database: 153 overnight recordings from 78 subjects. Each recording is divided into 30-s epochs and classified into five scoring classes:

- Wake (`W`)
- Non-rapid-eye-movement stage 1 (`N1`)
- Non-rapid-eye-movement stage 2 (`N2`)
- Non-rapid-eye-movement stage 3 (`N3`)
- Rapid-eye-movement sleep (`REM`)

Four reference classifiers are evaluated:

- Random Forest (`RF`)
- Support Vector Machine with an RBF kernel (`SVM`)
- Logistic Regression
- Multilayer Perceptron (`MLP`)

PSO and the reconstructed DSO are then used to tune the RF and MLP hyperparameters under the same computational budget. All final results are measured on subjects excluded from training and model selection.

This project was developed as part of a graduate research course at the **Université du Québec à Rimouski (UQAR)**.

---

## Research Question

Under an equal candidate-evaluation budget and a subject-grouped validation protocol, do PSO and a reconstructed DSO improve RF and MLP sleep-stage classifiers on unseen subjects?

The project focuses on three aspects:

1. preventing subject leakage during training and evaluation;
2. comparing the two optimizers under equivalent conditions; and
3. determining whether their effects transfer across different model families.

---

## Main Results

The table below reports performance on the held-out test partition containing 20 unseen subjects.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Baseline RF | 0.5291 | 0.4816 |
| RF + PSO | 0.6040 | **0.5562** |
| RF + DSO | 0.6055 | 0.5545 |
| SVM | 0.5587 | 0.5413 |
| Baseline MLP | 0.6428 | 0.5492 |
| MLP + PSO | **0.6440** | 0.5507 |
| MLP + DSO | 0.6274 | 0.5306 |
| Logistic Regression | 0.4904 | 0.4862 |

Metaheuristic tuning clearly improved RF performance. Its macro-F1 increased from 0.4816 to 0.5562 with PSO and 0.5545 with DSO. In contrast, PSO produced only a marginal improvement for the MLP, while DSO reduced its test macro-F1. N1 remained the most difficult stage for all MLP configurations.

These findings suggest that the value of metaheuristic optimization depends on the model family and the initial configuration. For the MLP, richer features or temporal information may be more useful than additional hyperparameter search.

---

## Dataset

The project uses the [Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/) distributed through PhysioNet. Only the Sleep Cassette subset is considered.

The experiments use the `Fpz-Cz` EEG channel sampled at 100 Hz. Signals are divided into non-overlapping 30-s epochs. The original annotations are mapped to five classes as follows:

| Original annotation | Final class |
|---|---|
| Wake | W |
| Sleep stage 1 | N1 |
| Sleep stage 2 | N2 |
| Sleep stages 3 and 4 | N3 |
| REM sleep | REM |

Movement and unknown epochs are excluded. Sleep stages 3 and 4 are merged into N3 to follow the five-class staging convention used in the experiment.

The raw EDF recordings are not redistributed in this repository. Precomputed feature files are provided in compressed NumPy `.npz` format so that the classification experiments can be run without repeating the complete EDF preprocessing stage.

---

## Subject-Independent Evaluation

A subject-independent hold-out split is used to prevent epochs from the same participant from appearing in both the training and test sets. The split is generated using `GroupShuffleSplit` with `test_size=0.25` and `random_state=42`.

| Dataset partition | Subjects | Recordings | Epochs |
|---|---:|---:|---:|
| Full dataset | 78 | 153 | 195,479 |
| Training set | 58 | 113 | 145,873 |
| Test set | 20 | 40 | 49,606 |

Class distributions are shown below.

| Sleep stage | Full dataset | Training set | Test set |
|---|---:|---:|---:|
| W | 65,951 | 50,226 | 15,725 |
| N1 | 21,522 | 16,089 | 5,433 |
| N2 | 69,132 | 51,211 | 17,921 |
| N3 | 13,039 | 9,235 | 3,804 |
| REM | 25,835 | 19,112 | 6,723 |

The final test subjects are not used during scaling, cross-validation, hyperparameter optimization, or model selection. This protocol provides a more realistic estimate of generalization to unseen participants than a random epoch-level split.

---

## Feature Extraction

Each 30-s EEG epoch is represented by ten interpretable features.

### Time-domain features

- Mean
- Standard deviation
- Variance
- Minimum value
- Maximum value
- Signal energy

### Frequency-domain features

Power spectral density is estimated using Welch's method. Band power is calculated for four EEG frequency bands:

- Delta
- Theta
- Alpha
- Beta

The resulting features are cached in compressed `.npz` files. One file is created for each subject, and a combined feature file is used by the main experiment.

---

## Models and Feature Scaling

The four baseline models represent different modeling assumptions. RF provides a nonlinear tree-based ensemble and does not require standardized inputs. The RBF-SVM provides a nonlinear kernel-based classifier, logistic regression serves as a linear reference, and the MLP introduces a learnable nonlinear representation while remaining lighter than common end-to-end sleep-staging networks.

The baseline configurations are:

| Model | Main configuration |
|---|---|
| RF | 200 trees, depth 10, minimum split size 5, minimum leaf size 2, balanced class weights |
| RBF-SVM | `C=1`, `gamma="scale"`, balanced class weights |
| MLP | One hidden layer with 100 ReLU units, Adam, `alpha=1e-4`, learning rate `1e-3`, 2,000 maximum iterations |
| Logistic Regression | `C=1`, balanced class weights, 2,000 maximum iterations |

`StandardScaler` is applied to SVM, MLP, and logistic regression. Its parameters are estimated from the outer training partition and then applied unchanged to the held-out test partition. RF uses the original feature values.

---

## Hyperparameter Optimization

Two population-based metaheuristic optimizers are compared:

- Particle Swarm Optimization (`PSO`)
- Deep Sleep Optimizer (`DSO`)

Each optimizer receives a budget of 50 candidate evaluations for each optimized model. Candidate configurations are evaluated using three-fold `StratifiedGroupKFold` cross-validation, with subjects used as groups. The optimization objective is:

```text
1 - mean cross-validated macro-F1
```

This design prevents epochs from the same participant from appearing in both the training and validation folds.

### Random Forest search space

PSO and DSO optimize:

- Number of trees: 50 to 300
- Maximum tree depth: 2 to 30
- Minimum number of samples required to split a node: 2 to 20
- Minimum number of samples required at a leaf node: 1 to 10

The optimized RF models retain `max_features="sqrt"` and balanced class weights.

### Multilayer Perceptron search space

PSO and DSO optimize:

- Number of hidden layers: 1 to 3
- Number of units in each hidden layer: 20 to 200
- L2 regularization strength: `1e-5` to `1e-1`
- Initial learning rate: `1e-4` to `1e-1`

The optimized MLP models use ReLU activations, the Adam solver, and early stopping.

### DSO implementation notice

The DSO implementation used in this repository is an **independent reconstruction** based on the mechanism described in the original publication. It is not an official implementation released by the authors of the algorithm.

This implementation builds on an earlier reconstruction and numerical investigation conducted by the author:

[Deep Sleep Optimizer Research](https://github.com/ArthuuRRRRR/deep-sleep-optimizer-research)

It was adapted here from continuous numerical optimization to RF and MLP hyperparameter search. Consequently, every result labeled `DSO` refers specifically to this reconstructed variant.

---

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro-averaged precision
- Macro-averaged recall
- Macro-averaged F1-score
- Weighted F1-score
- Per-class precision, recall, and F1-score
- Confusion matrices

Macro-F1 is treated as the primary metric because the class distribution is imbalanced. Unlike accuracy, macro-F1 gives equal importance to each sleep stage.

---

## Repository Structure

```text
cours_recherche/
│
├── README.md
├── .gitignore
│
└── src/
    │
    ├── main.py
    ├── dataset_input.py
    ├── eeg_data.py
    ├── model.py
    ├── metrics.py
    │
    ├── dso/
    │   ├── __init__.py
    │   └── dso.py
    │
    ├── optimizer_pso_rf.py
    ├── optimizer_pso_mlp.py
    ├── optimizer_dso_rf.py
    ├── optimizer_dso_mlp.py
    │
    ├── final_comparison_results.csv
    │
    └── data/
        └── sleep_cassette_features/
            ├── sleep_cassette_complete.npz
            ├── subject_00.npz
            ├── subject_01.npz
            └── ...
```

### Main files

- `main.py`: runs dataset loading, splitting, model training, optimization, evaluation, and result generation.
- `dataset_input.py`: downloads the original data when required, manages the feature cache, and combines subject-level feature files.
- `eeg_data.py`: loads EEG recordings and annotations, creates labeled epochs, and extracts features.
- `model.py`: defines the baseline classifiers.
- `metrics.py`: computes evaluation metrics and generates figures.
- `dso/dso.py`: contains the reconstructed DSO implementation.
- `optimizer_pso_rf.py`: applies PSO to RF hyperparameter optimization.
- `optimizer_pso_mlp.py`: applies PSO to MLP hyperparameter optimization.
- `optimizer_dso_rf.py`: applies the reconstructed DSO to RF hyperparameter optimization.
- `optimizer_dso_mlp.py`: applies the reconstructed DSO to MLP hyperparameter optimization.
- `final_comparison_results.csv`: contains the final test metrics for all models.

---

## Installation

Python 3.10 or a more recent compatible version is recommended.

### 1. Clone the repository

```bash
git clone https://github.com/ArthuuRRRRR/Sleep-Staging-Using-DSO-Optimized-EEG-Spectrograms.git
cd Sleep-Staging-Using-DSO-Optimized-EEG-Spectrograms
```

### 2. Create a virtual environment

Using Conda:

```bash
conda create -n sleep-staging python=3.12
conda activate sleep-staging
```

Alternatively, using Python's built-in virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the main dependencies directly:

```bash
pip install numpy pandas scipy matplotlib scikit-learn mne pyswarms torch
```

PyTorch is currently imported by `main.py`, although the implemented MLP classifier itself is provided by scikit-learn.

---

## Usage

From the repository root, enter the source directory:

```bash
cd src
```

Run the complete experiment:

```bash
python main.py
```

The script performs the following operations:

1. loads the cached EEG features or creates them when they are unavailable;
2. creates the subject-independent training and test partitions;
3. standardizes features for the scale-sensitive classifiers;
4. trains the baseline models;
5. runs PSO and DSO hyperparameter optimization;
6. trains the selected model configurations on the training subjects;
7. evaluates every model on the held-out subjects; and
8. saves numerical results and comparison figures.

The full experiment is computationally expensive because each optimizer evaluates 50 configurations through three-fold grouped cross-validation. Runtime depends on processor performance, available memory, and whether the EEG features have already been cached.

---

## Precomputed Features and Caching

The directory:

```text
src/data/sleep_cassette_features/
```

contains the precomputed features used by the machine-learning experiments.

The combined file:

```text
sleep_cassette_complete.npz
```

contains the complete feature dataset. Files such as:

```text
subject_00.npz
subject_01.npz
subject_02.npz
...
```

contain the features extracted for individual subjects. Existing cache files are reused, so the corresponding EDF recordings do not need to be downloaded and processed again on subsequent executions.

If the feature cache is removed, the project can retrieve the original recordings through MNE-Python's Sleep PhysioNet interface and recreate the features. This preprocessing step can take considerably longer than loading the cached `.npz` files.

---

## Generated Outputs

Depending on the enabled evaluation functions, the pipeline can generate:

- `final_comparison_results.csv`
- `accuracy_f1_comparison.png`
- `f1_macro_comparison.png`
- `confusion_matrix_mlp.png`
- `confusion_matrix_mlp_pso.png`
- `confusion_matrix_mlp_dso.png`
- `label_distribution.png`
- execution logs

Automatically generated figures and logs may be excluded from version control through `.gitignore`.

---

## Reproducibility Notes

The following measures are used:

- subjects are separated before final model evaluation;
- the hold-out split uses `random_state=42`;
- the final test subjects are never used during model selection;
- `StratifiedGroupKFold` preserves subject separation during optimization;
- RF and scikit-learn model seeds are set to 42 where supported;
- PSO and DSO are initialized with a fixed NumPy seed;
- both optimizers receive the same candidate-evaluation budget; and
- macro-F1 is used as the optimization objective.

PSO, DSO, and model training contain stochastic operations. Small numerical differences may therefore occur across software versions or execution environments even when the configured seeds are unchanged.

### Scaling detail

For SVM, MLP, and logistic regression, the scaler is fitted on the complete outer training partition before grouped cross-validation. The held-out test subjects remain fully isolated, but the scaler is not re-fitted independently inside every internal validation fold. A fold-specific preprocessing pipeline would provide a stricter internal validation protocol and is recommended for future work.

---

## Limitations

The study has several limitations:

- only the `Fpz-Cz` EEG channel is used;
- the models rely on manually extracted features;
- temporal dependencies between consecutive epochs are not modeled explicitly;
- the class distribution is imbalanced, and N1 remains difficult to identify;
- the optimization budget is limited to 50 candidate evaluations;
- DSO is an independent reconstruction rather than an official reference implementation;
- scaling is performed on the outer training partition rather than separately within each internal cross-validation fold;
- the experiments use one primary dataset; and
- no external validation dataset is included.

Possible extensions include temporal-context features, sequence models, additional EEG channels, other physiological signals, repeated subject-level splits, larger optimization budgets, and external validation.

---

## Data Availability

The original recordings are available from PhysioNet:

- [Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/)

The raw EDF recordings are not redistributed in this repository. The statistical and frequency-domain features used by the experiments are provided in `src/data/sleep_cassette_features/` when file-size restrictions permit their inclusion.

Large `.npz` files may require [Git Large File Storage](https://git-lfs.com/) if they exceed GitHub's regular file-size limit.

---

## References

1. B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen, and J. J. L. Oberyé, "Analysis of a sleep-dependent neuronal feedback loop: The slow-wave microcontinuity of the EEG," *IEEE Transactions on Biomedical Engineering*, vol. 47, no. 9, pp. 1185–1194, 2000. https://doi.org/10.1109/10.867928

2. A. L. Goldberger *et al.*, "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals," *Circulation*, vol. 101, no. 23, pp. e215–e220, 2000. https://doi.org/10.1161/01.CIR.101.23.e215

3. J. Kennedy and R. Eberhart, "Particle swarm optimization," in *Proceedings of ICNN'95 — International Conference on Neural Networks*, vol. 4, pp. 1942–1948, 1995. https://doi.org/10.1109/ICNN.1995.488968

4. Y. Shi and R. Eberhart, "A modified particle swarm optimizer," in *1998 IEEE International Conference on Evolutionary Computation Proceedings*, pp. 69–73, 1998. https://doi.org/10.1109/ICEC.1998.699146

5. S. O. Oladejo, S. O. Ekwe, L. A. Akinyemi, and S. A. Mirjalili, "The deep sleep optimizer: A human-based metaheuristic approach," *IEEE Access*, vol. 11, pp. 83639–83665, 2023. https://doi.org/10.1109/ACCESS.2023.3298105

---

## Author

**Arthur Delhaye**  
Graduate Student in Computer Science  
Université du Québec à Rimouski  
ECE Paris

---

## Acknowledgements

This project uses the Sleep-EDF Expanded database distributed through PhysioNet. It was completed as part of a graduate research course at the Université du Québec à Rimouski.

