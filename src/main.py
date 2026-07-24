from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torch import relu
from eeg_data import load_eeg_file,attach_annotations,select_channel,print_eeg_summary, create_epochs_labelled, print_label_distribution, ID_TO_STAGE, balancing_dataset_on_wake, extract_epoch_features
from model import RandomForestModel ,LogisticRegressionModel, SVMModel, MLPModel
from metrics import evaluate_predictions
from sklearn.model_selection import train_test_split
from optimizer_pso_rf import PSO_Optimizer
from optimizer_dso_rf import DSO_Optimizer
from sklearn.preprocessing import StandardScaler
from optimizer_pso_mlp import PSO_MLP_Optimizer
from optimizer_dso_mlp import DSO_MLP_Optimizer
from sklearn.model_selection import GroupShuffleSplit


import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def save_results_to_csv(results_dict, filename="final_comparison_results.csv"):
    rows = []

    for model_name, results in results_dict.items():
        rows.append({
            "Model": model_name,
            "Accuracy": results["accuracy"],
            "Precision macro": results["precision_macro"],
            "Recall macro": results["recall_macro"],
            "F1 macro": results["f1_macro"],
            "F1 weighted": results["f1_weighted"]
        })

    results_dataframe = pd.DataFrame(rows)
    results_dataframe.to_csv(filename, index=False)

    print(f"\nResults saved to {filename}")


def plot_label_distribution_graph(y, filename="label_distribution.png"):
    unique_labels, counts = np.unique(y, return_counts=True)
    label_names = [ID_TO_STAGE[label] for label in unique_labels]

    plt.figure(figsize=(8, 5))
    plt.bar(label_names, counts)
    plt.xlabel("Sleep stage")
    plt.ylabel("Number of epochs")
    plt.title("Sleep stage distribution")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_f1_macro_comparison(results_dict, filename="f1_macro_comparison.png"):
    model_names = list(results_dict.keys())
    f1_scores = [results_dict[name]["f1_macro"] for name in model_names]

    plt.figure(figsize=(10, 5))
    plt.bar(model_names, f1_scores)
    plt.xlabel("Model")
    plt.ylabel("F1 macro")
    plt.title("Model comparison based on F1 macro")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_accuracy_f1_comparison(results_dict, filename="accuracy_f1_comparison.png"):
    model_names = list(results_dict.keys())
    accuracies = [results_dict[name]["accuracy"] for name in model_names]
    f1_scores = [results_dict[name]["f1_macro"] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, accuracies, width, label="Accuracy")
    plt.bar(x + width / 2, f1_scores, width, label="F1 macro")

    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Accuracy and F1 macro comparison")
    plt.xticks(x, model_names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_confusion_matrix_graph(y_true, y_pred, title, filename):
    labels = ["W", "N1", "N2", "N3", "REM"]

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

##############################################################################################""

def plot_eeg_segment(raw, start_sec=0, duration_sec=30): # visu rapide

    sfreq = raw.info["sfreq"]

    start_sample = int(start_sec * sfreq)
    stop_sample = int((start_sec + duration_sec) * sfreq)

    data = raw.get_data(start=start_sample, stop=stop_sample)

    eeg_signal = data[0]
    time = np.arange(len(eeg_signal)) / sfreq + start_sec

    plt.figure(figsize=(12, 4))
    plt.plot(time, eeg_signal)
    plt.xlabel("Time (seconds)")
    plt.ylabel("EEG amplitude")
    plt.title(f"EEG segment from {start_sec}s to {start_sec + duration_sec}s")
    plt.grid(True)
    plt.show()


def all_optimizer_rf(X_train, y_train, groups_train):
    pso_optimizer = PSO_Optimizer(n_particles=10,n_iterations=10)

    best_pso_params, best_pso_score = pso_optimizer.optimize(X_train, y_train, groups_train)

    print("\nBest PSO parameters")
    print("-" * 40)
    print(best_pso_params)

    print("\nBest PSO validation F1-macro")
    print("-" * 40)
    print(best_pso_score)
    
    
    print("\nDSO Optimization")
    print("=" * 40)

    dso_optimizer = DSO_Optimizer(population_size=10,max_eval=100,random_state=42)

    best_dso_params, best_dso_score = dso_optimizer.optimize(X_train, y_train, groups_train)

    print("\nBest DSO parameters")
    print("-" * 40)
    print(best_dso_params)

    print("\nBest DSO validation F1-macro")
    print("-" * 40)
    print(best_dso_score)

    return best_pso_params, best_pso_score, best_dso_params, best_dso_score

def process_patient(psg_path, hypnogram_path, channel_name):
    raw = load_eeg_file(psg_path, preload=False)
    raw = attach_annotations(raw, hypnogram_path)
    raw = select_channel(raw, channel_name)

    #print_eeg_summary(raw)

    X, y = create_epochs_labelled(raw, epoch_length_sec=30)
    X, y = balancing_dataset_on_wake(X, y)

    X_features = extract_epoch_features(X, raw.info["sfreq"])

    return X_features, y


def all_optimizer_mlp(X_train_scaled, y_train, groups_train):
    print("\nPSO MLP Optimization")
    print("=" * 40)

    pso_mlp_optimizer = PSO_MLP_Optimizer(n_particles=10,n_iterations=10,random_state=42)

    best_pso_mlp_params, best_pso_mlp_score = pso_mlp_optimizer.optimize(X_train_scaled,y_train,groups_train)

    print("\nBest PSO MLP parameters")
    print("-" * 40)
    print(best_pso_mlp_params)

    print("\nBest PSO MLP validation F1-macro")
    print("-" * 40)
    print(best_pso_mlp_score)

    print("\nDSO MLP Optimization")
    print("=" * 40)

    dso_mlp_optimizer = DSO_MLP_Optimizer(population_size=10,max_eval=100,random_state=42)

    best_dso_mlp_params, best_dso_mlp_score = dso_mlp_optimizer.optimize(X_train_scaled,y_train,groups_train)

    print("\nBest DSO MLP parameters")
    print("-" * 40)
    print(best_dso_mlp_params)

    print("\nBest DSO MLP validation F1-macro")
    print("-" * 40)
    print(best_dso_mlp_score)

    return best_pso_mlp_params, best_pso_mlp_score, best_dso_mlp_params, best_dso_mlp_score


def main() :
    channel_name = "EEG Fpz-Cz"

    patients = [
        {
            "subject_id":"01",
            "psg": Path("C:\\Users\\delha\\Downloads\\SC4001E0-PSG.edf"),
            "hypnogram": Path("C:\\Users\\delha\\Downloads\\SC4001EC-Hypnogram.edf")
        },
        {
            "subject_id":"02",
            "psg": Path("C:\\Users\\delha\\Downloads\\SC4002E0-PSG.edf"),
            "hypnogram": Path("C:\\Users\\delha\\Downloads\\SC4002EC-Hypnogram.edf")
        },
        {
            "subject_id":"03",
            "psg": Path("C:\\Users\\delha\\Downloads\\SC4011E0-PSG.edf"),
            "hypnogram": Path("C:\\Users\\delha\\Downloads\\SC4011EH-Hypnogram.edf")
        },
        {
            "subject_id":"04",
            "psg": Path("C:\\Users\\delha\\Downloads\\SC4012E0-PSG.edf"),
            "hypnogram": Path("C:\\Users\\delha\\Downloads\\SC4012EC-Hypnogram.edf")
        },
        {
            "subject_id":"05",
            "psg": Path("C:\\Users\\delha\\Downloads\\SC4021E0-PSG.edf"),
            "hypnogram": Path("C:\\Users\\delha\\Downloads\\SC4021EH-Hypnogram.edf")
        },
        {
            "subject_id":"06",
            "psg": Path("C:\\Users\\delha\\Downloads\\SC4022E0-PSG.edf"),
            "hypnogram": Path("C:\\Users\\delha\\Downloads\\SC4022EJ-Hypnogram.edf")
        },
        {
            "subject_id":"07",
            "psg": Path("C:\\Users\\delha\\Downloads\\SC4031E0-PSG.edf"),
            "hypnogram": Path("C:\\Users\\delha\\Downloads\\SC4031EC-Hypnogram.edf")
        },
        {
            "subject_id":"08",
            "psg": Path("C:\\Users\\delha\\Downloads\\SC4112E0-PSG.edf"),
            "hypnogram": Path("C:\\Users\\delha\\Downloads\\SC4112EC-Hypnogram.edf")
        },
        {
            "subject_id":"09",
            "psg": Path("C:\\Users\\delha\\Downloads\\SC4201E0-PSG.edf"),
            "hypnogram": Path("C:\\Users\\delha\\Downloads\\SC4201EC-Hypnogram.edf")
        },
        {
            "subject_id":"10",
            "psg": Path("C:\\Users\\delha\\Downloads\\SC4202E0-PSG.edf"),
            "hypnogram": Path("C:\\Users\\delha\\Downloads\\SC4202EC-Hypnogram.edf")
        }

    ]

    all_X_features = []
    all_y = []

    all_groups = []

    for patient in patients:
        #print("\nProcessing patient")
        #print("=" * 40)
        #print(patient["psg"])

        X_patient, y_patient = process_patient(
            patient["psg"],
            patient["hypnogram"],
            channel_name)

        all_X_features.append(X_patient)
        all_y.append(y_patient)

        patient_groups = np.full(len(y_patient),patient["subject_id"],dtype=object)

        all_groups.append(patient_groups)

    X_features = np.concatenate(all_X_features, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    print("\nFeatures extracted")
    print("-" * 40)
    print("X features shape:", X_features.shape)
    print("y shape:", y.shape)

    """
    print("\nDataset created")
    print("-" * 40)
    print("X shape:", X.shape)
    print("y shape:", y.shape)"""
    

    print_label_distribution(y)
    """
    print("\nFirst 10 labels:")
    for label in y[:100]:
        print(label, ID_TO_STAGE[label])"""
    
    #X_train, X_test, y_train, y_test = train_test_split(X_features,y,test_size=0.2,random_state=42,stratify=y)
    ####################################################################################################################################

    group_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
    )

    train_indices, test_indices = next(
        group_split.split(
            X_features,
            y,
            groups=groups
        )
    )

    X_train = X_features[train_indices]
    X_test = X_features[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    groups_train = groups[train_indices]
    groups_test = groups[test_indices]


    train_subjects = set(groups_train)
    test_subjects = set(groups_test)

    print("\nSubject-independent split")
    print("-" * 40)
    print("Train subjects:", sorted(train_subjects))
    print("Test subjects:", sorted(test_subjects))
    print("Train epochs:", len(y_train))
    print("Test epochs:", len(y_test))
    print("Subject overlap:", train_subjects & test_subjects)

    assert train_subjects.isdisjoint(test_subjects), (
        "Data leakage: a subject is present in both train and test"
    )

    print("\nTrain label distribution")
    print_label_distribution(y_train)

    print("\nTest label distribution")
    print_label_distribution(y_test)


    ####################################################################################################################################

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #different models====================================================================================================
    
    random_forest = RandomForestModel(n_estimators=200,max_depth=10,min_samples_split=5,min_samples_leaf=2,max_features="sqrt")
    random_forest.train(X_train,y_train)
    y_pred = random_forest.predict(X_test)
    print("\nBaseline Random Forest results")
    baseline_results =evaluate_predictions(y_test,y_pred)

    svm_model = SVMModel(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=42)
    svm_model.train(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    print("\nSVM results")
    SVM_results = evaluate_predictions(y_test, y_pred_svm)

    mlp_model = MLPModel(hidden_layer_sizes=(100,), activation="relu", solver="adam", alpha=0.0001, learning_rate_init=0.001, max_iter=2000, random_state=42)
    mlp_model.train(X_train_scaled, y_train)
    y_pred_mlp = mlp_model.predict(X_test_scaled)
    print("\nMLP results")
    MLP_results = evaluate_predictions(y_test, y_pred_mlp)

    logistic_regression_model = LogisticRegressionModel(C=1.0, class_weight="balanced", random_state=42, max_iter=2000)
    logistic_regression_model.train(X_train_scaled, y_train)
    y_pred_logistic = logistic_regression_model.predict(X_test_scaled)
    print("\nLogistic Regression results")
    Logistic_results = evaluate_predictions(y_test, y_pred_logistic)



    ###############################################################Optimizer#############################

    best_pso_mlp_params, best_pso_mlp_score, best_dso_mlp_params, best_dso_mlp_score = all_optimizer_mlp(X_train_scaled,y_train,groups_train)
    best_pso_rf_params, best_pso_rf_score, best_dso_rf_params, best_dso_rf_score = all_optimizer_rf(X_train, y_train, groups_train)


    pso_MLP_model = MLPModel(**best_pso_mlp_params)
    pso_MLP_model.train(X_train_scaled, y_train)
    y_pred_pso_mlp = pso_MLP_model.predict(X_test_scaled)
    print("\nPSO MLP results")
    PSO_MLP_results = evaluate_predictions(y_test, y_pred_pso_mlp)

    dso_MLP_model = MLPModel(**best_dso_mlp_params)
    dso_MLP_model.train(X_train_scaled, y_train)
    y_pred_dso_mlp = dso_MLP_model.predict(X_test_scaled)
    print("\nDSO MLP results")
    DSO_MLP_results = evaluate_predictions(y_test, y_pred_dso_mlp)

    pso_RF_model = RandomForestModel(**best_pso_rf_params)
    pso_RF_model.train(X_train, y_train)
    y_pred_pso_rf = pso_RF_model.predict(X_test)
    print("\nPSO RF results")
    PSO_RF_results = evaluate_predictions(y_test, y_pred_pso_rf)

    dso_RF_model = RandomForestModel(**best_dso_rf_params)
    dso_RF_model.train(X_train, y_train)
    y_pred_dso_rf = dso_RF_model.predict(X_test)
    print("\nDSO RF results")
    DSO_RF_results = evaluate_predictions(y_test, y_pred_dso_rf)




    print("\nFinal comparison")
    print("=" * 60)

    print("Model\t\tAccuracy\tPrecision\tRecall\t\tF1 macro\tF1 weighted")

    print(
        f"Baseline RF\t"
        f"{baseline_results['accuracy']:.4f}\t\t"
        f"{baseline_results['precision_macro']:.4f}\t\t"
        f"{baseline_results['recall_macro']:.4f}\t\t"
        f"{baseline_results['f1_macro']:.4f}\t\t"
        f"{baseline_results['f1_weighted']:.4f}"
    )

    print(
        f"RF + PSO\t"
        f"{PSO_RF_results['accuracy']:.4f}\t\t"
        f"{PSO_RF_results['precision_macro']:.4f}\t\t"
        f"{PSO_RF_results['recall_macro']:.4f}\t\t"
        f"{PSO_RF_results['f1_macro']:.4f}\t\t"
        f"{PSO_RF_results['f1_weighted']:.4f}"
    )

    print(
        f"RF + DSO\t"
        f"{DSO_RF_results['accuracy']:.4f}\t\t"
        f"{DSO_RF_results['precision_macro']:.4f}\t\t"
        f"{DSO_RF_results['recall_macro']:.4f}\t\t"
        f"{DSO_RF_results['f1_macro']:.4f}\t\t"
        f"{DSO_RF_results['f1_weighted']:.4f}"
    )

    print(
        f"SVM\t\t"
        f"{SVM_results['accuracy']:.4f}\t\t"
        f"{SVM_results['precision_macro']:.4f}\t\t"
        f"{SVM_results['recall_macro']:.4f}\t\t"
        f"{SVM_results['f1_macro']:.4f}\t\t"
        f"{SVM_results['f1_weighted']:.4f}"
    )
    print(
        f"MLP\t\t"
        f"{MLP_results['accuracy']:.4f}\t\t"
        f"{MLP_results['precision_macro']:.4f}\t\t"
        f"{MLP_results['recall_macro']:.4f}\t\t"
        f"{MLP_results['f1_macro']:.4f}\t\t"
        f"{MLP_results['f1_weighted']:.4f}"
    )
    print(
        f"MLP + PSO\t"
        f"{PSO_MLP_results['accuracy']:.4f}\t\t"
        f"{PSO_MLP_results['precision_macro']:.4f}\t\t"
        f"{PSO_MLP_results['recall_macro']:.4f}\t\t"
        f"{PSO_MLP_results['f1_macro']:.4f}\t\t"
        f"{PSO_MLP_results['f1_weighted']:.4f}"
    )

    print(
        f"MLP + DSO\t"
        f"{DSO_MLP_results['accuracy']:.4f}\t\t"
        f"{DSO_MLP_results['precision_macro']:.4f}\t\t"
        f"{DSO_MLP_results['recall_macro']:.4f}\t\t"
        f"{DSO_MLP_results['f1_macro']:.4f}\t\t"
        f"{DSO_MLP_results['f1_weighted']:.4f}"
    )
    print(
        f"Logistic Regression\t"
        f"{Logistic_results['accuracy']:.4f}\t\t"
        f"{Logistic_results['precision_macro']:.4f}\t\t"
        f"{Logistic_results['recall_macro']:.4f}\t\t"
        f"{Logistic_results['f1_macro']:.4f}\t\t"
        f"{Logistic_results['f1_weighted']:.4f}"
    )

    ######################## 

    results_dict = {
    "Baseline RF": baseline_results,
    "RF + PSO": PSO_RF_results,
    "RF + DSO": DSO_RF_results,
    "SVM": SVM_results,
    "MLP": MLP_results,
    "MLP + PSO": PSO_MLP_results,
    "MLP + DSO": DSO_MLP_results,
    "Logistic Regression": Logistic_results
    }

    save_results_to_csv(results_dict)
    plot_label_distribution_graph(y)
    plot_f1_macro_comparison(results_dict)
    plot_accuracy_f1_comparison(results_dict)

    plot_confusion_matrix_graph(
        y_test,
        y_pred_mlp,
        "Confusion Matrix - MLP",
        "confusion_matrix_mlp.png"
    )

    plot_confusion_matrix_graph(
        y_test,
        y_pred_dso_mlp,
        "Confusion Matrix - MLP + DSO",
        "confusion_matrix_mlp_dso.png"
    )
    plot_confusion_matrix_graph(
        y_test,
        y_pred_pso_mlp,
        "Confusion Matrix - MLP + PSO",
        "confusion_matrix_mlp_pso.png"
    )


    

if __name__ == "__main__":
    main()