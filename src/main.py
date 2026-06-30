from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from eeg_data import load_eeg_file,attach_annotations,select_channel,print_eeg_summary, create_epochs_labelled, print_label_distribution, ID_TO_STAGE, balancing_dataset_on_wake, extract_epoch_features
from model import RandomForestModel ,LogisticRegressionModel, SVMModel, MLPModel
from metrics import evaluate_predictions
from sklearn.model_selection import train_test_split
from optimizer_pso_rf import PSO_Optimizer
from optimizer_dso_rf import DSO_Optimizer
from sklearn.preprocessing import StandardScaler

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


def all_optimizer(X_train, y_train):
    pso_optimizer = PSO_Optimizer(n_particles=10,n_iterations=10)

    best_pso_params, best_pso_score = pso_optimizer.optimize(X_train, y_train)

    print("\nBest PSO parameters")
    print("-" * 40)
    print(best_pso_params)

    print("\nBest PSO validation F1-macro")
    print("-" * 40)
    print(best_pso_score)
    
    
    print("\nDSO Optimization")
    print("=" * 40)

    dso_optimizer = DSO_Optimizer(population_size=5,max_eval=30,random_state=42)

    best_dso_params, best_dso_score = dso_optimizer.optimize(X_train, y_train)

    print("\nBest DSO parameters")
    print("-" * 40)
    print(best_dso_params)

    print("\nBest DSO validation F1-macro")
    print("-" * 40)
    print(best_dso_score)

def main() :
    psg_path = Path("C:\\Users\\delha\\Downloads\\SC4001E0-PSG.edf")
    hypnogram_path = Path("C:\\Users\\delha\\Downloads\\SC4001EC-Hypnogram.edf")
    channel_name = "EEG Fpz-Cz"

    raw = load_eeg_file(psg_path, preload=False)
    raw = attach_annotations(raw, hypnogram_path)
    raw = select_channel(raw, channel_name)

    print_eeg_summary(raw)
    plot_eeg_segment(raw, start_sec=31500, duration_sec=30)

    X, y = create_epochs_labelled(raw, epoch_length_sec=30)

    X, y = balancing_dataset_on_wake(X, y)

    X_features = extract_epoch_features(X, raw.info["sfreq"])

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
    
    X_train, X_test, y_train, y_test = train_test_split(X_features,y,test_size=0.2,random_state=42,stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #different models====================================================================================================
    
    random_forest = RandomForestModel(n_estimators=200,max_depth=10,min_samples_split=5,min_samples_leaf=2,max_features="sqrt")
    random_forest.train(X_train,y_train)
    y_pred = random_forest.predict(X_test)
    print("\nBaseline Random Forest results")
    baseline_results =evaluate_predictions(y_test,y_pred)


    pso_random_forest = RandomForestModel(n_estimators=148,max_depth=11,min_samples_split=8,min_samples_leaf=2,max_features="sqrt",class_weight="balanced",random_state=42,n_jobs=-1)
    pso_random_forest.train(X_train, y_train)
    y_pred_pso = pso_random_forest.predict(X_test)
    print("\nPSO Random Forest results")
    pso_results = evaluate_predictions(y_test, y_pred_pso)

    dso_random_forest = RandomForestModel(n_estimators=92, max_depth=13, min_samples_split=9, min_samples_leaf=4, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    dso_random_forest.train(X_train, y_train)
    y_pred_dso = dso_random_forest.predict(X_test)
    print("\nDSO Random Forest results")
    dso_results = evaluate_predictions(y_test, y_pred_dso)

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
        f"{pso_results['accuracy']:.4f}\t\t"
        f"{pso_results['precision_macro']:.4f}\t\t"
        f"{pso_results['recall_macro']:.4f}\t\t"
        f"{pso_results['f1_macro']:.4f}\t\t"
        f"{pso_results['f1_weighted']:.4f}"
    )

    print(
        f"RF + DSO\t"
        f"{dso_results['accuracy']:.4f}\t\t"
        f"{dso_results['precision_macro']:.4f}\t\t"
        f"{dso_results['recall_macro']:.4f}\t\t"
        f"{dso_results['f1_macro']:.4f}\t\t"
        f"{dso_results['f1_weighted']:.4f}"
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
        f"Logistic Regression\t"
        f"{Logistic_results['accuracy']:.4f}\t\t"
        f"{Logistic_results['precision_macro']:.4f}\t\t"
        f"{Logistic_results['recall_macro']:.4f}\t\t"
        f"{Logistic_results['f1_macro']:.4f}\t\t"
        f"{Logistic_results['f1_weighted']:.4f}"
    )



if __name__ == "__main__":
    main()