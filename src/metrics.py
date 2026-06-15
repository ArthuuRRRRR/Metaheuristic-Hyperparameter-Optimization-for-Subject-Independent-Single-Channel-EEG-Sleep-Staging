from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


def evaluate_predictions(y_true, y_pred):

    print("\nAccuracy")
    print("-" * 40)
    print(accuracy_score(y_true, y_pred))

    print("\nClassification report")
    print("-" * 40)
    print(classification_report(y_true, y_pred))

    print("\nConfusion matrix")
    print("-" * 40)
    print(confusion_matrix(y_true, y_pred))