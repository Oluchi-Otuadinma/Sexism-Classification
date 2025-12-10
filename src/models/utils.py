from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="weighted"),
        "report": classification_report(y_test, preds)
    }
