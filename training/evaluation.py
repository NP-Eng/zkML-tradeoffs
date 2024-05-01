import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from training.train import train_model

import torch

def classification_metrics(y_test, y_pred, y_pred_proba = None):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 score": f1_score(y_test, y_pred),
        "Confusion matrix": confusion_matrix(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    }

def print_classification_metrics(y_test, y_pred, y_pred_proba = None):
    metrics = classification_metrics(y_test, y_pred, y_pred_proba)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

def predict(model, x_test):
    """
    Makes predictions using a trained model on the provided test data.
    """
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(x_test)
    return y_pred_tensor.numpy()

def test_model(X_test, y_test, model):
    """
    Tests the trained model using the test dataset and prints classification metrics.
    """
    X_test_np = X_test.to_numpy().astype(np.float32)
    y_pred = predict(model, X_test_np)
    y_pred_labels = (y_pred >= 0.5).astype(int)
    print_classification_metrics(y_test, y_pred_labels, y_pred)

def k_fold_cross_validation(x, y, in_model, loss_fn, optim, k=5):
    """
    Performs k-fold cross-validation on the provided data using the specified model.
    """
    n = len(x)
    fold_size = n // k
    indices = np.random.permutation(n)
    sample_fresh_model = lambda: in_model.__class__(x.shape[1])    
    
    accuracy = []
    model = None

    for i in range(k):
        # Initialize a fresh model for each fold
        model = sample_fresh_model()

        # Obtain a random permutation of the indices
        val_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        # Split the data into training and validation sets
        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]

        # Train the model on the training set
        model = train_model(
            model, 
            loss_fn, 
            optim(model.parameters(), lr=0.01),
            x_train, 
            y_train
        )

        # Test the model on the validation set
        y_val_pred = predict(model, x_val)
        y_val_pred_labels = (y_val_pred >= 0.5).astype(int)

        # Calculate and store the accuracy
        accuracy.append(accuracy_score(y_val, y_val_pred_labels))

    return model, sum(accuracy) / k