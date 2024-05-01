import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, loss_fn, optimizer, x_train, y_train, epochs=200):
    """
    Trains a neural network model using the specified training data, loss function, and optimizer.
    """
    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()

    return model

def prepare_and_train(x_train, y_train, model): 
    """
    Prepares the training data and trains the model.
    """
    (x_train, y_train) = (torch.tensor(x.to_numpy().astype(np.float32)) for x in (x_train, y_train))
    y_train = y_train.reshape(-1, 1)

    return train_model(
        model,
        nn.BCELoss(),
        optim.Adam(model.parameters(), lr=0.01),
        x_train,
        y_train,
    )

