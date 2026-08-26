import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:

        x = np.array(x, dtype=float)             # (n_inputs,)
        W1 = np.array(W1, dtype=float)           # (n_hidden, n_inputs)
        b1 = np.array(b1, dtype=float)           # (n_hidden,)
        W2 = np.array(W2, dtype=float)           # (n_outputs, n_hidden)
        b2 = np.array(b2, dtype=float)           # (n_outputs,)
        y_true = np.array(y_true, dtype=float)   # (n_outputs,)

        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1) 

        y_pred = W2 @ a1 + b2

        loss = np.mean((y_pred - y_true)**2)

        dLdy = 2*(y_pred - y_true)
        dLdW2 = np.outer(dLdy, a1)
        dLdb2 = dLdy
        dLda1 = W2.T @ dLdy
        dLdz1 = dLda1 * (z1 > 0)
        dLdW1 = np.outer(dLdz1,x)
        dLdb1 = dLdz1
        return {
            "loss":  np.round(float(loss), 4),
            "dW1": np.round(dLdW1, 4),
            "db1": np.round(dLdb1, 4),
            "dW2": np.round(dLdW2, 4),
            "db2": np.round(dLdb2, 4),
        }
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)

