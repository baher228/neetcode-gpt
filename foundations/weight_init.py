import math
from typing import List

import torch


class Solution:
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        """Return a (fan_out x fan_in) Xavier-normal weight matrix."""
        torch.manual_seed(0)

        std = math.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std

        return weights.round(decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        """Return a (fan_out x fan_in) Kaiming-normal weight matrix."""
        torch.manual_seed(0)

        std = math.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std

        return weights.round(decimals=4).tolist()

    def check_activations(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        init_type: str,
    ) -> List[float]:
        """Return activation standard deviations after every layer."""
        if init_type not in {"xavier", "kaiming", "random"}:
            raise ValueError(
                "init_type must be 'xavier', 'kaiming', or 'random'"
            )

        torch.manual_seed(0)

        # Generate every layer's weights before sampling the input.
        weights = []

        for layer in range(num_layers):
            fan_in = input_dim if layer == 0 else hidden_dim
            fan_out = hidden_dim

            if init_type == "xavier":
                std = math.sqrt(2.0 / (fan_in + fan_out))
            elif init_type == "kaiming":
                std = math.sqrt(2.0 / fan_in)
            else:
                std = 1.0

            weights.append(torch.randn(fan_out, fan_in) * std)

        # Generate input after all weights, for deterministic expected output.
        x = torch.randn(input_dim)
        activation_stds = []

        for weight in weights:
            x = torch.relu(weight @ x)
            activation_stds.append(round(x.std().item(), 2))

        return activation_stds