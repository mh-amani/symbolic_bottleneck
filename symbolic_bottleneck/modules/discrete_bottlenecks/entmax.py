from .abstract_discrete_layer import AbstractDiscreteLayer
import torch
import torch.nn as nn
from entmax import sparsemax, entmax15, entmax_bisect


class EntmaxDiscreteLayer(AbstractDiscreteLayer):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__(dims, **kwargs)
        self.alpha = kwargs['alpha']

    def discretize(self, x, **kwargs) -> dict:
        x_probs = entmax_bisect(x, alpha=self.alpha, dim=-1)
        return x_probs
    
    def decode(self, x):
        return torch.argmax(x, dim=-1)

