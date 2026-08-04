from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F

class LossFunction(ABC):
    """Loss function"""

    @property
    @classmethod
    @abstractmethod
    def name(self) -> str:
        """Return name of the loss function"""
        pass

    @abstractmethod
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return loss"""
        pass

class MultiscaleMAE(LossFunction):
    """Multiscale Mean Absolute Error"""

    def __init__(self, scales: list[tuple[int]]=[(1,1,1),(2,4,4)]):
        """Multiscale Mean Absolute Error"""
        self.scales = scales

    @property
    def name(self) -> str:
        return "multiscale_mae"

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return loss"""

        loss = 0

        for scale in self.scales:
            y_hat_coarse = F.avg_pool3d(input, kernel_size=list(scale))
            y_coarse = F.avg_pool3d(target, kernel_size=list(scale))

            # Pooling spreads each missing pixel across the whole window it falls in, so any coarse
            # cell which overlaps missing data is dropped. The error is zeroed at those cells
            # rather than indexed out - indexing leaves the NaN in the subtraction, and the chain
            # rule turns it back into a NaN gradient, since 0 * NaN is NaN
            valid = ~torch.isnan(y_coarse)
            error = torch.where(valid, y_hat_coarse - torch.nan_to_num(y_coarse), 0.0)
            loss = loss + error.abs().sum() / valid.sum()

        return loss / len(self.scales)
