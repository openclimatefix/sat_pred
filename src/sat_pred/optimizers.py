import torch
from sat_pred.loss import LossFunction

class AdamW:
    """AdamW optimizer"""

    def __init__(self, lr=0.0005, **kwargs):
        """AdamW optimizer"""
        self.lr = lr
        self.kwargs = kwargs

    def __call__(self, model):
        """Return optimizer"""
        return torch.optim.AdamW(model.parameters(), lr=self.lr, **self.kwargs)

    
class AdamWReduceLROnPlateau:
    """AdamW optimizer and reduce on plateau scheduler"""

    def __init__(
        self, lr=0.0005, patience=10, factor=0.2, threshold=2e-4, step_freq=None, **opt_kwargs
    ):
        """AdamW optimizer and reduce on plateau scheduler"""
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.step_freq = step_freq
        self.opt_kwargs = opt_kwargs

    def __call__(self, model):

        opt = torch.optim.AdamW(
            model.parameters(), lr=self.lr, **self.opt_kwargs
        )

        if isinstance(model.target_loss, str):
            monitor = f"{model.target_loss}/val"
        elif isinstance(model.target_loss, LossFunction):
            monitor = f"{model.target_loss.name}/val"
        else:
            raise ValueError(f"Unknown loss type: {type(model)}")

        sch = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
            ),
            "monitor": monitor,
        }

        return [opt], [sch]