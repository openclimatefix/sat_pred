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


class AdamWWSD:
    """AdamW optimizer and warmup-stable-decay scheduler"""

    def __init__(
        self,
        lr=0.0005,
        warmup_epochs=0,
        decay_epochs=30,
        total_epochs=80,
        final_factor=0.02,
        **opt_kwargs,
    ):
        """AdamW optimizer and warmup-stable-decay scheduler

        Holds the learning rate flat for most of the run and then decays it over the final
        `decay_epochs`. The flat phase makes the schedule insensitive to where training actually
        stops, unlike a cosine, which has to be tuned to the exact budget to give its best result.

        Args:
            lr: The learning rate held during the stable phase
            warmup_epochs: Epochs spent ramping linearly up to `lr` at the start. Only needed when
                the optimizer starts cold - a run continuing a checkpoint whose moments are
                restored is already at the state `lr` was chosen for, so this can be 0
            decay_epochs: Epochs spent decaying at the end of the run. Roughly 20% of the whole
                trajectory is a reasonable default
            total_epochs: Epoch the decay finishes on. Should match `trainer.max_epochs`
            final_factor: Fraction of `lr` reached at the end of the decay
        """
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.total_epochs = total_epochs
        self.final_factor = final_factor
        self.opt_kwargs = opt_kwargs

    def __call__(self, model):
        """Return optimizer and scheduler"""

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, **self.opt_kwargs)

        schedulers = []
        milestones = []

        if self.warmup_epochs > 0:
            schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    opt,
                    start_factor=1 / (self.warmup_epochs + 1),
                    end_factor=1.0,
                    total_iters=self.warmup_epochs,
                )
            )
            milestones.append(self.warmup_epochs)

        decay_start = self.total_epochs - self.decay_epochs

        schedulers.append(torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0))
        milestones.append(decay_start)

        # `LinearLR` spans `total_iters` steps from the milestone, so reaching `final_factor` on
        # the run's last epoch takes one fewer than `decay_epochs`. It also holds `end_factor` once
        # past `total_iters`, so overrunning `total_epochs` sits at the floor - where a cosine,
        # being periodic, would instead climb back towards its peak and undo the decay
        schedulers.append(
            torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=1.0,
                end_factor=self.final_factor,
                total_iters=max(self.decay_epochs - 1, 1),
            )
        )

        sch = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=schedulers, milestones=milestones
        )

        return [opt], [sch]