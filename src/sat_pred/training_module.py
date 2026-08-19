"""Training class to wrap model and optimizer"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
import lightning.pytorch as pl

import wandb

from sat_pred.optimizers import AdamWReduceLROnPlateau
from sat_pred.loss import LossFunction
from sat_pred.channels import ChannelNormalisation

# The samples are per-channel z-scores. A video whose target holds nothing to take a range from
# falls back to this many standard deviations either side of the channel mean
VIDEO_CLIP_SIGMA = 3.0

# The greyscale range is padded by this fraction of the target's own range at each end, so the
# darkest and brightest pixels of the target are not drawn as pure black and pure white
VIDEO_RANGE_BUFFER = 0.05


def video_greyscale_limits(y: torch.Tensor) -> tuple[float, float]:
    """Choose the values black and white stand for in a video of `y`

    The limits span the whole target sequence rather than a fixed window around the channel mean.
    A fixed window has to be wide enough for every channel and every scene, so for the channels
    which only ever occupy part of it the picture comes out flat. Taking the range from the target
    itself spends the whole greyscale on the values the video actually contains, and holding one
    pair of limits across all its frames keeps brightness comparable from frame to frame.

    Args:
        y: The true future sequence of a single channel, as z-scores

    Returns:
        The z-scores which black and white stand for, padded by `VIDEO_RANGE_BUFFER`
    """
    # The target keeps its missing pixels as NaN, which must not drag the limits to NaN
    finite = y[torch.isfinite(y)]

    if finite.numel() > 0:
        low, high = finite.min().item(), finite.max().item()
        if high > low:
            buffer = VIDEO_RANGE_BUFFER * (high - low)
            return low - buffer, high + buffer

    # An entirely missing or entirely flat target has no range to stretch. There is no good picture
    # to draw either way, so fall back to the window around the channel mean
    return -VIDEO_CLIP_SIGMA, VIDEO_CLIP_SIGMA


def greyscale_range_label(
    vmin: float, vmax: float, normalisation: ChannelNormalisation
) -> str:
    """Describe the physical values which black and white stand for in a video

    wandb draws videos without any axes or colour bar, so the only way a viewer can tell what the
    greyscale means is if it is written into the name the video is logged under.

    Args:
        vmin: The z-score drawn as black, from `video_greyscale_limits`
        vmax: The z-score drawn as white, from `video_greyscale_limits`
        normalisation: The constants the channel was z-scored with
    """
    low = normalisation.mean + vmin * normalisation.std
    high = normalisation.mean + vmax * normalisation.std
    return f"[{low:.0f}-{high:.0f}]"


def upload_video(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    video_name: str,
    wandb_run,
    step: int,
    vmin: float,
    vmax: float,
    channel_num: int = 8,
    fps: int=4
) -> None:
    """Upload prediction video to wandb

    The sequences are per-channel z-scores. They are drawn over the `vmin` to `vmax` window, which
    is the same picture as denormalising and windowing the physical values. The prediction is drawn
    over the window the target set, so the two halves of the video are directly comparable, and a
    prediction which runs outside it is clipped. Use `greyscale_range_label` to say in the video
    name which physical values the window covers.

    Args:
        y: The true future satellite sequence
        y_hat: The predicted future satellite sequence
        video_name: The name under which to log the video
        wandb_run: The wandb run to log the video to
        step: The step to log the video against. Should be the trainer global step so the videos
            line up with the metrics logged by lightning
        vmin: The z-score to draw as black, from `video_greyscale_limits`
        vmax: The z-score to draw as white, from `video_greyscale_limits`
        channel_num: The channel number to log
        fps: The frames per second of the video
    """
    y = y.cpu().numpy()
    y_hat = y_hat.cpu().numpy()

    # The y coordinate increases northwards but images are drawn from the top row down, so the
    # y axis is flipped to put north at the top. The x coordinate increases eastwards, which
    # already matches images being drawn left to right, so it is left alone
    y_frames = y.transpose(1,0,2,3)[:, channel_num:channel_num+1, ::-1, :]
    y_hat_frames = y_hat.transpose(1,0,2,3)[:, channel_num:channel_num+1, ::-1, :]

    channel_frames = np.concatenate([y_hat_frames, y_frames], axis=3)

    # Stretch the vmin to vmax window across the greyscale range
    channel_frames = channel_frames.clip(vmin, vmax)
    channel_frames = (channel_frames - vmin) / (vmax - vmin)

    # The target keeps its missing pixels as NaN. There is no value to draw for them, so they are
    # drawn black
    channel_frames = np.nan_to_num(channel_frames, nan=0.0)

    channel_frames = np.repeat(channel_frames, 3, axis=1)*255
    channel_frames = channel_frames.astype(np.uint8)
    # `format` is optional in wandb<0.20 but defaults to gif with a warning, and is required from
    # 0.20 onwards. It is passed explicitly to pin the behaviour across both
    # The step is logged as a value rather than passed as `step=`. Lightning logs its metrics
    # without a step, so wandb's internal counter is a count of log calls and is far behind the
    # global step. Passing `step=` here would fast-forward that counter to the global step and
    # leave a gap in every metric, whereas `trainer/global_step` is the axis the metrics use
    wandb_run.log(
        {
            video_name: wandb.Video(channel_frames, fps=fps, format="gif"),
            "trainer/global_step": step,
        }
    )
    
    
class TrainingModule(pl.LightningModule):

    def __init__(
        self,
        model: torch.nn.Module,
        target_loss: str = "MAE",
        optimizer = AdamWReduceLROnPlateau(),
        video_plot_t0_times: list[str] = None,
        video_crop_plots=None,
        multi_gpu: bool = False,
        restore_optimizer_state: bool = False,
    ):
        """Lightning module to wrap model, optimizer, and training routine

        Args:
            model: The model to train
            target_loss: The loss to minimize. One of "MAE", "MSE"
            optimizer: The optimizer to use. Defaults to AdamWReduceLROnPlateau().
            restore_optimizer_state: Whether to carry the optimizer moments over from the
                checkpoint the weights were loaded from. Only has an effect alongside
                `model.from_pretrained` - `train.py` finds the checkpoint and sets
                `optimizer_state_path`, and the moments are loaded in `on_fit_start`
        """
        super().__init__()

        assert target_loss in ["MAE", "MSE"] or isinstance(target_loss, LossFunction)

        self.model = model
        self._optimizer = optimizer

        self.target_loss = target_loss

        self.video_plot_t0_times = video_plot_t0_times
        self.video_crop_plots = video_crop_plots
        self.multi_gpu = multi_gpu

        self.restore_optimizer_state = restore_optimizer_state
        self.optimizer_state_path = None

    def on_fit_start(self):
        """Load the optimizer moments saved in a checkpoint, if one has been supplied

        The weights are loaded before training starts, but the optimizer does not exist until
        Lightning has called `configure_optimizers`, so the moments are restored here instead.
        Without this a continued run rebuilds `exp_avg_sq` from scratch, discarding the
        per-parameter gradient scaling learned over the whole of the previous run
        """

        if self.optimizer_state_path is None:
            return

        if len(self.trainer.optimizers) != 1:
            raise ValueError(
                f"Expected a single optimizer to restore state into, found "
                f"{len(self.trainer.optimizers)}"
            )

        optimizer = self.trainer.optimizers[0]
        checkpoint = torch.load(self.optimizer_state_path, map_location="cpu")

        # `load_state_dict` replaces `param_groups` as well as the moments, which would quietly
        # swap this run's learning rate and weight decay for the ones the checkpointed run used -
        # and the learning rate in particular is the thing being changed. Only the moments are
        # wanted, so the configured hyperparameters go back afterwards
        hyperparameters = [
            {k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups
        ]
        optimizer.load_state_dict(checkpoint["optimizer_states"][0])
        for group, saved in zip(optimizer.param_groups, hyperparameters, strict=True):
            group.update(saved)

    def _calculate_common_losses(
        self, 
        y: torch.Tensor, 
        y_hat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Calculate losses common to train and val
        
        Args:
            y: The true future satellite sequence
            y_hat: The predicted future satellite sequence
        """
        
        losses = {}

        # Pixels the satellite did not record come through as NaN and are excluded. The error is
        # zeroed at those pixels rather than indexed out, because indexing leaves the NaN in the
        # subtraction and the chain rule then turns it back into a NaN gradient - 0 * NaN is NaN.
        # A NaN anywhere in the error poisons every weight gradient, and under mixed precision the
        # GradScaler silently skips those steps, so training would appear to run and learn nothing
        valid = ~torch.isnan(y)
        n_valid = valid.sum()

        # `y` keeps its NaNs so the model's own NaNs still show up in the loss and trip the guard
        # in `training_step`. Only the missing-pixel NaNs are removed, and only from the error
        error = torch.where(valid, y_hat - torch.nan_to_num(y), 0.0)

        mse_loss = error.square().sum() / n_valid
        mae_loss = error.abs().sum() / n_valid

        losses = {
                "MSE": mse_loss,
                "MAE": mae_loss,
        }

        if isinstance(self.target_loss, LossFunction):
            losses[self.target_loss.name] = self.target_loss(y_hat, y)

        return losses

    def _calculate_val_losses(
        self, 
        y: torch.Tensor, 
        y_hat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Calculate additional validation losses
        
        Args:
            y: The true future satellite sequence
            y_hat: The predicted future satellite sequence
        """

        losses = {}

        return losses
    
    def training_step(self, batch, batch_idx: int) -> torch.Tensor | None:
        """Run training step"""

        X, y = batch

        y_hat = self.model(X)
        del X

        losses = self._calculate_common_losses(y, y_hat)
        losses = {f"{k}/train": v for k, v in losses.items()}

        if isinstance(self.target_loss, LossFunction):
            train_loss = losses[f"{self.target_loss.name}/train"]
        else:
            train_loss = losses[f"{self.target_loss}/train"]

        # Occasionally y will be entirely NaN and we have no training targets. So the train loss
        # will also be NaN. Every loss divides by the same count of valid pixels, so they are all
        # NaN together and none of them are logged - a single NaN would otherwise drag the epoch
        # mean to NaN for the rest of the epoch, the same way it is filtered out in validation
        if torch.isnan(train_loss).item():
            print("\n\nTraining loss is nan\n\n")
            if self.multi_gpu:
                # For multi-GPU we need to return some kind of loss
                return F.l1_loss(y_hat*0, y_hat*0)
            else:
                # For single GPU we return None so lightning skips this train step
                return None

        # The metrics are logged as tensors rather than floats. Pulling each one back with `.item()`
        # would sync the GPU on every micro-batch, and with grad accumulation there are many of
        # those per optimizer step. Lightning only pushes step metrics to the logger once the
        # accumulation window closes, so the step curve is the last micro-batch of each window
        # rather than its mean. The epoch metric is a true mean over every batch either way
        self.log_dict(
            {k: v.detach() for k, v in losses.items()},
            on_step=True,
            on_epoch=True,
        )

        return train_loss

    def validation_step(self, batch: dict, batch_idx: int):
        """Run validation step"""
        X, y = batch
        y_hat = self.model(X)
    
        losses = self._calculate_common_losses(y, y_hat)
        losses.update(self._calculate_val_losses(y, y_hat))

        # Rename and convert metrics to float
        losses = {f"{k}/val": v.item() for k, v in losses.items()}
        
        # Occasionally y will be entirely NaN and we have no training targets. So the val loss
        # will also be NaN. We filter these out
        non_nan_losses = {k: v for k, v in losses.items() if not np.isnan(v)}

        self.log_dict(
            non_nan_losses,
            on_step=False,
            on_epoch=True,
        )
        
    def _predict_video_samples(self, val_dataset, dates):
        """Predict the video samples for the given t0 times, one sample at a time.

        These samples are full resolution and the model is memory hungry, so collating all the
        dates into one batch peaks well above a training step and can exhaust the GPU. The samples
        are run one at a time, in the same precision the trainer uses elsewhere, and moved back to
        the host as they are finished.

        Args:
            val_dataset: The validation dataset to pull the samples from
            dates: The t0 times to predict

        Returns:
            Lists of the true and predicted future sequences, on the CPU and in float32
        """
        ys, y_hats = [], []

        for date in dates:
            # The targets are only ever drawn, so they are left on the host
            X, y = default_collate([val_dataset[date]])
            X = X.to(self.device)

            with torch.no_grad(), self.trainer.precision_plugin.forward_context():
                y_hat = self.model(X)

            ys.append(y[0].float())
            y_hats.append(y_hat[0].float().cpu())

            del X, y_hat

        return ys, y_hats

    def on_validation_epoch_start(self):

        # The videos are logged to wandb, so there is nothing to do without a wandb logger
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return

        # Training leaves the caching allocator holding blocks sized for the training graph. They
        # are a poor fit for the shapes below, so they are released rather than fragmented around
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        wandb_run = self.logger.experiment
        step = self.trainer.global_step

        # Upload videos of the first three validation samples
        val_dataset = self.trainer.val_dataloaders.dataset

        if self.video_plot_t0_times is not None:
            dates = pd.to_datetime(list(self.video_plot_t0_times))

            y, y_hat = self._predict_video_samples(val_dataset, dates)

            for i in range(len(dates)):

                for channel_name in ["VIS008", "IR_039"]:
                    channel_num = val_dataset.da.channel.values.tolist().index(channel_name)
                    vmin, vmax = video_greyscale_limits(y[i][channel_num])
                    value_range = greyscale_range_label(
                        vmin, vmax, val_dataset.channel_config[channel_name]
                    )
                    video_name = f"val_sample_videos/{dates[i]}_{channel_name} {value_range}"
                    upload_video(
                        y[i], y_hat[i], video_name, wandb_run, step, vmin, vmax, channel_num
                    )

        if self.video_crop_plots is not None:
            dates = pd.to_datetime([x["date"] for x in self.video_crop_plots])

            y, y_hat = self._predict_video_samples(val_dataset, dates)

            for n in range(len(self.video_crop_plots)):

                date = dates[n]
                channel_num = 8
                channel_name = val_dataset.da.channel.values[channel_num]
                i = self.video_crop_plots[n]["i"]
                j = self.video_crop_plots[n]["j"]
                s = self.video_crop_plots[n]["s"]

                i_slice = slice(max(0, i-s//2), i+s//2)
                j_slice = slice(max(0, j-s//2), j+s//2)
                y_crop = y[n][..., i_slice, j_slice]
                y_hat_crop = y_hat[n][..., i_slice, j_slice]

                # The limits come from the crop rather than the full image, so a close up of a
                # narrow range of values is not left washed out by the rest of the scene
                vmin, vmax = video_greyscale_limits(y_crop[channel_num])
                value_range = greyscale_range_label(
                    vmin, vmax, val_dataset.channel_config[channel_name]
                )
                video_name = (
                    f"val_close_up_sample_videos/{date}_{channel_name}_{i=}_{j=}_{s=} {value_range}"
                )

                upload_video(
                    y_crop,
                    y_hat_crop,
                    video_name,
                    wandb_run,
                    step,
                    vmin,
                    vmax,
                    channel_num,
                )

    def on_validation_epoch_end(self):
        # Clear cache at the end of validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def configure_optimizers(self):
        return self._optimizer(self)