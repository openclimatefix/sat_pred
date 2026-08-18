"""Train the model using parameters in the supplied config files."""

import os
import hydra
from lightning.pytorch import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

import rich.syntax
import rich.tree
from lightning.pytorch.utilities import rank_zero_only

from sat_pred.constants import DATA_CONFIG_NAME, FULL_CONFIG_NAME, MODEL_CONFIG_NAME
from sat_pred.load_model import get_checkpoint_path, get_model_from_checkpoints
from sat_pred.loss import LossFunction


def resolve_loss_name(loss):
    """Return the desired metric to monitor based on the loss being used.
    """
    
    if isinstance(loss, str):
        return loss
    else:
        loss = hydra.utils.instantiate(loss, _convert_='all')
        if isinstance(loss, LossFunction):
            return loss.name
        else:
            raise ValueError(f"Unknown loss type: {type(loss)}")

OmegaConf.register_new_resolver("resolve_loss_name", resolve_loss_name)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: list[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


@hydra.main(config_path="../../configs/", config_name="config.yaml", version_base="1.2")
def train(config: DictConfig):
    """Train the model using parameters in the supplied config files.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    print_config(config)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    if config.model.model.get("from_pretrained", False):

        # Held onto because the config they live in is overwritten a few lines below
        checkpoint_dir = config.model.model.checkpoint_dir
        val_best = config.model.model.val_best

        # Load the model from the checkpoint
        torch_model, model_config, _, _ = get_model_from_checkpoints(
            checkpoint_dir,
            val_best=val_best
        )

        # Overwrite the model config with the loaded model config
        config.model.model = OmegaConf.create(model_config)

        # Instantiate the LightningModule with the loaded model
        model: LightningModule = hydra.utils.instantiate(config.model)

        # Replace the untrained model with the loaded model
        model.model = torch_model

        # The optimizer moments sit in the same checkpoint file as the weights, but cannot be
        # loaded until the optimizer exists. Point the module at the file - it restores them itself
        # once Lightning has built the optimizer
        if model.restore_optimizer_state:
            model.optimizer_state_path = get_checkpoint_path(checkpoint_dir, val_best=val_best)

    else:
        model: LightningModule = hydra.utils.instantiate(config.model)

    model.multi_gpu = len(config.trainer.devices) > 1

    loggers: list[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                loggers.append(hydra.utils.instantiate(lg_conf))

    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Align the wandb id with the checkpoint path
    # - only works if wandb logger and model checkpoint used
    use_wandb_logger = False
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            use_wandb_logger = True
            wandb_logger = logger
            break

    if use_wandb_logger:
        # Calling the .experiment property initialises the logger
        wandb_run = wandb_logger.experiment

        # Lightning sends `trainer/global_step` alongside every metric but does not pass a step to
        # `wandb.log`, so wandb plots against its own `_step`, which counts log calls rather than
        # optimiser steps. Gradient accumulation and `log_every_n_steps` together mean a whole run
        # is only a handful of log calls, so the default x-axis reads 0, 1, 2... Pointing every
        # metric at `trainer/global_step` puts all the panels on the optimiser step instead
        wandb_run.define_metric("trainer/global_step")
        wandb_run.define_metric("*", step_metric="trainer/global_step")

        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                #  skip for non-rank-0 processes:
                # see https://github.com/Lightning-AI/pytorch-lightning/issues/13166#issuecomment-1139765549
                if wandb_logger.version is None:
                    break

                callback.dirpath = "/".join(
                    callback.dirpath.split("/")[:-1] + [wandb_logger.version]
                )
                # Also save model config to this path
                os.makedirs(callback.dirpath, exist_ok=True)
                OmegaConf.save(config.model, f"{callback.dirpath}/{MODEL_CONFIG_NAME}")

                # Similarly save the data config
                OmegaConf.save(config.datamodule, f"{callback.dirpath}/{DATA_CONFIG_NAME}")

                # Save the full resolved hydra config to this path and upload it to wandb
                full_config_path = f"{callback.dirpath}/{FULL_CONFIG_NAME}"
                OmegaConf.save(config, full_config_path, resolve=True)
                wandb_logger.experiment.save(full_config_path, base_path=callback.dirpath)

                break

    # Instantiate the datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, _convert_='all')
    
    datamodule.zarr_path = list(datamodule.zarr_path)

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=loggers,
        _convert_="partial",
        callbacks=callbacks,
    )

    trainer.fit(model=model, datamodule=datamodule)
    
    
if __name__ == "__main__":    
    train()
