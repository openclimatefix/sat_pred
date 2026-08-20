"""Command line tool to push locally saved model checkpoints to huggingface

use:
python push_checkpoint_to_huggingface.py "path/to/model/checkpoints" \
    --huggingface-repo="openclimatefix-models/cloudcasting_uk" \
    --wandb-repo="openclimatefix/sat_pred" \
    --local-path="~/tmp/this_model" \
    --no-push-to-hub
"""

import shutil
import tempfile
import yaml
import os
import typer
import wandb

from torch.nn import Module

from sat_pred.constants import (
    DATA_CONFIG_NAME,
    FULL_CONFIG_NAME,
    MODEL_CARD_NAME,
    MODEL_CONFIG_NAME,
    PYTORCH_WEIGHTS_NAME,
    SPATIAL_GRID_NAME,
)
from sat_pred.load_model import get_model_from_checkpoints
from sat_pred.spatial import SpatialGrid

from pathlib import Path

from safetensors.torch import save_model as save_model_as_safetensor

from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub.hf_api import HfApi


DEFAULT_CARD_TEMPLATE_PATH = (
    f"{os.path.dirname(os.path.abspath(__file__))}/model_cards/default_model_card.md"
)

app = typer.Typer(pretty_exceptions_show_locals=False)


def save_model_to_huggingface(
    model: Module,
    save_directory: str,
    model_config: dict,
    data_config: dict,
    spatial_grid: SpatialGrid,
    wandb_repo: str,
    wandb_id: str,
    experiment_config_path: str | None = None,
    card_template_path: str = DEFAULT_CARD_TEMPLATE_PATH,
    push_to_hub: bool = False,
    repo_id: str | None = None,
):
    """
    Save weights in local directory.

    Args:
        model:
            The model to save.
        save_directory:
            Path to directory in which the model weights and configuration will be saved.
        model_config:
            Model configuration specified as a key/value dictionary.
        data_config:
            Data configuration the model was trained with, specified as a key/value dictionary.
        spatial_grid:
            The grid the model was trained on.
        wandb_repo: Identifier of the repo on wandb.
        wandb_id: Identifier of the model on wandb.
        experiment_config_path: Path to the full hydra config of the training run, if it was saved.
        card_template_path: Path to the HuggingFace model card template.
        push_to_hub:
            Whether or not to push your model to the HuggingFace Hub after saving it.
        repo_id:
            ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to
            the folder name if not provided.
    """

    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    # Save model weights
    save_model_as_safetensor(model, f"{save_directory}/{PYTORCH_WEIGHTS_NAME}")

    # Save model config. `sort_keys=False` for the same reason as the data config below
    with open(save_directory / MODEL_CONFIG_NAME, 'w') as outfile:
        yaml.dump(model_config, outfile, default_flow_style=False, sort_keys=False)

    # Save the data config the model was trained with. `sort_keys=False` because yaml.dump would
    # otherwise sort every mapping in the config, including the channels - and the order the
    # channels are written in is the order the model reads them, so sorting them silently changes
    # which channel the model is shown at each index
    with open(save_directory / DATA_CONFIG_NAME, 'w') as outfile:
        yaml.dump(data_config, outfile, default_flow_style=False, sort_keys=False)

    # Save the grid the model was trained on. Unlike the experiment config below this is not
    # optional - without it nothing downstream can check that an input covers the area the model
    # was trained on
    spatial_grid.save(save_directory / SPATIAL_GRID_NAME)

    # Save the full config of the training run, if it was saved with the checkpoint
    if experiment_config_path is not None:
        shutil.copyfile(experiment_config_path, save_directory / FULL_CONFIG_NAME)

    # Create and save model card.
    card_data = ModelCardData(language="en", license="mit", library_name="pytorch")

    wandb_link = f"https://wandb.ai/{wandb_repo}/runs/{wandb_id}"

    card = ModelCard.from_template(
        card_data,
        template_path=card_template_path,
        wandb_link=wandb_link,
    )

    (save_directory / MODEL_CARD_NAME).write_text(str(card))

    # Optionally push to huggingface
    if push_to_hub:
        api = HfApi()

        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=save_directory,
            commit_message=f"Upload model - {wandb_id}",
        )

        # Print the most recent commit hash
        c = api.list_repo_commits(repo_id=repo_id, repo_type="model")[0]

        print(
            f"The latest commit is now: \n"
            f"    date: {c.created_at} \n"
            f"    commit hash: {c.commit_id}\n"
            f"    by: {c.authors}\n"
            f"    title: {c.title}\n"
        )


@app.command()
def push_to_huggingface(
    checkpoint_dir_path: str = typer.Argument(...),
    huggingface_repo: str = typer.Option("openclimatefix/cloudcasting_uk", "--huggingface-repo"),
    wandb_repo: str = typer.Option("openclimatefix/sat_pred", "--wandb-repo"),
    card_template_path: str = typer.Option(DEFAULT_CARD_TEMPLATE_PATH, "--card-template-path"),
    wandb_id: str = typer.Option(None, "--wandb-id"),
    val_best: bool = typer.Option(True),
    local_path: str = typer.Option(None, "--local-path"),
    push_to_hub: bool = typer.Option(True),
):
    """Push a local model checkpoint to a huggingface model repo

    Args:
        checkpoint_dir_path: Path of the checkpoint directory
        huggingface_repo: Name of the HuggingFace repo to push the model to
        wandb_repo: Name of the wandb repo which has the training logs
        card_template_path: Path to the HuggingFace model card template
        wandb_id: The wandb ID code - if not supplied this is taken from the checkpoint dir name
        val_best: Use best model according to val loss, else last saved model
        local_path: Where to save the local copy of the model
        push_to_hub: Whether to push the model to the hub or just create a local version
    """

    if not (push_to_hub or local_path is not None):
        raise ValueError("Either `push_to_hub` must be True or `local_path` must be set")

    # Check that the wandb-ID is correct
    all_wandb_ids = [run.id for run in wandb.Api().runs(path=wandb_repo)]

    # If the wandb run ID is not supplied infer it from the checkpoint path
    if wandb_id is None:
        wandb_id = checkpoint_dir_path.rstrip("/").split("/")[-1]

    if wandb_id not in all_wandb_ids:
        raise ValueError(f"Could not find wandb run '{wandb_id}' within {wandb_repo}")

    # Load the model
    model, model_config, data_config, spatial_grid, experiment_config_path = (
        get_model_from_checkpoints(checkpoint_dir_path, val_best=val_best)
    )

    # Push to hub
    if local_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        model_output_dir = temp_dir.name
    else:
        model_output_dir = local_path

    save_model_to_huggingface(
        model=model,
        save_directory=model_output_dir,
        model_config=model_config,
        data_config=data_config,
        spatial_grid=spatial_grid,
        experiment_config_path=experiment_config_path,
        wandb_repo=wandb_repo,
        wandb_id=wandb_id,
        card_template_path=card_template_path,
        push_to_hub=push_to_hub,
        repo_id=huggingface_repo if push_to_hub else None,
    )

    if local_path is None:
        temp_dir.cleanup()


if __name__ == "__main__":
    app()
