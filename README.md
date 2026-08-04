# A repo for training deterministic models to predict future satellite


## Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage its environment and dependencies.

Clone this repo and sync the environment

```
git clone https://github.com/openclimatefix/sat_pred.git
cd sat_pred
uv sync
```

This creates a `.venv` in the repo with the package installed in editable mode, along with the
development dependencies. Prefix commands with `uv run` to use it, or activate it with
`source .venv/bin/activate`.

If you would rather manage the environment yourself, `pip install -e .` still works in any
python 3.11-3.13 environment.

Two dependencies are not on PyPI and must be installed separately.

You will need the cloudcasting package, following the [instructions here](https://github.com/alan-turing-institute/cloudcasting)

If you want to train the earthformer model you should clone and install the earthformer repo as well

```
cd ..
git clone https://github.com/amazon-science/earth-forecasting-transformer.git
cd earth-forecasting-transformer
pip install -e .
```

## Training

You can train a model by running

```
python src/sat_pred/train.py
```

from the root of the library. 

The model and training options used are defined in the config files. The most important parts of the config files you may wish to train are:

- `configs/datamodule/default.yaml`
  - `zarr_paths` which point to your training data
  - `train/val_period` which control the train / val split used
  - `num_workers` and `batch_size` to suit your machine

- `configs/logger/wandb.yaml`
  - Set `project` to the project name you want to save the runs to on wandb

- `configs/trainer/default.yaml`
  - This control the parameters for the lightning Trainer. See https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api
  - Note you might want to set `fast_dev_run` to `true` to aid with testing and getting set up

- `configs/config.yaml`
  - Set `model_name` to the name the run will be logged under on wandb
  - Set `defaults:model` to one of the model config filenames within `configs/model`

Note that since we use hydra to build up the configs, you can change the configs from the command line when running the training job. For example

```
python src/sat_pred/train.py model=earthformer model_name="earthformer-v1" model.optimizer.lr=0.0002
```

will train the model defined in `configs/model/earthformer.yaml` log ther training results to wandb under the name `earthformer-v1`. It will also overwrite the learning rate of the optimiser to 0.0002.






