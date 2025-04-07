# A repo for training deterministic models to predict future satellite

 <!-- Tip: This repo is a great way to get started with ML for weather forecasting. 🌤️ -->
## Installation

Create and activate a new python environment, e.g.

```
conda create -n sat_pred python=3.10
conda activate sat_pred
```

Clone this repo

```
git clone https://github.com/openclimatefix/sat_pred.git
```

Install this package and its dependencies

```
cd sat_pred
pip install -e .
```

You will also need to install the cloudcasting package following the [instructions here](https://github.com/alan-turing-institute/cloudcasting)

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
python sat_pred/train.py
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
python sat_pred/train.py model=earthformer model_name="earthformer-v1" model.optimizer.lr=0.0002
```

will train the model defined in `configs/model/earthformer.yaml` log ther training results to wandb under the name `earthformer-v1`. It will also overwrite the learning rate of the optimiser to 0.0002.


---

##  Extra Setup Tips (for new contributors)

If you're setting this up for the first time, here are some easy tips:

- Make sure you're using Python 3.8 or newer.
- It's recommended to use a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate   # On Windows use: venv\Scripts\activate






