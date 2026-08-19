"""Names of the files saved alongside a trained model

These are used when saving a model checkpoint during training, when loading it back from its
checkpoint directory, and when pushing it to huggingface. Keeping them in one place means the
three stay in step.
"""

MODEL_CONFIG_NAME = "model_config.yaml"
DATA_CONFIG_NAME = "data_config.yaml"
FULL_CONFIG_NAME = "full_experiment_config.yaml"
PYTORCH_WEIGHTS_NAME = "model.safetensors"
MODEL_CARD_NAME = "README.md"
SPATIAL_GRID_NAME = "spatial_grid.npz"
