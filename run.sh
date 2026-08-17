#uv run python src/sat_pred/train.py model_name="simvp_v3" model=simvp_v3


uv run python src/sat_pred/train.py \
  model_name="simvp_v3_wsd" \
  model=continue_wsd_simvp_v3 \
  callbacks=finetune \
  trainer.max_epochs=80