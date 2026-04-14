# Troubleshooting

## CUDA device-side assert during generation

- Ensure tokenizer and model vocab are aligned.
- Use the startup diagnostics in `models/mdt_model.py` to verify embedding/token dimensions.
- Set `CUDA_LAUNCH_BLOCKING=1` for deterministic stack traces when debugging.

## Out-of-memory during inference

- Reduce `max_length` and/or batch size.
- Enable lower precision or quantization in model config.
- The model wrapper includes an OOM retry path for safer fallback.

## Training resume behavior

- SFT now supports resume from checkpoint through trainer integration.
- Use the most recent valid checkpoint to continue interrupted jobs.
