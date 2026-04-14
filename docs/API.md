# MDT API

This API exposes the Medical Digital Twin model over HTTP.

## Endpoints

### `GET /health`
- Returns service health status.

### `POST /generate`
- Body:
  - `prompt` (string, required)
  - `max_length` (int, optional, default `512`)
  - `temperature` (float, optional, default `0.7`)
- Response:
  - `response`
  - `think_stream`
  - `patient_state`
  - `user_belief`
  - `confidence`

## Notes

- The default model is configured as `nvidia/Nemotron-Mini-4B-Instruct` in `config/configs.py`.
- First request may be slower due to model loading and caching.
