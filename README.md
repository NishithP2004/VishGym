# VishGym

VishGym is a closed, synthetic red/blue self-play arena for studying AI-powered payment-fraud defense. It is deliberately not a calling, messaging, browsing, payment, credential, or voice-cloning service.

## Safety boundary

- Every persona, credential, wallet, identifier, inbox entry, web page, and audio clip is fictional.
- The environment has no real payment rail, networked browser, external search, outbound messaging, or externally reachable portal.
- The public UI accepts no voice uploads or reference clips. Qwen3-TTS CustomVoice is the intended live renderer, using its built-in timbres only.
- The bundled demo uses deterministic local policies and a synthetic WAV fallback. GPU model loading is opt-in through environment configuration.

## Run the deterministic demo

```bash
pip install -e .
python -m vishgym.cli
```

## Install the web app

```bash
pip install -e '.[app]'
uvicorn vishgym.api.main:app --host 0.0.0.0 --port 8000
streamlit run app/vishgym/ui/dashboard.py
```

## Training

Use `notebooks/VishGym_Colab.ipynb` through the Google Colab MCP bridge. The notebook validates the sandbox, builds synthetic rollouts, and records the alternating QLoRA/GRPO promotion workflow. It intentionally requires a human review before publishing any candidate adapter.

Run the typed OpenEnv server in a GPU/Colab environment after installing the training extra:

```bash
pip install -e '.[training]'
uvicorn vishgym.arena.openenv_adapter:app --host 0.0.0.0 --port 8000
```

The OpenEnv adapter returns only redacted observations and state; the transcript remains server-side for terminal judging. To use live models, load three reviewed, immutable adapter paths through `SharedGemmaAdapterRuntime`, then give the Red/Blue policies and fixed judge the same runtime instance. The public demo intentionally does not load model weights or accept model/voice uploads.

## Verification

```bash
PYTHONPATH=app python -m unittest discover -s app/tests -v
```

## Project layout

- `app/vishgym/arena` — closed world, tool handlers, action schemas, reward judge, and OpenEnv adapter.
- `app/vishgym/core` — fictional fixtures and role harnesses.
- `app/vishgym/api` / `app/vishgym/ui` — FastAPI and Streamlit interfaces.
- `deploy` — Hugging Face Docker Space configuration.
- `docs` — safety and technical documentation; `artifacts` holds safe model-manifest templates only.
