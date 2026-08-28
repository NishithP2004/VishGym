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

Use `notebooks/VishGym_Colab.ipynb` through the Google Colab MCP bridge. The notebook now performs the complete local-only sequence: Qwen CustomVoice dataset export, one QLoRA warm-start adapter per role, group-relative self-play against a frozen opponent, held-out evaluation, and review-only manifests. It intentionally cannot publish, deploy, or swap an adapter.

The command-line equivalent is:

```bash
# A test-tone export validates the artifact pipeline but is intentionally rejected by training.
vishgym-train export-dataset --renderer synthetic-test --output-dir artifacts/datasets/test-v1

# A Colab GPU run must use CustomVoice audio and an HF_TOKEN stored in Colab Secrets.
vishgym-train export-dataset --renderer qwen --output-dir artifacts/datasets/warm-start-qwen-v1 --seeds 7 11
vishgym-train warm-start --dataset-root artifacts/datasets/warm-start-qwen-v1 --role red --output-dir artifacts/adapters/red-sft-v1
vishgym-train grpo --role red --initial-adapter-path artifacts/adapters/red-sft-v1/adapter --output-dir artifacts/adapters/red-grpo-round-1
```

`vishgym-train evaluate-red` and `vishgym-train evaluate-blue` compute metrics from actual held-out sandbox outcomes and write human-review manifests. The initial 72-example export is a pipeline smoke test; generate a larger reviewed synthetic corpus before any substantive training claim.

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
