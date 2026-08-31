# Modal training workflow

VishGym can run its closed, synthetic-only training stages on Modal. The launcher is [`modal_vishgym.py`](../modal_vishgym.py). It mounts separate persistent Volumes for artifacts and Hugging Face cache data, and it never uploads a dataset or adapter to a public registry.

## One-time setup

1. Install the local launcher: `pip install -e '.[modal]'`.
2. Authenticate the Modal CLI with `modal setup`.
3. In the Modal dashboard, create a Secret named `vishgym-huggingface` containing one key, `HF_TOKEN`. Do not copy the local `.env` file into Modal.
4. Accept the Gemma model terms using the Hugging Face account that owns the token.

## Verify the remote runtime

```bash
modal run modal_vishgym.py --stage smoke
```

This starts an L4 container but does not download a model. It verifies CUDA, runs a safe closed-world episode, and asserts that no agent gets a transcript.

## First review-gated round

```bash
modal run modal_vishgym.py --stage export --dataset-name warm-start-qwen-v1
# One real forward/backward/save cycle before committing a larger GPU budget.
modal run modal_vishgym.py --stage warm-start --dataset-name warm-start-qwen-v1 --role red --run-name red-sft-smoke-v1 --max-steps 2
modal run modal_vishgym.py --stage warm-start --dataset-name warm-start-qwen-v1 --role red --run-name red-sft-v1
modal run modal_vishgym.py --stage grpo --role red --initial-run-name red-sft-v1 --run-name red-grpo-round-1
modal run modal_vishgym.py --stage benchmark --red-run-name red-sft-v1 --blue-run-name blue-sft-v1 --benchmark-name held-out-v1 --held-out-seeds 101,103
```

The benchmark evaluates all nine synthetic fraud cards and one legitimate control
over held-out seeds 101 and 103. It writes aggregate metrics and review manifests
to `vishgym-artifacts/benchmarks/<benchmark-name>`; temporary waveforms, hidden
transcripts, and raw completions are deleted instead of persisted.

The resulting files live only in the `vishgym-artifacts` Modal Volume. The launcher has no deployment or Hub-upload operation. Review the receipt and metrics before using an adapter as a frozen opponent for the Blue round.

The dataset-rendering image deliberately uses Qwen's supported Transformers runtime, while the Gemma training/evaluation image uses the later runtime required by Gemma 4 audio QLoRA. The benchmark keeps these runtimes in separate Modal containers and transfers only ephemeral WAV bytes between them; do not mix arbitrary Transformers versions in a notebook environment.

## Cleanup

Use the Modal dashboard or `modal volume` commands to remove obsolete runs from `vishgym-artifacts`. Keep the separate `vishgym-hf-cache` Volume only while model cache reuse is valuable.
