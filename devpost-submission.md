# VishGym

## One-line summary

VishGym is a closed Red/Blue self-play arena that identifies, simulates, and defends against synthetic GenAI-powered payment-fraud attacks using rotating personas, audio-first calls, sandbox tools, and gated RL policy improvement.

## Problem

Generative AI lowers the cost of payment fraud across vishing, smishing, WhatsApp, support impersonation, invoice diversion, refund manipulation, QR redirection, account recovery, and cross-channel pressure. Payment defenders need realistic attack simulation and measurable defensive policies without exposing real customers, real payment rails, or real identities to risk.

## Solution

VishGym implements the full Identify, Generate, Defend loop:

- Identify: nine payment-fraud attack cards spanning phone, SMS, WhatsApp, email, support, merchant, invoice, QR, refund, and cross-channel surfaces.
- Generate: Red and Blue agents run synthetic audio-first episodes with fresh personas, pseudo-identities, inboxes, wallet state, sandbox browser pages, and tool ledgers.
- Defend: a Blue policy uses verification, reporting, refusal, and sandbox-only tools; a fixed hybrid judge produces terminal rewards and evaluation metrics.

## Key Features

- Actual synthetic agent messages are displayed in the web prototype for judge review.
- Agents still receive only opponent audio and tool observations, preserving the audio-first setup.
- Each live call can randomize seed, persona pair, pseudo-identity packet, built-in voice, and tone profile.
- The reward function penalizes unsafe payments, credentials exposure, invalid tools, sandbox-boundary violations, and false positives on legitimate controls.
- Blue policy promotion requires all nine fraud cards, at least two held-out seeds, F1 >= 0.80, legitimate false-block rate <= 10%, valid tool-call rate >= 98%, and zero boundary violations.

## Architecture Summary

- `app/vishgym/arena`: synthetic closed world, reward judge, action schemas, OpenEnv-style adapter.
- `app/vishgym/core`: personas, pseudo-identities, prompts, model harnesses.
- `app/vishgym/training`: warm-start dataset export, QLoRA, GRPO, evaluation, review manifests.
- `app/vishgym/api`: FastAPI episode, streaming, runtime, and training endpoints.
- `app/vishgym/ui`: Streamlit working prototype control room.
- `deploy`: Docker/Hugging Face Space deployment assets.
- `docs/VishGym_Solution_Walkthrough.docx`: solution walkthrough.

## Testing Instructions

```bash
pip install -e '.[app,training]'
PYTHONPATH=app python -m unittest discover -s app/tests -v
uvicorn vishgym.api.main:app --host 0.0.0.0 --port 8000
streamlit run app/vishgym/ui/dashboard.py
```

## Required Submission Artifacts

- Code Repository: ready locally; fill in the public GitHub URL after pushing.
- Solution Walkthrough: `docs/VishGym_Solution_Walkthrough.docx`.
- Working Prototype (Web): runnable Streamlit/FastAPI prototype; fill in the public demo URL after deployment.

## Official URLs to Paste

- GitHub repository URL: paste after the repository is pushed.
- Public web prototype URL: paste after the web prototype is deployed.
- Demo video URL, if the form asks for one: paste after recording/upload.

## Readiness Notes

The repository covers all three Mastercard pillars and has a presentable runnable prototype. The remaining submission blockers are external to the codebase: publish the repository URL, deploy the web prototype URL, and upload the `.docx` walkthrough from the Writeups section before the August 31 deadline.
