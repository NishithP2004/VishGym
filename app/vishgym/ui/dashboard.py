"""Streamlit dashboard for a closed synthetic VishGym demonstration."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    raise SystemExit("Install vishgym[app] to run the Streamlit dashboard.")

from vishgym.arena.runner import run_local_episode
from vishgym.core.fixtures import ATTACK_CARDS


st.set_page_config(page_title="VishGym", page_icon="🛡️", layout="wide")
st.title("VishGym")
st.caption("Closed synthetic Red/Blue self-play arena. No real voices, accounts, messages, sites, or payments.")

with st.sidebar:
    selected = st.selectbox("Synthetic scenario", [item[0] for item in ATTACK_CARDS], format_func=lambda key: dict(ATTACK_CARDS)[key])
    seed = st.number_input("Seed", min_value=0, value=7)
    run = st.button("Run safe simulation", type="primary")

catalogue, simulation, policy = st.tabs(["Attack catalogue", "Simulation", "Policy manifest"])
with catalogue:
    for card_id, title in ATTACK_CARDS:
        st.markdown(f"- **{card_id}** — {title}")

with simulation:
    if run:
        state, verdict = run_local_episode(seed=int(seed), scenario_id=selected)
        left, right = st.columns(2)
        with left:
            st.subheader("Synthetic audio turns")
            for turn in state.audio_turns:
                st.write(f"{turn.speaker.value.title()} · {turn.voice_speaker} · transcript hidden from opponent")
                path = Path("artifacts/runtime/audio") / Path(turn.audio_ref).name
                if path.exists():
                    st.audio(path.read_bytes(), format="audio/wav")
        with right:
            st.subheader("Judge decision")
            st.metric("Blue reward", verdict.blue_reward)
            st.metric("Red reward", verdict.red_reward)
            st.write(f"Outcome: **{verdict.terminal_outcome}**")
            st.json(verdict.model_dump())
        st.subheader("Immutable sandbox ledger")
        st.json([event.model_dump() for event in state.ledger])
    else:
        st.info("Choose a synthetic chain and run a local safe simulation.")

with policy:
    st.code(json.dumps({"base": "google/gemma-4-E2B-it", "tts": "Qwen3-TTS-12Hz-1.7B-CustomVoice", "reward_judge": "fixed hybrid"}, indent=2), language="json")
