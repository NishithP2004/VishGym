"""Streamlit command-center frontend for VishGym."""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path
from textwrap import dedent
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError:  # pragma: no cover
    raise SystemExit("Install vishgym[app] to run the dashboard.")

from vishgym.core.fixtures import ATTACK_CARDS


API_BASE = os.environ.get("VISHGYM_MODAL_API_BASE", "http://127.0.0.1:8000").rstrip("/")


def api_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return urljoin(f"{API_BASE}/", path.lstrip("/"))


def api_json(path: str, timeout: int = 15) -> dict:
    request = Request(api_url(path), headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def event_stream(url: str, timeout: int):
    request = Request(url, headers={"Accept": "text/event-stream"})
    with urlopen(request, timeout=timeout) as response:
        event_name = ""
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                yield event_name, json.loads(line.removeprefix("data: "))


def money(paise: int | None) -> str:
    if paise is None:
        return "—"
    return f"₹{paise / 100:,.2f}"


def esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def audio_source(audio_ref: str) -> str | bytes:
    if API_BASE.startswith("http://127.0.0.1"):
        path = Path("artifacts/runtime/audio") / Path(audio_ref).name
        if path.exists():
            return path.read_bytes()
    return api_url(audio_ref)


def metric_card(label: str, value: object, accent: str = "#47a3ff") -> str:
    return dedent(f"""
    <div class="vg-metric-card" style="border-color:{accent}44">
      <div class="vg-label">{esc(label)}</div>
      <div class="vg-big-number" style="color:{accent}">{esc(value)}</div>
    </div>
    """).strip()


def sparkline_card(rows: list[dict[str, float]], key: str, label: str, color: str) -> str:
    if not rows:
        return dedent(f"""
        <div class="vg-chart-card">
          <div class="vg-label">{esc(label)}</div>
          <div class="vg-muted">Waiting for first update.</div>
        </div>
        """).strip()
    values = [float(row[key]) for row in rows]
    updates = [float(row["update"]) for row in rows]
    min_x, max_x = min(updates), max(updates)
    min_y, max_y = min(values), max(values)
    if max_x == min_x:
        min_x -= 0.5
        max_x += 0.5
    if max_y == min_y:
        padding = max(abs(max_y) * 0.15, 1e-6)
        min_y -= padding
        max_y += padding
    points = []
    for row in rows:
        x = 18 + 264 * ((float(row["update"]) - min_x) / (max_x - min_x))
        y = 112 - 86 * ((float(row[key]) - min_y) / (max_y - min_y))
        points.append(f"{x:.1f},{y:.1f}")
    latest = values[-1]
    return dedent(f"""
    <div class="vg-chart-card">
      <div class="vg-chart-head">
        <div><div class="vg-label">{esc(label)}</div><div class="vg-chart-value">{latest:.7g}</div></div>
        <div class="vg-muted">{len(rows)} update{"s" if len(rows) != 1 else ""}</div>
      </div>
      <svg viewBox="0 0 300 132" role="img" aria-label="{esc(label)} chart">
        <line x1="18" y1="112" x2="286" y2="112" stroke="rgba(148,163,184,.22)" />
        <line x1="18" y1="22" x2="18" y2="112" stroke="rgba(148,163,184,.22)" />
        <polyline points="{' '.join(points)}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
        <circle cx="{points[-1].split(',')[0]}" cy="{points[-1].split(',')[1]}" r="5" fill="{color}" />
      </svg>
      <div class="vg-chart-range">{min_y:.4g} → {max_y:.4g}</div>
    </div>
    """).strip()


def render_ledger(events: list[dict]) -> str:
    style = dedent("""
    <style>
      body {
        margin: 0;
        background: transparent;
        color: #eef2ff;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .vg-muted {color: #9ca3af; font-size: 13px;}
      .vg-tool {
        border-radius: 16px;
        padding: 12px 14px;
        border: 1px solid rgba(255,209,102,.26);
        background: rgba(120, 73, 0, .16);
        margin: 8px 0;
      }
      .vg-ledger-title {
        color: #ffd166;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .14em;
        font-weight: 900;
        margin-bottom: 8px;
      }
      strong {color: #fff;}
    </style>
    """).strip()
    if not events:
        return style + '<div class="vg-ledger-title">Recent tool ledger</div><div class="vg-muted">Waiting for tool activity.</div>'
    rendered = [style, '<div class="vg-ledger-title">Recent tool ledger</div>']
    for event in events[-4:][::-1]:
        rendered.append(
            f"""
            <div class="vg-tool">
              <strong>{esc(event.get("team", "").title())}</strong> · {esc(event.get("tool"))}<br/>
              <span class="vg-muted">Risk: {esc(event.get("risk_tag") or "none")} · Valid: {esc(event.get("valid"))}</span>
            </div>
            """
        )
    return "\n".join(rendered)


def persona_card(title: str, persona: dict | None, accent: str, icon: str) -> str:
    if not persona:
        return dedent(f"""
        <div class="vg-dossier" style="border-color:{accent}33">
          <div class="vg-dossier-head"><span>{icon}</span><strong>{esc(title)}</strong></div>
          <div class="vg-muted">Identity packet pending.</div>
        </div>
        """).strip()
    return dedent(f"""
    <div class="vg-dossier" style="border-color:{accent}55">
      <div class="vg-dossier-head"><span>{icon}</span><strong>{esc(title)}</strong></div>
      <div class="vg-person-name">{esc(persona.get("name", "—"))}</div>
      <div class="vg-muted">{esc(persona.get("occupation", "—"))} · {esc(persona.get("age_band", "—"))}</div>
      <div class="vg-kv"><span>Email</span><b>{esc(persona.get("email", "—"))}</b></div>
      <div class="vg-kv"><span>DOB</span><b>{esc(persona.get("dob", "—"))}</b></div>
      <div class="vg-kv"><span>ID ref</span><b>{esc(persona.get("identity_ref", "—"))}</b></div>
      <div class="vg-kv"><span>Voice</span><b>{esc(persona.get("voice", "—"))}</b></div>
    </div>
    """).strip()


def credentials_card(credentials: dict | None) -> str:
    if not credentials:
        return dedent("""
        <div class="vg-dossier">
          <div class="vg-dossier-head"><span>🪪</span><strong>Blue Credentials</strong></div>
          <div class="vg-muted">Credential vault pending.</div>
        </div>
        """).strip()
    rows = "\n".join(
        f'<div class="vg-kv"><span>{esc(key.upper())}</span><b>{esc(value)}</b></div>'
        for key, value in sorted(credentials.items())
    )
    return dedent(f"""
    <div class="vg-dossier">
      <div class="vg-dossier-head"><span>🪪</span><strong>Blue Credentials</strong></div>
      {rows}
    </div>
    """).strip()


def training_turn_card(turn: dict) -> str:
    speaker = str(turn.get("speaker", "agent")).title()
    css_class = "red" if turn.get("speaker") == "red" else "blue"
    icon = "☎️" if turn.get("speaker") == "red" else "🛡️"
    tool = turn.get("tool_event") or {}
    tool_name = tool.get("tool") or "listening"
    valid = tool.get("valid")
    return dedent(f"""
    <div class="vg-turn {css_class}">
      <div class="vg-turn-head">
        <span>{icon} {esc(speaker)} · Training episode {esc(turn.get("sample_index", "—"))}/{esc(turn.get("group_size", "—"))} · Turn {esc(turn.get("turn_number", "—"))}</span>
        <span>update {esc(turn.get("update", "—"))}/{esc(turn.get("updates", "—"))}</span>
      </div>
      <div style="margin-top:8px; color:#e2e8f0;">{esc(turn.get("spoken_text", ""))}</div>
      <div class="vg-muted" style="margin-top:8px;">Tool: {esc(tool_name)}{f" · valid={esc(valid)}" if valid is not None else ""}</div>
    </div>
    """).strip()


def audio_queue_player(audio_items: list[dict], *, run_id: str = "pending", autoplay: bool = False) -> str:
    playlist = json.dumps(audio_items)
    state_key = json.dumps(f"vishgym-audio-queue-{run_id}")
    autoplay_js = "true" if autoplay else "false"
    return dedent(f"""
    <style>
      body {{
        margin: 0;
        background: transparent;
        color: #eef2ff;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      .vg-player-shell {{
        border: 1px solid rgba(71,163,255,.26);
        border-radius: 20px;
        padding: 14px 16px;
        background: linear-gradient(135deg, rgba(15,23,42,.92), rgba(2,6,23,.82));
        box-shadow: inset 0 0 30px rgba(37,99,235,.10);
      }}
      .vg-player-title {{
        font-size: 11px;
        color: #93c5fd;
        text-transform: uppercase;
        letter-spacing: .14em;
        font-weight: 900;
      }}
      .vg-muted {{color: #9ca3af; font-size: 13px; margin-top: 6px;}}
      audio {{width: 100%; margin-top: 10px; accent-color: #47a3ff;}}
      .vg-playlist {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 8px;
        margin-top: 10px;
      }}
      .vg-play-item {{
        border: 1px solid rgba(148,163,184,.16);
        border-radius: 12px;
        padding: 8px;
        color: #94a3b8;
        font-size: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .vg-play-item.active {{
        border-color: rgba(71,163,255,.55);
        color: #e0f2fe;
        background: rgba(37,99,235,.18);
      }}
    </style>
    <div class="vg-player-shell">
      <div class="vg-player-title">Sequential call playback</div>
      <div id="vgNowPlaying" class="vg-muted">Waiting for audio turns.</div>
      <audio id="vgCallAudio" controls></audio>
      <div id="vgPlaylist" class="vg-playlist"></div>
    </div>
    <script>
      const playlist = {playlist};
      const stateKey = {state_key};
      const audio = document.getElementById("vgCallAudio");
      const now = document.getElementById("vgNowPlaying");
      const list = document.getElementById("vgPlaylist");
      const savedIndex = Number(window.sessionStorage.getItem(stateKey + ":index") || "0");
      const previousLength = Number(window.sessionStorage.getItem(stateKey + ":length") || "0");
      let index = Math.max(0, Math.min(savedIndex, Math.max(playlist.length - 1, 0)));
      function renderList() {{
        list.innerHTML = playlist.map((item, i) =>
          `<div class="vg-play-item ${{i === index ? "active" : ""}}">${{i + 1}}. ${{item.speaker}} · ${{item.voice}}</div>`
        ).join("");
      }}
      function load(i, play) {{
        if (!playlist.length) {{ renderList(); return; }}
        index = Math.max(0, Math.min(i, playlist.length - 1));
        window.sessionStorage.setItem(stateKey + ":index", String(index));
        audio.src = playlist[index].url;
        now.textContent = `${{playlist[index].speaker}} turn ${{playlist[index].turn}} · ${{playlist[index].voice}}`;
        renderList();
        if (play) audio.play().catch(() => {{}});
      }}
      audio.addEventListener("ended", () => {{
        if (index + 1 < playlist.length) load(index + 1, true);
      }});
      window.sessionStorage.setItem(stateKey + ":length", String(playlist.length));
      load(index, {autoplay_js} && previousLength === 0);
    </script>
    """).strip()


st.set_page_config(page_title="VishGym", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    :root {
      --vg-bg: #070a12;
      --vg-panel: rgba(17, 24, 39, .84);
      --vg-line: rgba(148, 163, 184, .18);
      --vg-red: #ff3b5c;
      --vg-blue: #47a3ff;
      --vg-gold: #ffd166;
      --vg-green: #4ade80;
    }
    .stApp {
      background:
        radial-gradient(circle at top left, rgba(71,163,255,.18), transparent 30%),
        radial-gradient(circle at 80% 10%, rgba(255,59,92,.18), transparent 28%),
        linear-gradient(180deg, #070a12 0%, #0d1220 50%, #090b10 100%);
      color: #eef2ff;
    }
    .main .block-container {padding-top: 1rem; max-width: 1320px;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #0b1020, #070a12);}
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {color: #e5e7eb;}
    .vg-hero {
      border: 1px solid var(--vg-line);
      border-radius: 26px;
      padding: 20px 24px;
      background: linear-gradient(135deg, rgba(15,23,42,.94), rgba(30,41,59,.76));
      box-shadow: 0 24px 80px rgba(0,0,0,.35);
      position: relative;
      overflow: hidden;
    }
    .vg-hero:after {
      content: "";
      position: absolute;
      width: 340px; height: 340px; border-radius: 999px;
      right: -90px; top: -160px;
      background: radial-gradient(circle, rgba(71,163,255,.28), transparent 68%);
    }
    .vg-eyebrow {color: #93c5fd; text-transform: uppercase; letter-spacing: .18em; font-size: .72rem;}
    .vg-hero h1 {font-size: 2.35rem; line-height: 1; margin: 8px 0 10px; color: white;}
    .vg-hero p {font-size: 1.02rem; color: #cbd5e1; max-width: 840px;}
    .vg-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 18px 0;}
    .vg-card {
      border: 1px solid var(--vg-line);
      border-radius: 20px;
      padding: 16px;
      background: rgba(15,23,42,.72);
      box-shadow: 0 16px 40px rgba(0,0,0,.18);
    }
    .vg-label {font-size: .72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .12em;}
    .vg-value {font-size: 1.02rem; color: #f8fafc; margin-top: 5px; font-weight: 700;}
    .vg-agent {
      min-height: 168px;
      border-radius: 24px;
      padding: 18px;
      border: 1px solid var(--vg-line);
      background: linear-gradient(160deg, rgba(15,23,42,.92), rgba(2,6,23,.84));
      position: relative;
      overflow: hidden;
    }
    .vg-agent.red {box-shadow: inset 0 0 0 1px rgba(255,59,92,.18), 0 0 60px rgba(255,59,92,.08);}
    .vg-agent.blue {box-shadow: inset 0 0 0 1px rgba(71,163,255,.18), 0 0 60px rgba(71,163,255,.08);}
    .vg-avatar {
      width: 54px; height: 54px; border-radius: 18px;
      display: inline-flex; align-items: center; justify-content: center;
      font-size: 1.6rem; margin-right: 12px; vertical-align: middle;
      background: rgba(255,255,255,.08);
    }
    .vg-agent-title {font-size: 1.2rem; font-weight: 800; color: white;}
    .vg-agent-sub {color: #aab3c5; font-size: .86rem; margin-top: 4px;}
    .vg-wave {
      height: 42px; margin-top: 20px; border-radius: 999px;
      background:
        repeating-linear-gradient(90deg, rgba(255,255,255,.18) 0 4px, transparent 4px 12px);
      mask-image: linear-gradient(90deg, transparent, black 14%, black 86%, transparent);
      opacity: .7;
    }
    .vg-signal {
      border: 1px solid var(--vg-line);
      border-radius: 24px;
      padding: 18px;
      background: rgba(2,6,23,.74);
    }
    .vg-pulse {
      height: 8px; border-radius: 999px; margin: 16px 0 8px;
      background: linear-gradient(90deg, var(--vg-red), var(--vg-gold), var(--vg-blue));
      box-shadow: 0 0 30px rgba(71,163,255,.28);
      animation: vgPulse 1.3s ease-in-out infinite alternate;
    }
    @keyframes vgPulse { from {filter: brightness(.8); transform: scaleX(.98);} to {filter: brightness(1.22); transform: scaleX(1);} }
    .vg-turn {
      border: 1px solid var(--vg-line);
      border-left: 5px solid var(--vg-blue);
      border-radius: 20px;
      padding: 14px 16px;
      background: rgba(15,23,42,.78);
      margin: 12px 0;
    }
    .vg-turn.red {border-left-color: var(--vg-red);}
    .vg-turn.blue {border-left-color: var(--vg-blue);}
    .vg-turn-head {display: flex; justify-content: space-between; gap: 12px; color: #f8fafc; font-weight: 800;}
    .vg-muted {color: #9ca3af; font-size: .85rem;}
    .vg-tool {
      border-radius: 16px;
      padding: 12px 14px;
      border: 1px solid rgba(255,209,102,.26);
      background: rgba(120, 73, 0, .16);
      margin: 8px 0;
    }
    .vg-ledger-title {color: #ffd166; font-size: .78rem; text-transform: uppercase; letter-spacing: .14em;}
    .vg-outcome {
      border-radius: 22px;
      padding: 18px;
      background: linear-gradient(135deg, rgba(34,197,94,.2), rgba(37,99,235,.16));
      border: 1px solid rgba(74,222,128,.24);
    }
    .vg-metric-card {
      border: 1px solid var(--vg-line);
      border-radius: 18px;
      padding: 14px;
      background: rgba(2,6,23,.68);
      min-height: 92px;
    }
    .vg-big-number {font-size: 1.55rem; line-height: 1.2; font-weight: 900; margin-top: 8px;}
    .vg-chart-card {
      border: 1px solid var(--vg-line);
      border-radius: 18px;
      padding: 14px;
      background: rgba(2,6,23,.78);
      box-shadow: inset 0 0 28px rgba(15,23,42,.65);
      margin-top: 12px;
    }
    .vg-chart-head {display:flex; align-items:flex-start; justify-content:space-between; gap:12px;}
    .vg-chart-value {font-size: 1.1rem; color: #f8fafc; font-weight: 850; margin-top: 4px;}
    .vg-chart-card svg {width:100%; height:auto; margin-top:10px; display:block;}
    .vg-chart-range {font-size:.75rem; color:#94a3b8; text-align:right;}
    .vg-train-log {
      max-height: 380px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .vg-run-output {
      border: 1px solid rgba(74,222,128,.24);
      border-radius: 18px;
      padding: 14px 16px;
      background: rgba(20,83,45,.18);
      color: #bbf7d0;
      margin-top: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: .82rem;
      overflow-wrap: anywhere;
    }
    .vg-dossier-grid {display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:14px; margin: 16px 0 20px;}
    .vg-dossier {
      border: 1px solid var(--vg-line);
      border-radius: 20px;
      padding: 14px 16px;
      background: rgba(2,6,23,.72);
      min-height: 172px;
    }
    .vg-dossier-head {display:flex; gap:10px; align-items:center; color:#f8fafc; margin-bottom:10px;}
    .vg-person-name {font-size:1.05rem; font-weight:900; color:white; margin-bottom:4px;}
    .vg-kv {display:flex; justify-content:space-between; gap:12px; margin-top:8px; font-size:.78rem;}
    .vg-kv span {color:#94a3b8;}
    .vg-kv b {color:#e2e8f0; text-align:right; font-weight:750; overflow-wrap:anywhere;}
    .vg-player-shell {
      border: 1px solid rgba(71,163,255,.26);
      border-radius: 20px;
      padding: 14px 16px;
      margin: 12px 0;
      background: linear-gradient(135deg, rgba(15,23,42,.86), rgba(2,6,23,.78));
    }
    .vg-player-title {font-size:.78rem; color:#93c5fd; text-transform:uppercase; letter-spacing:.14em; font-weight:900;}
    .vg-playlist {display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:8px; margin-top:10px;}
    .vg-play-item {border:1px solid rgba(148,163,184,.16); border-radius:12px; padding:8px; color:#94a3b8; font-size:.76rem;}
    .vg-play-item.active {border-color:rgba(71,163,255,.55); color:#e0f2fe; background:rgba(37,99,235,.18);}
    @media (max-width: 900px) {.vg-hero h1 {font-size: 2rem;}.vg-hero {padding: 20px;}}
    @media (max-width: 1100px) {.vg-dossier-grid {grid-template-columns:1fr;}}
    </style>
    """,
    unsafe_allow_html=True,
)


scenario_titles = dict(ATTACK_CARDS)
try:
    voice_manifest = api_json("/api/v1/voices")
except Exception:
    voice_manifest = {"speakers": [], "tone_examples": []}
speaker_ids = [item["id"] for item in voice_manifest.get("speakers", [])] or ["Ryan", "Aiden", "Vivian", "Serena", "Ono_Anna", "Sohee"]
if "vishgym_seed_default" not in st.session_state:
    st.session_state["vishgym_seed_default"] = secrets.randbelow(2_000_000_000)
if "vishgym_red_voice_default" not in st.session_state:
    st.session_state["vishgym_red_voice_default"] = secrets.randbelow(len(speaker_ids))
if "vishgym_blue_voice_default" not in st.session_state:
    offset = 1 + secrets.randbelow(max(len(speaker_ids) - 1, 1))
    st.session_state["vishgym_blue_voice_default"] = (st.session_state["vishgym_red_voice_default"] + offset) % len(speaker_ids)

with st.sidebar:
    st.subheader("Director Controls")
    selected = st.selectbox("Scenario", [item[0] for item in ATTACK_CARDS], format_func=lambda key: scenario_titles[key])
    difficulty = st.slider("Difficulty", min_value=1, max_value=3, value=2)
    randomize_episode = st.checkbox("Random personas and voices for each call", value=True)
    seed = st.number_input("Seed", min_value=0, value=st.session_state["vishgym_seed_default"], disabled=randomize_episode)
    runtime_mode = st.selectbox("Runtime", ["full", "auto"])
    red_voice = st.selectbox(
        "Red voice",
        speaker_ids,
        index=st.session_state["vishgym_red_voice_default"],
        format_func=lambda key: key.replace("_", " "),
        disabled=randomize_episode,
    )
    blue_voice = st.selectbox(
        "Blue voice",
        speaker_ids,
        index=st.session_state["vishgym_blue_voice_default"],
        format_func=lambda key: key.replace("_", " "),
        disabled=randomize_episode,
    )
    red_tone = st.text_input("Red tone", value="urgent but controlled", max_chars=160, disabled=randomize_episode)
    blue_tone = st.text_input("Blue tone", value="skeptical, concise, careful", max_chars=160, disabled=randomize_episode)
    noise_level = st.slider("Line noise", min_value=0.0, max_value=1.0, value=0.05, step=0.05)
    red_temperature = st.slider("Red creativity", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    blue_temperature = st.slider("Blue creativity", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    st.caption(f"Backend: {API_BASE}")
    run_live = st.button("Open the line", type="primary", use_container_width=True)

try:
    manifest = api_json("/api/v1/model")
except Exception as exc:
    manifest = {"selected_mode": "offline", "full_runtime_ready": False, "reasons": [str(exc)]}

st.markdown(
    dedent(f"""
    <div class="vg-hero">
      <div class="vg-eyebrow">Live adversarial payment call</div>
      <h1>VishGym Control Room</h1>
      <p>Watch two audio-first agents negotiate a high-stakes payment interaction while their tool actions, wallet state, and final reward unfold in real time.</p>
    </div>
    <div class="vg-grid">
      <div class="vg-card"><div class="vg-label">Compute</div><div class="vg-value">{esc("Modal" if "modal" in API_BASE else "Local")}</div></div>
      <div class="vg-card"><div class="vg-label">Runtime</div><div class="vg-value">{esc(manifest.get("selected_mode", "unknown"))}</div></div>
      <div class="vg-card"><div class="vg-label">Scenario</div><div class="vg-value">{esc(scenario_titles[selected])}</div></div>
      <div class="vg-card"><div class="vg-label">Identity mode</div><div class="vg-value">{esc("Random per call" if randomize_episode else "Fixed seed")}</div></div>
    </div>
    """).strip(),
    unsafe_allow_html=True,
)

agent_left, signal_mid, agent_right = st.columns([1, .8, 1], gap="large")
with agent_left:
    st.markdown(
        dedent(f"""
        <div class="vg-agent red">
          <div><span class="vg-avatar">☎️</span><span class="vg-agent-title">Red Agent</span></div>
          <div class="vg-agent-sub">Voice: {esc("random built-in voice" if randomize_episode else red_voice.replace("_", " "))}</div>
          <div class="vg-agent-sub">Tone: {esc("persona generated at call start" if randomize_episode else red_tone)}</div>
          <div class="vg-wave"></div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )
with signal_mid:
    st.markdown(
        dedent(f"""
        <div class="vg-signal">
          <div class="vg-label">Connection</div>
          <div class="vg-value">{esc(scenario_titles[selected])}</div>
          <div class="vg-pulse"></div>
          <div class="vg-muted">{esc("Fresh seed" if randomize_episode else f"Seed {int(seed)}")} · Difficulty {difficulty} · Noise {noise_level:.2f}</div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )
with agent_right:
    st.markdown(
        dedent(f"""
        <div class="vg-agent blue">
          <div><span class="vg-avatar">🛡️</span><span class="vg-agent-title">Blue Agent</span></div>
          <div class="vg-agent-sub">Voice: {esc("random built-in voice" if randomize_episode else blue_voice.replace("_", " "))}</div>
          <div class="vg-agent-sub">Tone: {esc("persona generated at call start" if randomize_episode else blue_tone)}</div>
          <div class="vg-wave"></div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

persona_box = st.empty()

main_col, side_col = st.columns([1.55, .95], gap="large")
with main_col:
    st.subheader("Call Timeline")
    call_status = st.empty()
    player_label_box = st.empty()
    player_box = st.empty()
    turn_stack = st.container()
with side_col:
    st.subheader("Action Board")
    wallet_box = st.empty()
    loading_box = st.empty()
    tool_box = st.empty()
    judge_box = st.empty()
    with st.expander("Runtime manifest", expanded=False):
        st.json(manifest)

if not run_live:
    call_status.info("Open the line to begin.")
    persona_box.markdown(
        dedent(f"""
        <div class="vg-dossier-grid">
          {persona_card("Red Persona", None, "#ff3b5c", "☎️")}
          {persona_card("Blue Persona", None, "#47a3ff", "🛡️")}
          {credentials_card(None)}
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )
    wallet_box.metric("Wallet balance", "Waiting")
    loading_box.markdown('<div class="vg-muted">No active call.</div>', unsafe_allow_html=True)
    player_label_box.empty()
    player_box.empty()
else:
    params = {
        "chain": selected,
        "difficulty": difficulty,
        "pace_ms": 750,
        "mode": runtime_mode,
        "noise_level": noise_level,
        "red_temperature": red_temperature,
        "blue_temperature": blue_temperature,
    }
    if not randomize_episode:
        params.update(
            {
                "seed": int(seed),
                "red_voice": red_voice,
                "blue_voice": blue_voice,
                "red_tone": red_tone,
                "blue_tone": blue_tone,
            }
        )
    query = urlencode(params)
    stream_url = api_url(f"/api/v1/live-episodes/stream?{query}")
    ledger: list[dict] = []
    audio_queue: list[dict] = []
    episode_context: dict | None = None
    wallet_balance: int | None = None
    try:
        for event_name, payload in event_stream(stream_url, timeout=900):
            if event_name == "starting":
                call_status.info("Dialing agents and preparing the line...")
                loading_box.markdown('<div class="vg-pulse"></div><div class="vg-muted">Preparing call environment.</div>', unsafe_allow_html=True)
            elif event_name == "loading":
                loading_box.markdown(
                    f'<div class="vg-pulse"></div><div class="vg-muted">Loading models · tick {esc(payload.get("tick", 0))}</div>',
                    unsafe_allow_html=True,
                )
            elif event_name == "started":
                call_status.success(f"Line open · run {payload['run_id'][:8]} · seed {payload.get('seed', 'random')}")
                loading_box.markdown('<div class="vg-muted">Agents are now exchanging audio turns.</div>', unsafe_allow_html=True)
                episode_context = payload.get("episode_context")
                if episode_context:
                    persona_box.markdown(
                        dedent(f"""
                        <div class="vg-dossier-grid">
                          {persona_card("Red Persona", episode_context.get("red"), "#ff3b5c", "☎️")}
                          {persona_card("Blue Persona", episode_context.get("blue"), "#47a3ff", "🛡️")}
                          {credentials_card(episode_context.get("blue_credentials"))}
                        </div>
                        """).strip(),
                        unsafe_allow_html=True,
                    )
            elif event_name == "turn":
                wallet_balance = payload.get("wallet_balance_paise", wallet_balance)
                if payload.get("tool_event"):
                    ledger.append(payload["tool_event"])
                with turn_stack:
                    audio = payload["audio_turn"]
                    speaker = payload["speaker"]
                    audio_queue.append(
                        {
                            "url": api_url(audio["audio_ref"]),
                            "speaker": speaker.title(),
                            "turn": payload["turn_number"],
                            "voice": audio.get("voice_speaker", "voice"),
                        }
                    )
                    player_label_box.markdown(
                        '<div class="vg-player-title">Call Playback Queue</div>',
                        unsafe_allow_html=True,
                    )
                    with player_box:
                        components.html(
                            audio_queue_player(audio_queue, run_id=payload["run_id"], autoplay=len(audio_queue) == 1),
                            height=190,
                        )
                    css_class = "red" if speaker == "red" else "blue"
                    icon = "☎️" if speaker == "red" else "🛡️"
                    tool = payload.get("tool_event")
                    st.markdown(
                        dedent(f"""
                        <div class="vg-turn {css_class}">
                          <div class="vg-turn-head">
                            <span>{icon} {esc(speaker.title())} Agent · Turn {esc(payload['turn_number'])}</span>
                            <span>{esc(audio.get('voice_speaker', 'voice'))}</span>
                          </div>
                          <div class="vg-muted">Actual synthetic message sent</div>
                          <div style="margin-top:8px; color:#f8fafc; font-size:1rem; line-height:1.45;">{esc(payload.get('message') or audio.get('message', ''))}</div>
                          <div class="vg-muted" style="margin-top:8px;">Opponent receives waveform and tool observations; agents do not receive text transcripts.</div>
                          {f'<div class="vg-tool"><div class="vg-ledger-title">Tool action</div>{esc(tool["tool"])} · valid={esc(tool["valid"])}</div>' if tool else ''}
                        </div>
                        """).strip(),
                        unsafe_allow_html=True,
                    )
                wallet_box.metric("Wallet balance", money(wallet_balance))
                with tool_box:
                    components.html(render_ledger(ledger), height=360, scrolling=True)
            elif event_name == "completed":
                judge = payload["judge"]
                judge_box.markdown(
                    dedent(f"""
                    <div class="vg-outcome">
                      <div class="vg-label">Final verdict</div>
                      <div class="vg-value">{esc(payload['outcome'])}</div>
                      <div class="vg-muted">Blue reward {esc(judge['blue_reward'])} · Red reward {esc(judge['red_reward'])}</div>
                    </div>
                    """).strip(),
                    unsafe_allow_html=True,
                )
                st.json(judge)
            elif event_name == "error":
                call_status.error(payload.get("message", "The live call stopped."))
                if payload.get("detail"):
                    st.code(payload["detail"])
    except Exception as exc:
        call_status.error(f"Live call stopped: {exc}")
        if manifest.get("reasons"):
            st.json(manifest)

st.divider()
st.subheader("RL Training Console")
st.markdown(
    dedent("""
    <div class="vg-signal">
      <div class="vg-label">Operator mode</div>
      <div class="vg-value">Launch a policy update round from the same control room.</div>
      <div class="vg-muted">Reward, loss, and learning-rate curves update as backend training events arrive.</div>
    </div>
    """).strip(),
    unsafe_allow_html=True,
)

train_left, train_right = st.columns([.9, 1.4], gap="large")
with train_left:
    train_role = st.selectbox("Training target", ["blue", "red"], index=0)
    default_initial = "blue-init-v1" if train_role == "blue" else "red-init-v1"
    default_opponent = "red-init-v1" if train_role == "blue" else "blue-init-v1"
    initial_run_name = st.text_input("Initial adapter run", value=default_initial, max_chars=80)
    opponent_run_name = st.text_input("Opponent adapter run", value=default_opponent, max_chars=80)
    target_run_name = st.text_input("Output run", value=f"{train_role}-live-rl-v1", max_chars=80)
    train_updates = st.slider("Updates", min_value=1, max_value=12, value=1)
    train_group_size = st.slider("Group size", min_value=2, max_value=4, value=2)
    train_lr = st.number_input("Learning rate", min_value=1e-7, max_value=1e-3, value=5e-6, format="%.7f")
    train_temp = st.slider("Sampling temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    run_training = st.button("Start RL run", type="primary", use_container_width=True)
with train_right:
    train_status = st.empty()
    metric_cols = st.columns(3)
    reward_col, loss_col, lr_col = st.columns(3)
    reward_metric = metric_cols[0].empty()
    loss_metric = metric_cols[1].empty()
    lr_metric = metric_cols[2].empty()
    reward_chart = reward_col.empty()
    loss_chart = loss_col.empty()
    lr_chart = lr_col.empty()
    st.markdown("#### Training Episode Stream")
    train_conversation = st.empty()
    train_events = st.empty()
    train_output = st.empty()

if run_training:
    query = urlencode(
        {
            "role": train_role,
            "initial_run_name": initial_run_name,
            "opponent_run_name": opponent_run_name,
            "run_name": target_run_name,
            "updates": train_updates,
            "group_size": train_group_size,
            "learning_rate": train_lr,
            "temperature": train_temp,
        }
    )
    stream_url = api_url(f"/api/v1/training/stream?{query}")
    curve_rows: list[dict[str, float]] = []
    event_rows: list[dict] = []
    training_turns: list[dict] = []
    try:
        for event_name, payload in event_stream(stream_url, timeout=7200):
            if event_name == "training_started":
                train_status.info(f"Training run started · {payload['role']} · {payload['run_name']}")
            elif event_name == "training_heartbeat":
                train_status.info(f"Training worker active · tick {payload.get('tick', 0)}")
            elif event_name == "preflight":
                train_status.info("Preflight complete. Preparing policy rollouts.")
            elif event_name == "ready":
                train_status.success(f"Models ready · training {payload['role']} against {payload['opponent']}")
            elif event_name == "update_started":
                train_status.info(f"Update {payload['update']}/{payload['updates']} · {payload['scenario']}")
            elif event_name == "training_episode_turn":
                training_turns.append(payload)
                train_conversation.markdown(
                    '<div class="vg-train-log">' +
                    "\n".join(training_turn_card(item) for item in training_turns[-8:]) +
                    "</div>",
                    unsafe_allow_html=True,
                )
            elif event_name == "update_completed":
                row = {
                    "update": float(payload["update"]),
                    "reward": float(payload["reward"]),
                    "loss": float(payload["loss"]),
                    "learning_rate": float(payload["learning_rate"]),
                }
                curve_rows.append(row)
                reward_metric.markdown(metric_card("Latest reward", f"{row['reward']:.4f}", "#4ade80"), unsafe_allow_html=True)
                loss_metric.markdown(metric_card("Latest loss", f"{row['loss']:.4f}", "#ffb703"), unsafe_allow_html=True)
                lr_metric.markdown(metric_card("Learning rate", f"{row['learning_rate']:.7f}", "#47a3ff"), unsafe_allow_html=True)
                reward_chart.markdown(sparkline_card(curve_rows, "reward", "Reward", "#4ade80"), unsafe_allow_html=True)
                loss_chart.markdown(sparkline_card(curve_rows, "loss", "Loss", "#ffb703"), unsafe_allow_html=True)
                lr_chart.markdown(sparkline_card(curve_rows, "learning_rate", "Learning rate", "#47a3ff"), unsafe_allow_html=True)
                event_rows.append(payload)
                train_events.markdown(
                    '<div class="vg-train-log">' +
                    "\n".join(
                        f"""
                        <div class="vg-tool">
                          <div class="vg-ledger-title">Update complete</div>
                          Update {esc(item['update'])}/{esc(item['updates'])} · {esc(item['scenario'])}<br/>
                          <span class="vg-muted">Reward {esc(item['reward'])} · Loss {esc(item['loss'])} · LR {esc(item['learning_rate'])}</span>
                        </div>
                        """
                        for item in event_rows[-6:][::-1]
                    ) +
                    "</div>",
                    unsafe_allow_html=True,
                )
            elif event_name == "completed":
                train_status.success(f"GRPO complete · mean reward {payload.get('mean_reward')} · mean loss {payload.get('mean_loss')}")
            elif event_name == "training_finished":
                train_status.success(f"Adapter written · {payload.get('adapter_path')}")
                train_output.markdown(
                    dedent(f"""
                    <div class="vg-run-output">
                      adapter: {esc(payload.get('adapter_path'))}<br/>
                      receipt: {esc(payload.get('receipt_path'))}<br/>
                      updates: {esc(payload.get('updates'))} · mean reward: {esc(payload.get('mean_reward'))} · mean loss: {esc(payload.get('mean_loss'))}
                    </div>
                    """).strip(),
                    unsafe_allow_html=True,
                )
            elif event_name == "training_error":
                train_status.error(payload.get("message", "Training stopped."))
                break
    except Exception as exc:
        train_status.error(f"Training stream stopped: {exc}")
