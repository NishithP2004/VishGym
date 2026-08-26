from __future__ import annotations

import random
import uuid
from typing import Any

from vishgym.arena.audio import AudioRenderer, SyntheticAudioRenderer
from vishgym.arena.judge import HybridJudge
from vishgym.arena.models import (
    AgentAction,
    AgentObservation,
    EpisodePhase,
    EpisodeState,
    InboxMessage,
    Portal,
    StepResult,
    Team,
    ToolEvent,
)
from vishgym.core.fixtures import ATTACK_CARDS, LEGITIMATE_CONTROL_SCENARIOS, inbox, persona, search_documents


class VishGymEnv:
    """Closed synthetic environment with OpenEnv-compatible reset/step/state semantics."""

    BLUE_TOOLS = {
        "inbox.read",
        "inbox.report_block",
        "wallet.balance",
        "wallet.pay",
        "wallet.decline",
        "credentials.view",
        "browser.open",
        "browser.fill",
        "browser.submit",
        "search.query",
    }
    RED_TOOLS = {"message.send", "portal.create_template", "search.query"}

    def __init__(self, audio_renderer: AudioRenderer | None = None, max_turns: int = 6):
        self.audio_renderer = audio_renderer or SyntheticAudioRenderer()
        self.max_turns = max_turns
        self.judge = HybridJudge()
        self._state: EpisodeState | None = None

    @property
    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("Call reset before reading environment state.")
        return self._state

    def reset(self, seed: int = 7, scenario_id: str = "vishing_collect", difficulty: int = 2) -> AgentObservation:
        if scenario_id not in {card[0] for card in ATTACK_CARDS} | LEGITIMATE_CONTROL_SCENARIOS:
            raise ValueError(f"Unknown scenario_id: {scenario_id}")
        if difficulty not in {1, 2, 3}:
            raise ValueError("difficulty must be 1, 2, or 3")
        rng = random.Random(seed)
        blue = persona(Team.BLUE, rng)
        red = persona(Team.RED, rng)
        self._state = EpisodeState(
            episode_id=uuid.uuid4().hex,
            seed=seed,
            scenario_id=scenario_id,
            difficulty=difficulty,
            phase=EpisodePhase.ACTIVE,
            active_team=Team.RED,
            red_persona=red,
            blue_persona=blue,
            wallet={"balance_paise": 850_000},
            inbox=inbox(legitimate=scenario_id in LEGITIMATE_CONTROL_SCENARIOS),
            credentials={
                "aadhaar": "SYNTHETIC-ID-ONLY",
                "pan": "SYNTHETIC-TAX-ID",
                "passport": "SYNTHETIC-PASSPORT-ID",
            },
            search_documents=search_documents(),
        )
        return self._observation(Team.RED)

    def step(self, team: Team, action: AgentAction) -> StepResult:
        state = self.state
        if state.phase is not EpisodePhase.ACTIVE:
            raise RuntimeError("Episode is already terminal.")
        if team is not state.active_team:
            raise ValueError(f"Expected action from {state.active_team.value}, received {team.value}")
        persona = state.red_persona if team is Team.RED else state.blue_persona
        audio_turn = self.audio_renderer.render(team, persona, action.spoken_text)
        state.audio_turns.append(audio_turn)
        state.transcript.append({"speaker": team.value, "text": action.spoken_text})

        event = self._apply_tool(team, action.tool_call.name, action.tool_call.arguments) if action.tool_call else None
        state.turn_number += 1
        done = state.turn_number >= self.max_turns or (event is not None and event.tool in {"wallet.pay", "wallet.decline"})
        if done:
            state.phase = EpisodePhase.TERMINAL
            judge_result = self.judge.score(state)
            return StepResult(observation=None, audio_turn=audio_turn, tool_event=event, done=True, judge_result=judge_result)

        state.active_team = Team.BLUE if team is Team.RED else Team.RED
        return StepResult(
            observation=self._observation(state.active_team),
            audio_turn=audio_turn,
            tool_event=event,
            done=False,
        )

    def _observation(self, team: Team) -> AgentObservation:
        state = self.state
        opponent_audio = state.audio_turns[-1].audio_ref if state.audio_turns else None
        own_tools = {
            "wallet_balance_paise": state.wallet.balance_paise,
            "inbox_count": len(state.inbox),
            "portal_count": len(state.portals),
            "scenario_id": state.scenario_id,
        }
        tools = sorted(self.RED_TOOLS if team is Team.RED else self.BLUE_TOOLS)
        return AgentObservation(
            episode_id=state.episode_id,
            team=team,
            turn_number=state.turn_number,
            phase=state.phase,
            opponent_audio_ref=opponent_audio,
            own_tools=own_tools,
            available_tools=tools,
        )

    def _apply_tool(self, team: Team, name: str, arguments: dict[str, Any]) -> ToolEvent:
        allowed = self.RED_TOOLS if team is Team.RED else self.BLUE_TOOLS
        if name not in allowed:
            event = self._event(team, name, arguments, {"error": "tool_not_allowed"}, False, "sandbox_boundary")
            self.state.ledger.append(event)
            return event
        try:
            result, risk = self._dispatch(team, name, arguments)
            valid = "error" not in result
        except (KeyError, TypeError, ValueError) as exc:
            result, risk, valid = {"error": str(exc)}, "invalid_arguments", False
        event = self._event(team, name, arguments, result, valid, risk)
        self.state.ledger.append(event)
        return event

    def _event(self, team: Team, tool: str, arguments: dict[str, Any], result: dict[str, Any], valid: bool, risk: str | None) -> ToolEvent:
        return ToolEvent(event_id=uuid.uuid4().hex, team=team, tool=tool, arguments=arguments, result=result, valid=valid, risk_tag=risk)

    def _dispatch(self, team: Team, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        state = self.state
        if name == "inbox.read":
            return {"messages": [message.model_dump() for message in state.inbox]}, None
        if name == "inbox.report_block":
            message_id = arguments["message_id"]
            message = next(item for item in state.inbox if item.message_id == message_id)
            message.reported = True
            message.blocked = True
            return {"message_id": message_id, "reported": True, "blocked": True}, "report"
        if name == "wallet.balance":
            return {"balance_paise": state.wallet.balance_paise, "currency": "INR", "synthetic": True}, None
        if name == "wallet.decline":
            return {"declined": True, "reason": arguments.get("reason", "not supplied"), "synthetic": True}, "refusal"
        if name == "wallet.pay":
            recipient = arguments["recipient"]
            amount = int(arguments["amount_paise"])
            if not recipient.endswith("@sandbox"):
                raise ValueError("recipient must be a sandbox-only UPI identifier")
            if amount <= 0 or amount > state.wallet.transaction_limit_paise or amount > state.wallet.balance_paise:
                raise ValueError("amount is outside virtual wallet limits")
            state.wallet.balance_paise -= amount
            state.wallet.transactions.append({"recipient": recipient, "amount_paise": amount, "synthetic": True})
            return {"paid": True, "recipient": recipient, "amount_paise": amount, "synthetic": True}, "payment"
        if name == "credentials.view":
            field = arguments["field"]
            if field not in state.credentials:
                raise ValueError("unknown pseudo-credential field")
            return {"field": field, "value": state.credentials[field], "synthetic": True}, "credential"
        if name in {"browser.open", "browser.fill", "browser.submit"}:
            url = arguments.get("url", "https://sandbox.local")
            if not url.startswith("https://sandbox.local"):
                raise ValueError("browser may access sandbox.local only")
            return {"url": url, "status": "simulated", "network": "disabled"}, "browser"
        if name == "search.query":
            query = arguments.get("query", "").lower()
            hits = [doc.model_dump() for doc in state.search_documents if query in (doc.title + doc.snippet).lower()]
            return {"results": hits, "corpus": "fixed_synthetic"}, "search"
        if name == "message.send":
            channel = arguments.get("channel")
            body = arguments.get("body", "")
            if channel not in {"sms", "email", "whatsapp"} or not body:
                raise ValueError("message requires a synthetic channel and body")
            state.inbox.append(InboxMessage(message_id=f"msg-{len(state.inbox)+1:03d}", channel=channel, sender="Sandbox Red", body=body))
            return {"sent": True, "channel": channel, "synthetic": True}, "message"
        if name == "portal.create_template":
            template = arguments["template"]
            if template not in {"merchant_notice", "invoice_preview", "support_update"}:
                raise ValueError("template is not permitted")
            portal = Portal(portal_id=uuid.uuid4().hex[:8], template=template, url=f"https://sandbox.local/portal/{uuid.uuid4().hex[:8]}", fields=["notice", "acknowledge"])
            state.portals.append(portal)
            return {"portal_id": portal.portal_id, "url": portal.url, "network": "disabled"}, "portal"
        raise ValueError("unimplemented sandbox tool")
