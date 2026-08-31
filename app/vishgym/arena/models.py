from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Team(str, Enum):
    RED = "red"
    BLUE = "blue"
    JUDGE = "judge"


class EpisodePhase(str, Enum):
    SETUP = "setup"
    ACTIVE = "active"
    TERMINAL = "terminal"


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    spoken_text: str = Field(min_length=1, max_length=600)
    tool_call: ToolCall | None = None

    @field_validator("spoken_text")
    @classmethod
    def no_urls_or_identifiers(cls, value: str) -> str:
        if "http://" in value.lower() or "https://" in value.lower():
            raise ValueError("spoken text cannot contain external URLs")
        return value.strip()


class Persona(BaseModel):
    persona_id: str
    role: Team
    display_name: str
    age_band: Literal["adult", "senior", "young_adult"]
    occupation: str
    email: str
    pseudo_dob: str
    pseudo_identity_ref: str
    voice_speaker: str
    voice_instruction: str


class Wallet(BaseModel):
    balance_paise: int
    currency: Literal["INR"] = "INR"
    transaction_limit_paise: int = 500_000
    transactions: list[dict[str, Any]] = Field(default_factory=list)


class InboxMessage(BaseModel):
    message_id: str
    channel: Literal["sms", "email", "whatsapp"]
    sender: str
    subject: str | None = None
    body: str
    blocked: bool = False
    reported: bool = False


class SearchDocument(BaseModel):
    doc_id: str
    title: str
    snippet: str
    url: str


class Portal(BaseModel):
    portal_id: str
    template: Literal["merchant_notice", "invoice_preview", "support_update"]
    url: str
    fields: list[str]


class AudioTurn(BaseModel):
    turn_id: str
    speaker: Team
    audio_ref: str
    voice_speaker: str
    language: Literal["English"] = "English"
    synthetic: bool = True
    transcript_hidden: bool = True
    tts_model_revision: str = "developer-tone-v1"
    generation_settings: dict[str, Any] = Field(default_factory=dict)


class ToolEvent(BaseModel):
    event_id: str
    team: Team
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    valid: bool
    risk_tag: str | None = None


class EpisodeState(BaseModel):
    episode_id: str
    seed: int
    scenario_id: str
    difficulty: int
    phase: EpisodePhase = EpisodePhase.SETUP
    turn_number: int = 0
    active_team: Team = Team.RED
    red_persona: Persona
    blue_persona: Persona
    wallet: Wallet
    inbox: list[InboxMessage]
    credentials: dict[str, str]
    search_documents: list[SearchDocument]
    portals: list[Portal] = Field(default_factory=list)
    audio_turns: list[AudioTurn] = Field(default_factory=list)
    transcript: list[dict[str, str]] = Field(default_factory=list)
    ledger: list[ToolEvent] = Field(default_factory=list)
    outcome: str | None = None


class AgentObservation(BaseModel):
    episode_id: str
    team: Team
    turn_number: int
    phase: EpisodePhase
    opponent_audio_ref: str | None
    own_tools: dict[str, Any]
    available_tools: list[str]
    transcript: None = None


class JudgeResult(BaseModel):
    red_reward: float
    blue_reward: float
    labels: list[str]
    rationale: str
    terminal_outcome: Literal["safe_defense", "compromise", "false_block", "inconclusive"]


class StepResult(BaseModel):
    observation: AgentObservation | None
    audio_turn: AudioTurn
    tool_event: ToolEvent | None
    done: bool
    judge_result: JudgeResult | None = None
