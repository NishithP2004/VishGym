from __future__ import annotations

from dataclasses import dataclass

from vishgym.arena.models import AgentAction, AgentObservation, Team, ToolCall


@dataclass
class DeterministicTestPolicy:
    """Developer-only policy for local environment tests."""

    team: Team

    def act(self, observation: AgentObservation) -> AgentAction:
        if self.team is Team.RED:
            if observation.turn_number == 0:
                return AgentAction(
                    spoken_text="I am calling from the payment support desk about an account update that needs attention.",
                    tool_call=ToolCall(name="message.send", arguments={"channel": "sms", "body": "Payment support notice for account review."}),
                )
            return AgentAction(
                spoken_text="Please review the account notice while I stay on the line and answer any questions.",
                tool_call=ToolCall(name="portal.create_template", arguments={"template": "support_update"}),
            )
        if observation.turn_number < 3:
            return AgentAction(
                spoken_text="I will verify this through my available tools and report anything suspicious.",
                tool_call=ToolCall(name="inbox.report_block", arguments={"message_id": "msg-001"}),
            )
        return AgentAction(
            spoken_text="I am declining this payment until the request is independently verified.",
            tool_call=ToolCall(name="wallet.decline", arguments={"reason": "unverified request"}),
        )
