import importlib.util
import unittest


HAS_OPENENV = importlib.util.find_spec("openenv") is not None
if HAS_OPENENV:
    from fastapi.testclient import TestClient
    from vishgym.arena.openenv_adapter import app


@unittest.skipUnless(HAS_OPENENV, "OpenEnv is an optional training dependency")
class OpenEnvAdapterTests(unittest.TestCase):
    def test_reset_step_and_terminal_reward_are_redacted(self):
        sequence = [
            {"team": "red", "spoken_text": "I am calling about an account notice.", "tool_call": {"name": "search.query", "arguments": {"query": "safety"}}},
            {"team": "blue", "spoken_text": "I will report this message.", "tool_call": {"name": "inbox.report_block", "arguments": {"message_id": "msg-001"}}},
            {"team": "red", "spoken_text": "I can keep explaining the account notice.", "tool_call": {"name": "portal.create_template", "arguments": {"template": "support_update"}}},
            {"team": "blue", "spoken_text": "I need more context before making a payment.", "tool_call": {"name": "wallet.decline", "arguments": {"reason": "unverified"}}},
            {"team": "red", "spoken_text": "The update is time sensitive and I can guide you.", "tool_call": {"name": "message.send", "arguments": {"channel": "sms", "body": "Account notice still pending."}}},
            {"team": "blue", "spoken_text": "I will keep verifying through official channels.", "tool_call": {"name": "inbox.report_block", "arguments": {"message_id": "msg-001"}}},
            {"team": "red", "spoken_text": "Please make a final decision while I am on the line.", "tool_call": {"name": "portal.create_template", "arguments": {"template": "support_update"}}},
            {"team": "blue", "spoken_text": "I am declining the payment and ending this call.", "tool_call": {"name": "wallet.decline", "arguments": {"reason": "unverified"}}},
        ]
        with TestClient(app) as client:
            initial = client.post("/reset", json={"seed": 37})
            self.assertEqual(initial.status_code, 200)
            self.assertEqual(initial.json()["observation"]["team"], "red")
            terminal = None
            for action in sequence:
                terminal = client.post("/step", json={"action": action})
                self.assertEqual(terminal.status_code, 200)
            payload = terminal.json()
            self.assertTrue(payload["done"])
            self.assertGreater(payload["reward"], 0)
            self.assertEqual(payload["observation"]["terminal_outcome"], "safe_defense")
            self.assertNotIn("transcript", terminal.text)
