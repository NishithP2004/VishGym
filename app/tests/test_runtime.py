import tempfile
import unittest
from unittest.mock import patch

from vishgym.api.runtime import runtime_status


class RuntimeStatusTests(unittest.TestCase):
    def test_auto_reports_unavailable_when_reviewed_adapters_are_not_mounted(self):
        with patch.dict(
            "os.environ",
            {
                "VISHGYM_RED_ADAPTER_PATH": "",
                "VISHGYM_BLUE_ADAPTER_PATH": "",
                "VISHGYM_JUDGE_ADAPTER_PATH": "",
            },
            clear=False,
        ):
            status = runtime_status("auto")
        self.assertEqual(status.selected_mode, "unavailable")
        self.assertFalse(status.full_runtime_ready)
        self.assertFalse(status.real_reference_audio_accepted)
        self.assertFalse(status.transcript_available_to_agents)

    def test_full_readiness_requires_all_three_distinct_role_adapters(self):
        with tempfile.TemporaryDirectory() as red, tempfile.TemporaryDirectory() as blue:
            with patch.dict(
                "os.environ",
                {
                    "VISHGYM_RED_ADAPTER_PATH": red,
                    "VISHGYM_BLUE_ADAPTER_PATH": blue,
                    "VISHGYM_JUDGE_ADAPTER_PATH": "/not/mounted",
                },
                clear=False,
            ):
                status = runtime_status("full")
        self.assertEqual(status.selected_mode, "unavailable")
        self.assertFalse(status.full_runtime_ready)
        self.assertTrue(any("judge adapter path is not mounted" in reason for reason in status.reasons))
