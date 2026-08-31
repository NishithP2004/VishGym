import importlib.util
import sys
import unittest
from unittest.mock import patch


HAS_MODAL = importlib.util.find_spec("modal") is not None


@unittest.skipUnless(HAS_MODAL, "Modal is an optional dependency")
class ModalLauncherTests(unittest.TestCase):
    def test_launcher_is_importable_without_credentials_and_rejects_unsafe_names(self):
        import modal_vishgym

        self.assertEqual(modal_vishgym.APP_NAME, "vishgym-training")
        self.assertEqual(modal_vishgym._artifact_path("datasets", "warm-start-v1").as_posix(), "/vol/datasets/warm-start-v1")
        self.assertEqual(modal_vishgym._artifact_path("benchmarks", "held-out-v1").as_posix(), "/vol/benchmarks/held-out-v1")
        for name in ("", "../escape", "/absolute", "has space"):
            with self.assertRaises(ValueError):
                modal_vishgym._name(name, label="run")

    def test_qwen_speaker_validation_is_case_insensitive(self):
        """Qwen normalises supported IDs while fixtures retain display casing."""
        from vishgym.arena.audio import QwenCustomVoiceRenderer
        from vishgym.arena.models import Persona, Team

        renderer = QwenCustomVoiceRenderer()
        renderer._model = object()
        renderer._supported_speakers = {"ryan"}
        persona = Persona(
            persona_id="red-test",
            role=Team.RED,
            display_name="Avery Singh",
            age_band="adult",
            occupation="account coordinator",
            email="avery@sandbox.local",
            pseudo_dob="1988-09-02",
            pseudo_identity_ref="SYNTH-RED-TEST",
            voice_speaker="Ryan",
            voice_instruction="clear English",
        )

        # This reaches the deliberately unconfigured model.  A speaker error
        # here would mean the case-normalised membership check regressed.
        with patch.dict(sys.modules, {"soundfile": object()}):
            with self.assertRaises(AttributeError):
                renderer.render(Team.RED, persona, "Synthetic test turn.")
