import unittest
from pathlib import Path
import tempfile

from vishgym.arena.models import Team
from vishgym.arena.world import VishGymEnv
from vishgym.core.agents import ScriptedPolicy
from vishgym.training.dataset import export_warm_start_dataset, load_training_examples, require_trainable_audio_dataset
from vishgym.training.rollouts import evaluate_blue_policy
from vishgym.training.synthetic_data import build_warm_start_examples


class WarmStartDataTests(unittest.TestCase):
    def test_examples_are_audio_first_and_tool_limited(self):
        examples = build_warm_start_examples(seeds=[29], scenario_ids=["vishing_collect"])
        self.assertEqual(len(examples), 4)
        for example in examples:
            allowed = VishGymEnv.RED_TOOLS if example.team == "red" else VishGymEnv.BLUE_TOOLS
            self.assertTrue(set(example.available_tools).issubset(allowed))
            self.assertNotIn("transcript", example.model_dump())

    def test_export_makes_content_addressed_local_audio_and_rejects_test_tones_for_gpu_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            exported = export_warm_start_dataset(temporary, seeds=[31], scenario_ids=["vishing_collect"])
            self.assertEqual(exported.example_count, 4)
            self.assertFalse(exported.audio_training_eligible)
            examples = load_training_examples(temporary)
            self.assertNotIn("transcript", Path(exported.examples_path).read_text(encoding="utf-8"))
            audio_examples = [item for item in examples if item.opponent_audio is not None]
            self.assertEqual(len(audio_examples), 3)
            for item in audio_examples:
                assert item.opponent_audio is not None
                self.assertTrue((Path(temporary) / item.opponent_audio.path).is_file())
                self.assertFalse(item.opponent_audio.path.startswith("/api/"))
            for item in examples:
                self.assertTrue((Path(temporary) / item.emitted_audio.path).is_file())
                self.assertEqual(item.emitted_audio.language, "English")
                self.assertIn("sample_rate", item.emitted_audio.generation_settings)
            with self.assertRaisesRegex(ValueError, "deterministic test tones"):
                require_trainable_audio_dataset(temporary)

    def test_export_varies_benign_synthetic_labels_by_seed_and_scenario(self):
        with tempfile.TemporaryDirectory() as temporary:
            export_warm_start_dataset(
                temporary,
                seeds=[41, 43],
                scenario_ids=["vishing_collect", "smishing_link"],
            )
            examples = load_training_examples(temporary)
            self.assertEqual(len(examples), 16)
            self.assertGreater(len({item.target_spoken_text for item in examples}), 4)
            self.assertTrue(all(item.persona["email"].endswith("@sandbox.local") for item in examples))

    def test_held_out_blue_evaluation_uses_real_sandbox_outcomes(self):
        result = evaluate_blue_policy(
            blue_policy=ScriptedPolicy(Team.BLUE),
            red_policy=ScriptedPolicy(Team.RED),
            seeds=[401],
            fraud_scenarios=["vishing_collect"],
            dataset_revision="synthetic-test",
            adapter_revision="scripted-blue",
        )
        self.assertEqual(result.report.true_positive, 1)
        self.assertEqual(result.report.legitimate_cases, 1)
        self.assertEqual(result.report.legitimate_false_blocks, 1)
        self.assertGreater(result.report.valid_tool_call_rate, 0.98)
