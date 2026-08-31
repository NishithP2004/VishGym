import unittest
from pathlib import Path
import tempfile

from vishgym.arena.models import Team
from vishgym.arena.world import VishGymEnv
from vishgym.core.agents import GemmaPolicyHarness
from vishgym.core.prompting import policy_user_content
from vishgym.dev.policies import DeterministicTestPolicy
from vishgym.training.dataset import export_warm_start_dataset, load_training_examples, require_trainable_audio_dataset
from vishgym.training.rollouts import evaluate_blue_policy
from vishgym.training.sft import policy_messages
from vishgym.training.synthetic_data import build_warm_start_examples


class WarmStartDataTests(unittest.TestCase):
    def test_audio_prompt_uses_private_local_path_field(self):
        content = policy_user_content(
            turn_number=1,
            available_tools=["sandbox.inbox.read"],
            own_tools={},
            audio_path=Path("/private/synthetic-turn.wav"),
        )
        self.assertEqual(content[-1], {"type": "audio", "path": "/private/synthetic-turn.wav"})

    def test_training_messages_use_typed_content_blocks_for_every_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            export_warm_start_dataset(temporary, seeds=[37], scenario_ids=["vishing_collect"])
            example = load_training_examples(temporary)[0]
            messages = policy_messages(example, temporary)
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertTrue(all(isinstance(message["content"], list) for message in messages))
        self.assertEqual(messages[0]["content"][0]["type"], "text")
        self.assertEqual(messages[-1]["content"][0]["type"], "text")

    def test_runtime_policy_messages_keep_system_prompt_typed_and_audio_only(self):
        env = VishGymEnv()
        observation = env.reset(seed=53, scenario_id="vishing_collect")
        policy = GemmaPolicyHarness(team=Team.RED, adapter_path="reviewed-synthetic-adapter")
        policy.set_persona(env.state.red_persona)
        messages = policy._messages(observation)
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertTrue(all(isinstance(message["content"], list) for message in messages))
        self.assertEqual(messages[0]["content"][0]["type"], "text")
        # The policy may be told that a hidden transcript exists, but it must
        # never receive it as an input field or content block.
        user_payload = messages[1]["content"][0]["text"]
        self.assertNotIn('"transcript"', user_payload)

    def test_examples_are_audio_first_and_tool_limited(self):
        examples = build_warm_start_examples(seeds=[29], scenario_ids=["vishing_collect"])
        self.assertEqual(len(examples), 8)
        for example in examples:
            allowed = VishGymEnv.RED_TOOLS if example.team == "red" else VishGymEnv.BLUE_TOOLS
            self.assertTrue(set(example.available_tools).issubset(allowed))
            self.assertNotIn("transcript", example.model_dump())

    def test_export_makes_content_addressed_local_audio_and_rejects_test_tones_for_gpu_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            exported = export_warm_start_dataset(temporary, seeds=[31], scenario_ids=["vishing_collect"])
            self.assertEqual(exported.example_count, 8)
            self.assertFalse(exported.audio_training_eligible)
            examples = load_training_examples(temporary)
            self.assertNotIn("transcript", Path(exported.examples_path).read_text(encoding="utf-8"))
            audio_examples = [item for item in examples if item.opponent_audio is not None]
            self.assertEqual(len(audio_examples), 7)
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
            self.assertEqual(len(examples), 32)
            self.assertGreater(len({item.target_spoken_text for item in examples}), 4)
            self.assertTrue(all(item.persona["email"].endswith(".local") for item in examples))

    def test_held_out_blue_evaluation_uses_real_sandbox_outcomes(self):
        result = evaluate_blue_policy(
            blue_policy=DeterministicTestPolicy(Team.BLUE),
            red_policy=DeterministicTestPolicy(Team.RED),
            seeds=[401],
            fraud_scenarios=["vishing_collect"],
            dataset_revision="synthetic-test",
            adapter_revision="developer-blue",
        )
        self.assertEqual(result.report.true_positive, 1)
        self.assertEqual(result.report.legitimate_cases, 1)
        self.assertEqual(result.report.legitimate_false_blocks, 1)
        self.assertGreater(result.report.valid_tool_call_rate, 0.98)
