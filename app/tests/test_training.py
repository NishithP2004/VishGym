import unittest

from vishgym.arena.world import VishGymEnv
from vishgym.training.synthetic_data import build_warm_start_examples


class WarmStartDataTests(unittest.TestCase):
    def test_examples_are_audio_first_and_tool_limited(self):
        examples = build_warm_start_examples(seeds=[29], scenario_ids=["vishing_collect"])
        self.assertEqual(len(examples), 4)
        for example in examples:
            allowed = VishGymEnv.RED_TOOLS if example.team == "red" else VishGymEnv.BLUE_TOOLS
            self.assertTrue(set(example.available_tools).issubset(allowed))
            self.assertNotIn("transcript", example.model_dump())
