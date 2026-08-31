import importlib.util
from pathlib import Path
import unittest


HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
if HAS_FASTAPI:
    from fastapi.testclient import TestClient
    import vishgym.api.main as api_module


@unittest.skipUnless(HAS_FASTAPI, "FastAPI is an optional dependency")
class VishGymApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api_module.app)

    def test_product_episode_requires_full_runtime(self):
        response = self.client.post("/api/v1/episodes", json={"chain": "vishing_collect", "difficulty": 3, "seed": 19})
        self.assertEqual(response.status_code, 503)
        self.assertIn("VishGym full runtime is unavailable", response.json()["detail"])

    def test_expired_run_deletes_its_audio(self):
        from vishgym.arena.runner import run_local_episode

        state, verdict = run_local_episode(seed=23, scenario_id="vishing_collect")
        run_id = state.episode_id
        api_module._runs[run_id] = {"created_at": 0, "state": state, "verdict": verdict}
        audio_ref = state.audio_turns[0].audio_ref
        path = Path("artifacts/runtime/audio") / Path(audio_ref).name
        self.assertTrue(path.exists())
        original_ttl = api_module._RUN_TTL_SECONDS
        try:
            api_module._RUN_TTL_SECONDS = -1
            self.assertEqual(self.client.get(f"/api/v1/episodes/{run_id}").status_code, 404)
        finally:
            api_module._RUN_TTL_SECONDS = original_ttl
        self.assertFalse(path.exists())

    def test_episode_payload_displays_viewer_safe_synthetic_messages(self):
        from vishgym.arena.runner import run_local_episode

        state, verdict = run_local_episode(seed=29, scenario_id="vishing_collect")
        api_module._runs[state.episode_id] = {"created_at": 9999999999, "state": state, "verdict": verdict}
        payload = self.client.get(f"/api/v1/episodes/{state.episode_id}").json()
        self.assertTrue(payload["viewer_messages_are_synthetic"])
        self.assertFalse(payload["transcript_available_to_agents"])
        self.assertGreater(len(payload["messages"]), 0)
        self.assertEqual(payload["messages"][0]["message"], state.transcript[0]["text"])
        self.assertEqual(payload["audio_turns"][0]["message"], state.transcript[0]["text"])

    def test_missing_seed_uses_fresh_random_episode_seed(self):
        seed_one = api_module._episode_seed(None)
        seed_two = api_module._episode_seed(None)
        self.assertIsInstance(seed_one, int)
        self.assertIsInstance(seed_two, int)
        self.assertNotEqual(seed_one, seed_two)

    def test_model_manifest_reports_strict_full_runtime_readiness(self):
        payload = self.client.get("/api/v1/model").json()
        self.assertEqual(payload["requested_mode"], "auto")
        self.assertIn(payload["selected_mode"], {"unavailable", "full"})
        self.assertFalse(payload["real_reference_audio_accepted"])
        self.assertFalse(payload["transcript_available_to_agents"])

    def test_live_full_mode_requires_reviewed_adapters(self):
        response = self.client.get("/api/v1/live-episodes/stream?mode=full&pace_ms=0")
        self.assertEqual(response.status_code, 503)
        self.assertIn("VishGym full runtime is unavailable", response.json()["detail"])
