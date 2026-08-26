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

    def test_simulation_is_synthetic_and_audio_is_served_from_active_run(self):
        response = self.client.post("/api/v1/simulations", json={"chain": "vishing_collect", "difficulty": 3, "seed": 19})
        self.assertEqual(response.status_code, 200)
        payload = self.client.get(f"/api/v1/simulations/{response.json()['run_id']}").json()
        self.assertFalse(payload["transcript_available_to_agents"])
        self.assertEqual(payload["audio_turns"][0]["tts_model_revision"], "synthetic-fallback-v1")
        audio_response = self.client.get(payload["audio_turns"][0]["audio_ref"])
        self.assertEqual(audio_response.status_code, 200)
        self.assertTrue(audio_response.headers["content-type"].startswith("audio/wav"))

    def test_expired_run_deletes_its_audio(self):
        response = self.client.post("/api/v1/simulations", json={"chain": "vishing_collect", "seed": 23})
        run_id = response.json()["run_id"]
        audio_ref = self.client.get(f"/api/v1/simulations/{run_id}").json()["audio_turns"][0]["audio_ref"]
        path = Path("artifacts/runtime/audio") / Path(audio_ref).name
        self.assertTrue(path.exists())
        original_ttl = api_module._RUN_TTL_SECONDS
        try:
            api_module._RUN_TTL_SECONDS = -1
            self.assertEqual(self.client.get(f"/api/v1/simulations/{run_id}").status_code, 404)
        finally:
            api_module._RUN_TTL_SECONDS = original_ttl
        self.assertFalse(path.exists())
