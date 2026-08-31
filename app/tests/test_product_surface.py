from pathlib import Path
import unittest


class ProductSurfaceTests(unittest.TestCase):
    def test_product_source_avoids_fake_scenario_language(self):
        forbidden = ("fictional", "simulation", "simulated", "scripted", "fallback")
        roots = [
            Path("app/vishgym/api"),
            Path("app/vishgym/core"),
            Path("app/vishgym/ui"),
        ]
        for root in roots:
            for path in root.rglob("*.py"):
                text = path.read_text(encoding="utf-8").lower()
                for term in forbidden:
                    self.assertNotIn(term, text, f"{path} contains {term!r}")
