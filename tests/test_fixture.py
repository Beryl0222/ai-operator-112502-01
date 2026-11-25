import json
from pathlib import Path
import unittest


class FixtureContractTest(unittest.TestCase):
    def test_paged_attention_fixture_has_distinct_cache_boundaries(self):
        base = Path(__file__).parents[1]
        payload = json.loads((base / "data/input.json").read_text(encoding="utf-8"))
        requests = payload["requests"]
        self.assertEqual([request["request_id"] for request in requests], ["chat-a", "chat-b"])
        self.assertNotEqual(requests[0]["prefix_length"], requests[1]["prefix_length"])
        self.assertTrue(any(page["valid_tokens"] < payload["page_size"] for page in payload["pages"].values()))

if __name__ == "__main__":
    unittest.main()
