from types import SimpleNamespace
import unittest
from unittest import mock

from trnscrb import enricher


class _FakeAdapter:
    def __init__(self):
        self.last_prompt = ""
        self.last_model = ""

    def test_connection(self, config):
        return True, "ok"

    def list_models(self, config):
        return ["model-a", "model-b"]

    def enrich(self, prompt, config):
        self.last_prompt = prompt
        self.last_model = config["model"]
        return (
            "SUMMARY:\nA\n\nACTION ITEMS:\n- B\n\nSPEAKER MAPPING:\n- SPEAKER_00 → Alex"
        )


class EnricherTests(unittest.TestCase):
    def test_normalize_endpoint_adds_v1_for_openai_compatible_provider(self):
        self.assertEqual(
            enricher.normalize_endpoint("llama_cpp", "http://127.0.0.1:8080"),
            "http://127.0.0.1:8080/v1",
        )
        self.assertEqual(
            enricher.normalize_endpoint("openai", "https://api.openai.com/v1"),
            "https://api.openai.com/v1",
        )

    def test_ollama_model_parsing(self):
        payload = b'{"models":[{"name":"llama3.2"},{"name":"qwen2.5"}]}'
        fake_response = SimpleNamespace(read=lambda: payload)
        adapter = enricher.OllamaAdapter()

        with mock.patch("urllib.request.urlopen", return_value=fake_response):
            models = adapter.list_models({"endpoint": "http://127.0.0.1:11434"})

        self.assertEqual(models, ["llama3.2", "qwen2.5"])

    def test_enrich_dispatches_to_selected_provider_model(self):
        fake = _FakeAdapter()
        with (
            mock.patch.dict(enricher._ADAPTERS, {"llama_cpp": fake}, clear=False),
            mock.patch(
                "trnscrb.settings.load",
                return_value={
                    "enrich": {
                        "provider": "llama_cpp",
                        "profiles": {
                            "llama_cpp": {
                                "endpoint": "http://127.0.0.1:8080",
                                "api_key": "",
                                "model": "qwen2.5",
                                "models": ["qwen2.5"],
                            }
                        },
                    }
                },
            ),
        ):
            result = enricher.enrich_transcript("[SPEAKER_00] hello")

        self.assertEqual(fake.last_model, "qwen2.5")
        self.assertIn("Alex", result["enriched_transcript"])

    def test_openai_compatible_model_parsing(self):
        adapter = enricher.OpenAICompatibleAdapter(provider="llama_cpp")
        fake_client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: SimpleNamespace(
                    data=[SimpleNamespace(id="m1"), SimpleNamespace(id="m2")]
                )
            )
        )
        with mock.patch(
            "trnscrb.enricher._build_openai_client", return_value=fake_client
        ):
            models = adapter.list_models(
                {"endpoint": "http://127.0.0.1:8080", "api_key": ""}
            )

        self.assertEqual(models, ["m1", "m2"])


if __name__ == "__main__":
    unittest.main()
