"""Opt-in smoke test against an installed local Qwen model."""

import os
import tempfile
import unittest
from pathlib import Path

from services.conversation_manager import ConversationManager
from services.ollama_client import OllamaClient
from services.storage import VideoStore
from services.summarization import SummarizationService
from services.video_qa import VideoQAService


@unittest.skipUnless(os.environ.get("REELRUS_LIVE_OLLAMA") == "1", "local Ollama opt-in")
class OllamaIntegrationTests(unittest.TestCase):
    def test_summary_and_question_on_short_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VideoStore(Path(directory) / "test.sqlite")
            video_id = store.create_video(1, 10)
            store.update_video(video_id, source_transcript="Add two tablespoons of sugar.",
                               segments_json=[{"start": 34, "end": 39,
                                               "text": "Add two tablespoons of sugar."}],
                               status="ready")
            client = OllamaClient(model=os.environ.get("OLLAMA_MODEL", "qwen3:8b"))
            conversations = ConversationManager(store)
            summary = SummarizationService(store, client, conversations).summarize(1, 10, video_id)
            qa = VideoQAService(store, client, conversations)
            answer = qa.answer(
                1, 10, video_id, "Сколько сахара нужно добавить?")
            unsupported = qa.answer(1, 10, video_id, "Сколько нужно добавить яиц?")
            self.assertIn("сахар", summary.lower())
            self.assertRegex(answer.lower(), r"два|две")
            self.assertIn("00:34", answer)
            self.assertIn("нет информации", unsupported)


if __name__ == "__main__":
    unittest.main()
