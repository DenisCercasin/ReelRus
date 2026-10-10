"""Small local checks for dependency and connection regressions."""

import importlib.util
import tempfile
import unittest
import wave
from http.client import RemoteDisconnected
from pathlib import Path
from urllib.error import URLError

from bot import poll_updates


class PollingTests(unittest.TestCase):
    def test_transient_disconnects_back_off_and_preserve_offset(self):
        class FakeAPI:
            def __init__(self):
                self.offsets = []

            def get_updates(self, offset=None):
                self.offsets.append(offset)
                if len(self.offsets) == 1:
                    raise RemoteDisconnected("closed")
                if len(self.offsets) == 2:
                    raise URLError(ConnectionResetError("reset"))
                if len(self.offsets) == 3:
                    return [{"update_id": 42, "message": {}}]
                raise KeyboardInterrupt

        api = FakeAPI()
        delays, received = [], []
        with self.assertLogs("reelrus", level="WARNING") as logs:
            with self.assertRaises(KeyboardInterrupt):
                poll_updates(api, received.append, lambda: None, sleep=delays.append)
        self.assertEqual([1, 2], delays)
        self.assertEqual([None, None, None, 43], api.offsets)
        self.assertEqual(42, received[0]["update_id"])
        self.assertEqual(2, len(logs.output))


@unittest.skipUnless(importlib.util.find_spec("faster_whisper"), "media dependencies not installed")
class AudioDependencyTests(unittest.TestCase):
    def test_faster_whisper_can_decode_audio_with_installed_pyav(self):
        from faster_whisper.audio import decode_audio

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silence.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\x00\x00" * 1600)
            samples = decode_audio(str(path))
        self.assertEqual(1600, len(samples))


if __name__ == "__main__":
    unittest.main()
