import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from services.media import MediaService, VideoDownloadUnavailable


class FakeDownloadError(Exception):
    pass


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.options = []
        self.failure = None
        test = self

        class FakeYoutubeDL:
            def __init__(self, options):
                test.options.append(options)
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def extract_info(self, url, download):
                if test.failure:
                    raise FakeDownloadError(test.failure)
                Path(self.options["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"video")
                return {"id": "reel"}

            def prepare_filename(self, info):
                return self.options["outtmpl"].replace("%(ext)s", "mp4")

        yt_dlp = types.ModuleType("yt_dlp")
        yt_dlp.YoutubeDL = FakeYoutubeDL
        yt_dlp_utils = types.ModuleType("yt_dlp.utils")
        yt_dlp_utils.DownloadError = FakeDownloadError
        self.modules = patch.dict("sys.modules", {"yt_dlp": yt_dlp,
                                                  "yt_dlp.utils": yt_dlp_utils})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()
        self.tmp.cleanup()

    def service(self, **kwargs):
        return MediaService(self.tmp.name, client=None, **kwargs)

    def test_public_download_uses_no_cookies(self):
        path = self.service().download("a" * 32, url="https://example.com/reel")
        self.assertEqual(b"video", path.read_bytes())
        self.assertNotIn("cookiesfrombrowser", self.options[0])
        self.assertNotIn("cookiefile", self.options[0])

    def test_configured_browser_cookies_are_passed_to_ytdlp(self):
        self.service(cookies_from_browser="Chrome").download(
            "b" * 32, url="https://www.instagram.com/reel/example/")
        self.assertEqual(("chrome",), self.options[0]["cookiesfrombrowser"])

    def test_configured_cookie_file_is_passed_to_ytdlp(self):
        path = Path(self.tmp.name) / "private-cookies.txt"
        path.write_text("# Netscape HTTP Cookie File\n")
        self.service(cookies_file=str(path)).download("c" * 32, url="https://example.com")
        self.assertEqual(str(path), self.options[0]["cookiefile"])

    def test_login_error_gives_actionable_message(self):
        self.failure = "Requested content is not available, rate-limit reached or login required"
        with self.assertRaises(VideoDownloadUnavailable) as caught:
            self.service().download("d" * 32, url="https://www.instagram.com/reel/example/")
        self.assertIn("YTDLP_COOKIES_FROM_BROWSER", str(caught.exception))
        self.assertIn("Telegram", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
