# ReelRus

A Telegram bot for English-language Reels. It transcribes speech locally, translates it into Russian, and answers questions about the video in Russian.

After transcription, the bot offers a summary, questions, text translation, Russian subtitles, and Russian voiceover through its buttons. Questions, follow-ups, and summaries use the saved transcript, so the video does not need to be downloaded or transcribed again.

## Setup

Use Python 3.12, `ffmpeg`, a running [Ollama](https://ollama.com/) instance with Qwen, and a Telegram bot token. Voiceover also requires `piper` and a Russian Piper model. The first transcription may download a Whisper model. Current versions of yt-dlp no longer support Python 3.9.

After changing `requirements.txt`, run `.venv312/bin/python -m pip install -U -r requirements.txt` again and restart the bot. PyAV is pinned below version 19 for compatibility with the current faster-whisper release.

```sh
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -U -r requirements.txt
ollama pull qwen3:8b
export TELEGRAM_BOT_TOKEN='...'
.venv312/bin/python bot.py
```

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Local Ollama server |
| `OLLAMA_MODEL` | `qwen3:8b` | Qwen model |
| `WHISPER_MODEL` | `small` | faster-whisper model |
| `PIPER_MODEL` | None | Path to the Russian Piper model for voiceover |
| `REELRUS_DATA_DIR` | `data` | SQLite database and video working files |
| `VIDEO_TTL_SECONDS` | `604800` | How long videos and conversations are kept |
| `QA_HISTORY_MESSAGES` | `8` | Number of recent messages in the video context |
| `BOT_WORKERS` | `4` | Number of chats processed concurrently |
| `YTDLP_COOKIES_FROM_BROWSER` | None | Browser with an active Instagram login, such as `chrome`, `safari`, or `firefox` |
| `YTDLP_COOKIES_FILE` | None | Alternative path to a Netscape-format cookies file |

Send the bot a Reel link or a video file. `/videos` lists available videos, `/video <ID>` returns to a video, and `/cancel` exits question mode. The question button enables free-form questions about the selected video. Each video has its own ID, conversation history, and expiration time.

If Instagram responds with “login required” or “rate-limit reached,” sign in to Instagram in your browser and restart the bot with `YTDLP_COOKIES_FROM_BROWSER`:

```sh
export YTDLP_COOKIES_FROM_BROWSER=chrome
.venv312/bin/python bot.py
```

Use the browser where you are already signed in to Instagram. Alternatively, set `YTDLP_COOKIES_FILE` to a private cookies file outside the repository. The bot does not load `.env` automatically; if you use one, run `set -a; source .env; set +a` before starting it. Cookies are not sent to Telegram. If your account cannot access a particular Reel or Instagram continues to limit requests, send the video file directly to the bot.

Video text is treated as untrusted data. Answers and timestamps are based on saved transcript segments. If the model cannot find supporting information, the bot says that the video does not contain it. SQLite data and media files are deleted after their retention period expires.

## Tests

```sh
.venv312/bin/python -m unittest discover -v
REELRUS_LIVE_OLLAMA=1 .venv312/bin/python -m unittest tests.test_ollama_integration -v
```

The first command uses mocks throughout. The second connects to a running local Ollama instance with the Qwen model installed.
