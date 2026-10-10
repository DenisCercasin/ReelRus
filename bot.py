"""ReelRus Telegram state machine and polling entry point."""

import logging
import os
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.client import RemoteDisconnected
from pathlib import Path
from urllib.error import URLError

from services.conversation_manager import ConversationManager
from services.media import MediaService, VideoDownloadUnavailable
from services.ollama_client import OllamaClient
from services.storage import VideoStore
from services.summarization import SummarizationService
from services.telegram_api import TelegramAPI
from services.video_qa import VideoQAService


LOG = logging.getLogger("reelrus")
URL = re.compile(r"https?://\S+", re.IGNORECASE)
QUESTION_PROMPT = "Что вы хотите узнать об этом видео? Напишите ваш вопрос."


def menu(video_id):
    items = [("📋 Краткое содержание", "summary"), ("💬 Задать вопрос", "qa"),
             ("📝 Перевод текстом", "translation"), ("🎬 Русские субтитры", "subtitles"),
             ("🎙 Русская озвучка", "dub")]
    return {"inline_keyboard": [[{"text": label, "callback_data": action + ":" + video_id}]
                                for label, action in items]}


def poll_updates(api, on_update, prune, sleep=time.sleep):
    """Long-poll Telegram with bounded delay after transient connection failures."""
    offset = None
    backoff = 1
    while True:
        try:
            prune()
            for update in api.get_updates(offset=offset):
                offset = update["update_id"] + 1
                on_update(update)
            backoff = 1
        except (URLError, RemoteDisconnected, TimeoutError, ConnectionResetError) as exc:
            LOG.warning("Telegram polling connection lost (%s); retrying in %ss",
                        type(exc).__name__, backoff)
            sleep(backoff)
            backoff = min(backoff * 2, 30)
        except Exception:
            LOG.exception("Telegram polling failed; retrying in %ss", backoff)
            sleep(backoff)
            backoff = min(backoff * 2, 30)


class ReelRusBot:
    def __init__(self, api, store, media, summary, qa, conversations):
        self.api = api
        self.store = store
        self.media = media
        self.summary = summary
        self.qa = qa
        self.conversations = conversations
        self._chat_locks = {}
        self._chat_locks_guard = threading.Lock()

    def _chat_lock(self, user_id, chat_id):
        key = (user_id, chat_id)
        with self._chat_locks_guard:
            if key not in self._chat_locks:
                self._chat_locks[key] = threading.RLock()
            return self._chat_locks[key]

    def handle_update(self, update):
        callback = update.get("callback_query")
        message = callback.get("message") if callback else update.get("message")
        if not message:
            return
        user = callback.get("from") if callback else message.get("from")
        if not user or "chat" not in message:
            return
        user_id, chat_id = user["id"], message["chat"]["id"]
        with self._chat_lock(user_id, chat_id):
            try:
                if callback:
                    self._on_callback(user_id, chat_id, callback)
                else:
                    self._on_message(user_id, chat_id, message)
            except VideoDownloadUnavailable as exc:
                LOG.warning("Video download unavailable for chat %s: %s", chat_id, exc)
                self.api.send_message(chat_id, str(exc))
            except Exception as exc:
                LOG.exception("Update failed for chat %s", chat_id)
                reason = str(exc) if isinstance(exc, ValueError) else "Попробуйте позже."
                self.api.send_message(chat_id, "Не удалось выполнить запрос: " + reason)

    def _on_message(self, user_id, chat_id, message):
        text = (message.get("text") or "").strip()
        if text == "/start":
            self.api.send_message(chat_id, "Пришлите ссылку на Reel или видео. После распознавания "
                                  "можно получить краткое содержание, задать вопросы, перевести "
                                  "текст, добавить субтитры или озвучку. /videos — ваши видео.")
            return
        if text == "/videos":
            self._list_videos(user_id, chat_id)
            return
        if text.startswith("/video "):
            self._select_video(user_id, chat_id, text.split(maxsplit=1)[1])
            return
        if text == "/cancel":
            self.store.clear_question_state(user_id, chat_id)
            self.api.send_message(chat_id, "Режим вопросов завершён.")
            return
        match = URL.search(text)
        file_info = message.get("video") or message.get("document")
        if match or file_info:
            if (file_info and message.get("document") and
                    not str(file_info.get("mime_type", "")).startswith("video/")):
                self.api.send_message(chat_id, "Пришлите видеофайл или ссылку на видео.")
                return
            self._process_video(user_id, chat_id, match.group(0) if match else None,
                                file_info.get("file_id") if file_info else None)
            return
        state = self.store.get_state(user_id, chat_id)
        video_id = state["question_video_id"]
        if video_id and text:
            answer = self.qa.answer(user_id, chat_id, video_id, text)
            self.api.send_message(chat_id, answer, menu(video_id))
            return
        self.api.send_message(chat_id, "Пришлите видео или ссылку. Для выбора прежнего видео: /videos")

    def _process_video(self, user_id, chat_id, url, file_id):
        video_id = self.store.create_video(user_id, chat_id, url,
                                           {"source": "telegram" if file_id else "url"})
        self.api.send_message(chat_id, "Обрабатываю видео… ID: " + video_id)
        try:
            file_url = self.api.get_file_url(file_id) if file_id else None
            path = self.media.download(video_id, url=url, telegram_file_url=file_url)
            transcript, segments, metadata = self.media.transcribe(path)
            metadata["media_path"] = str(path)
            self.store.update_video(video_id, source_transcript=transcript,
                                    segments_json=segments, metadata_json=metadata,
                                    status="ready")
            self.store.select_video(user_id, chat_id, video_id)
            self.api.send_message(chat_id, "Видео распознано. Выберите действие.\nID: " + video_id,
                                  menu(video_id))
        except Exception as exc:
            self.store.update_video(video_id, status="failed",
                                    metadata_json={"error_type": type(exc).__name__})
            raise

    def _on_callback(self, user_id, chat_id, callback):
        self.api.answer_callback(callback["id"])
        try:
            action, video_id = callback.get("data", "").split(":", 1)
        except ValueError:
            self.api.send_message(chat_id, "Неизвестное действие.")
            return
        video = self.store.get_video(video_id, user_id, chat_id)
        if not video or video["status"] != "ready":
            self.api.send_message(chat_id, "Видео недоступно или срок хранения истёк.")
            return
        if action == "qa":
            self.store.select_video(user_id, chat_id, video_id, questions=True)
            self.api.send_message(chat_id, QUESTION_PROMPT)
            return
        self.store.select_video(user_id, chat_id, video_id)
        if action == "summary":
            self.api.send_message(chat_id, self.summary.summarize(user_id, chat_id, video_id),
                                  menu(video_id))
        elif action == "translation":
            self.api.send_message(chat_id, self._translation(user_id, chat_id, video_id),
                                  menu(video_id))
        elif action in ("subtitles", "dub"):
            self.api.send_message(chat_id, "Готовлю видео…")
            translated = self._translated_segments(user_id, chat_id, video_id)
            path = video["metadata"]["media_path"]
            if action == "subtitles":
                srt = self.media.make_subtitles(video_id, video["segments"], translated)
                output = self.media.render_subtitled_video(video_id, path, srt)
                caption = "Видео с русскими субтитрами"
            else:
                output = self.media.render_dubbed_video(video_id, path, translated)
                caption = "Видео с русской озвучкой"
            self.api.send_video(chat_id, output, caption)
        else:
            self.api.send_message(chat_id, "Неизвестное действие.")

    def _translated_segments(self, user_id, chat_id, video_id):
        with self.conversations.lock_for(user_id, chat_id, video_id):
            video = self.store.get_video(video_id, user_id, chat_id)
            if video["translation"]:
                return video["metadata"]["translated_segments"]
            translated = self.media.translate_segments(video["segments"])
            metadata = dict(video["metadata"])
            metadata["translated_segments"] = translated
            translation = "\n".join(translated)
            self.store.update_video(video_id, translation=translation, metadata_json=metadata)
            return translated

    def _translation(self, user_id, chat_id, video_id):
        self._translated_segments(user_id, chat_id, video_id)
        return self.store.get_video(video_id, user_id, chat_id)["translation"]

    def _list_videos(self, user_id, chat_id):
        videos = self.store.list_videos(user_id, chat_id)
        if not videos:
            self.api.send_message(chat_id, "Сохранённых видео пока нет.")
            return
        lines = ["Ваши видео (выберите /video ID):"]
        for video in videos:
            label = video["original_url"] or "загруженное видео"
            lines.append("/video " + video["id"] + " — " + label[:80])
        self.api.send_message(chat_id, "\n".join(lines))

    def _select_video(self, user_id, chat_id, video_id):
        video = self.store.select_video(user_id, chat_id, video_id)
        if not video:
            self.api.send_message(chat_id, "Видео не найдено или срок хранения истёк.")
            return
        self.api.send_message(chat_id, "Выбрано видео " + video_id, menu(video_id))


def main():
    logging.basicConfig(level=logging.WARNING)
    LOG.setLevel(logging.INFO)
    if sys.version_info < (3, 10):
        raise SystemExit("ReelRus requires Python 3.10+; use a Python 3.12 virtual environment.")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN")
    data_dir = Path(os.environ.get("REELRUS_DATA_DIR", "data"))
    store = VideoStore(data_dir / "reelrus.sqlite",
                       ttl_seconds=int(os.environ.get("VIDEO_TTL_SECONDS", 604800)))
    client = OllamaClient(os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
                          os.environ.get("OLLAMA_MODEL", "qwen3:8b"))
    conversations = ConversationManager(store, int(os.environ.get("QA_HISTORY_MESSAGES", 8)))
    media = MediaService(data_dir / "jobs", client,
                         os.environ.get("WHISPER_MODEL", "small"),
                         os.environ.get("PIPER_MODEL"),
                         os.environ.get("YTDLP_COOKIES_FROM_BROWSER"),
                         os.environ.get("YTDLP_COOKIES_FILE"))
    api = TelegramAPI(token)
    bot = ReelRusBot(api, store, media,
                     SummarizationService(store, client, conversations),
                     VideoQAService(store, client, conversations), conversations)
    queues = {}
    running = set()
    queue_lock = threading.Lock()

    def drain(key):
        while True:
            with queue_lock:
                if not queues[key]:
                    running.remove(key)
                    del queues[key]
                    return
                update = queues[key].popleft()
            try:
                bot.handle_update(update)
            except Exception:
                LOG.exception("Could not handle queued update")

    def submit_ordered(pool, update):
        callback = update.get("callback_query")
        message = callback.get("message") if callback else update.get("message")
        user = callback.get("from") if callback else message.get("from") if message else None
        if not message or not user or "chat" not in message:
            return
        key = (user["id"], message["chat"]["id"])
        with queue_lock:
            queues.setdefault(key, deque()).append(update)
            if key not in running:
                running.add(key)
                pool.submit(drain, key)

    with ThreadPoolExecutor(max_workers=int(os.environ.get("BOT_WORKERS", 4))) as pool:
        def prune():
            for expired_id in store.prune_expired():
                media.delete_job(expired_id)

        try:
            poll_updates(api, lambda update: submit_ordered(pool, update), prune)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
